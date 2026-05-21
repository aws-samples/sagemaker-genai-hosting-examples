"""Deploy monitor."""

import io
import sys
import time

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, RichLog, Static


STAGES = ["Validate", "Create Model", "Endpoint Config", "Create Endpoint",
          "Wait InService", "Create IC", "Wait IC", "Smoke Test"]


class DeployScreen(Screen):
    BINDINGS = [("h", "app.switch_mode('home')", "Home")]

    DEFAULT_CSS = """
    #dp-inputs { height: auto; padding: 1 0; }
    #dp-inputs Input { width: 1fr; }
    #dp-stages { height: auto; padding: 0 1; }
    #dp-log { height: 1fr; }
    #dp-status { dock: bottom; height: 1; padding: 0 1; }
    """

    def __init__(self):
        super().__init__()
        self._deploying = False
        self._stage = -1

    def compose(self) -> ComposeResult:
        with Horizontal(id="dp-inputs"):
            yield Input(placeholder="Recipe YAML path", id="dp-recipe")
            yield Button("Deploy", id="dp-go", variant="success")
            yield Button("Cleanup", id="dp-rm", variant="error")
        yield Static(self._render_stages(), id="dp-stages")
        yield RichLog(id="dp-log", highlight=True, markup=True)
        yield Static("Ready", id="dp-status")
        yield Footer()

    def _render_stages(self):
        lines = []
        for i, s in enumerate(STAGES):
            if i < self._stage:
                lines.append(f"  [green]OK[/green] {s}")
            elif i == self._stage:
                lines.append(f"  [yellow]>>[/yellow] {s}...")
            else:
                lines.append(f"  -- {s}")
        return "\n".join(lines)

    def _set_stage(self, n):
        self._stage = n
        self.query_one("#dp-stages", Static).update(self._render_stages())

    def on_button_pressed(self, event: Button.Pressed):
        recipe = self.query_one("#dp-recipe", Input).value.strip()
        if not recipe:
            self.notify("Recipe path required", severity="error")
            return
        if event.button.id == "dp-go" and not self._deploying:
            self._deploying = True
            self._stage = -1
            self.query_one("#dp-go", Button).disabled = True
            self.query_one("#dp-log", RichLog).clear()
            self._set_stage(0)
            self._run_deploy(recipe)
        elif event.button.id == "dp-rm":
            self._run_cleanup(recipe)

    @work(thread=True)
    def _run_deploy(self, recipe_path):
        log = lambda t: self.app.call_from_thread(
            self.query_one("#dp-log", RichLog).write, t)
        stage = lambda n: self.app.call_from_thread(self._set_stage, n)

        try:
            from scripts.config_loader import load_config, validate_config
            from scripts.deployer import deploy, smoke_test
        except ImportError as e:
            log(f"[red]Import error: {e}[/red]")
            self.app.call_from_thread(self._done, False)
            return

        try:
            cfg = load_config(recipe_path)
            for w in validate_config(cfg):
                log(f"[yellow]WARN: {w}[/yellow]")
            stage(1)
        except Exception as e:
            log(f"[red]Config error: {e}[/red]")
            self.app.call_from_thread(self._done, False)
            return

        log("Deploying...")
        old = sys.stdout
        buf = io.StringIO()

        class Tee:
            def write(_, t):
                buf.write(t)
                if t.strip():
                    log(t.strip())
                    self._detect(t.strip())
            def flush(_):
                buf.flush()

        sys.stdout = Tee()
        try:
            result = deploy(cfg)
            sys.stdout = old
            if result.success:
                log(f"[green]SUCCESS: {result.endpoint_name} ({result.elapsed_sec:.0f}s)[/green]")
                stage(7)
                try:
                    smoke_test(cfg, result.endpoint_name, result.ic_name)
                except Exception:
                    pass
            else:
                log(f"[red]FAILED: {result.endpoint_name}[/red]")
            self.app.call_from_thread(self._done, result.success)
        except Exception as e:
            sys.stdout = old
            log(f"[red]Error: {e}[/red]")
            self.app.call_from_thread(self._done, False)

    def _detect(self, t):
        lo = t.lower()
        for kw, n in [("creating sagemaker model", 1), ("model created", 2),
                       ("endpoint config", 2), ("creating endpoint", 3),
                       ("endpoint creating", 3), ("waiting for", 4),
                       ("inference component", 5), ("ic creating", 5),
                       ("waiting for ic", 6), ("endpoint deployed", 7)]:
            if kw in lo and n > self._stage:
                self.app.call_from_thread(self._set_stage, n)
                break

    @work(thread=True)
    def _run_cleanup(self, recipe_path):
        log = lambda t: self.app.call_from_thread(
            self.query_one("#dp-log", RichLog).write, t)
        try:
            from scripts.config_loader import load_config, build_endpoint_name
            from scripts.deployer import cleanup
            cfg = load_config(recipe_path)
            name = build_endpoint_name(cfg)
            log(f"Cleaning up: {name}")
            cleanup(name, cfg.deployment.endpoint.region, cfg.deployment.endpoint.pattern)
            log(f"[green]Done: {name}[/green]")
        except Exception as e:
            log(f"[red]Error: {e}[/red]")

    def _done(self, ok):
        self._deploying = False
        self.query_one("#dp-go", Button).disabled = False
        self.query_one("#dp-status", Static).update(
            "[green]Complete[/green]" if ok else "[red]Failed[/red]")
