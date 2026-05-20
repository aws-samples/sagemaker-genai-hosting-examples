"""Benchmark dashboard."""

import time

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Input, Select, Static


class BenchmarkScreen(Screen):
    BINDINGS = [
        ("h", "app.switch_mode('home')", "Home"),
        ("escape", "stop_benchmark", "Stop"),
    ]

    DEFAULT_CSS = """
    #bm-inputs { height: auto; padding: 1 0; }
    #bm-inputs Input { width: 1fr; }
    #bm-inputs Select { width: 28; }
    #bm-status { dock: bottom; height: 1; padding: 0 1; }
    """

    def __init__(self):
        super().__init__()
        self._running = False
        self._cancel = False
        self._t0 = None
        self._done = 0
        self._total = 0

    def compose(self) -> ComposeResult:
        with Horizontal(id="bm-inputs"):
            yield Input(placeholder="Endpoint name", id="ep")
            yield Input(placeholder="IC (optional)", id="ic")
            yield Input(placeholder="Recipe YAML", id="recipe")
            yield Select(
                [("All", "all"), ("Chat", "multiturn_chat"),
                 ("Tool", "tool_calling"), ("Long", "long_context")],
                value="all", id="uc",
            )
            yield Button("Start", id="go", variant="success")
            yield Button("Stop", id="stop", variant="error", disabled=True)
        yield DataTable(id="tbl")
        yield Static("Ready", id="bm-status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#tbl", DataTable).add_columns(
            "Use Case", "C", "p50ms", "p99ms", "TTFT", "tok/s", "RPS", "Agg", "Valid",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "go":
            ep = self.query_one("#ep", Input).value.strip()
            rcp = self.query_one("#recipe", Input).value.strip()
            if not ep or not rcp:
                self.notify("Endpoint and recipe required", severity="error")
                return
            ic = self.query_one("#ic", Input).value.strip() or None
            uc = self.query_one("#uc", Select).value
            self._running = True
            self._cancel = False
            self._t0 = time.time()
            self._done = 0
            self.query_one("#go", Button).disabled = True
            self.query_one("#stop", Button).disabled = False
            self.query_one("#tbl", DataTable).clear()
            self._run(ep, rcp, ic, uc)
        elif event.button.id == "stop":
            self._cancel = True

    @work(thread=True)
    def _run(self, endpoint, recipe_path, ic_name, uc_filter):
        import boto3
        from botocore.config import Config as BC
        from scripts.config_loader import load_config
        from scripts.benchmarker import _run_concurrency_level, warmup
        from scripts.prompts import USE_CASE_PROMPTS, USE_CASE_DESCRIPTIONS

        try:
            cfg = load_config(recipe_path)
        except Exception as e:
            self.app.call_from_thread(self._status, f"Config error: {e}")
            self.app.call_from_thread(self._finish)
            return

        region = cfg.deployment.endpoint.region
        client = boto3.client("sagemaker-runtime", region_name=region,
                              config=BC(read_timeout=300, retries={"max_attempts": 0}))
        bp = cfg.benchmark
        ucs = bp.use_cases if uc_filter == "all" else [uc_filter]
        levels = bp.concurrency_levels
        self._total = len(ucs) * len(levels)
        errs = 0

        for ui, uc in enumerate(ucs):
            if self._cancel or uc not in USE_CASE_PROMPTS:
                continue
            desc = USE_CASE_DESCRIPTIONS.get(uc, uc)
            self.app.call_from_thread(self._status, f"Warmup: {desc}")
            try:
                warmup(client, endpoint, uc, bp.inference_params, ic_name,
                       num_requests=bp.warmup_requests, streaming=bp.streaming)
            except Exception:
                pass

            for ci, c in enumerate(levels):
                if self._cancel:
                    break
                self.app.call_from_thread(self._status,
                    f"{desc} C={c}  ({self._done}/{self._total})")
                try:
                    s = _run_concurrency_level(cfg, client, endpoint, uc, c,
                                               bp.requests_per_level, ic_name)
                    errs += s.get("failed", 0)
                    self.app.call_from_thread(self._row, uc, s)
                except Exception as e:
                    self.app.call_from_thread(self._err_row, uc, c, str(e))
                    errs += 1
                self._done += 1
                if c != levels[-1]:
                    time.sleep(bp.pause_between_levels_sec)

        self.app.call_from_thread(self._finish)

    def _row(self, uc, s):
        ttft = s.get("ttft_p50")
        self.query_one("#tbl", DataTable).add_row(
            uc, str(s.get("concurrency", "")),
            f"{s.get('latency_p50',0):.0f}", f"{s.get('latency_p99',0):.0f}",
            f"{ttft:.0f}" if ttft else "-",
            f"{s.get('tok_per_sec_avg',0):.1f}", f"{s.get('rps',0):.2f}",
            f"{s.get('aggregate_output_tok_sec',0):.0f}",
            f"{s.get('output_validation_rate',0):.0%}",
        )

    def _err_row(self, uc, c, err):
        self.query_one("#tbl", DataTable).add_row(
            uc, str(c), "ERR", "-", "-", "-", "-", "-", err[:25])

    def _status(self, txt):
        el = int(time.time() - self._t0) if self._t0 else 0
        self.query_one("#bm-status", Static).update(f"{txt}  [{el}s]")

    def _finish(self):
        self._running = False
        self.query_one("#go", Button).disabled = False
        self.query_one("#stop", Button).disabled = True
        el = int(time.time() - self._t0) if self._t0 else 0
        tag = "Cancelled" if self._cancel else "Done"
        self.query_one("#bm-status", Static).update(f"{tag} {self._done}/{self._total} [{el}s]")
        self._cancel = False

    def action_stop_benchmark(self):
        if self._running:
            self._cancel = True
