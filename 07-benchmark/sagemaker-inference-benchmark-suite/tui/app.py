"""SageMaker Benchmark Suite TUI — single-screen tabbed interface."""

from __future__ import annotations

import csv
import glob
import os
import subprocess

import yaml
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Input, RichLog, Static, Tree


# ── Data loaders ─────────────────────────────────────────────────────────

def _load_recipes(d="recipes"):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.yaml"))):
        try:
            with open(p) as f:
                r = yaml.safe_load(f)
            dep = r.get("deployment", {})
            spec = dep.get("speculative_decoding", {})
            opt = ("eagle3" if spec.get("enabled") else
                   "lmcache" if dep.get("vllm", {}).get("swap_space") else
                   "prefix_cache" if dep.get("prefix_caching", {}).get("enabled") else
                   "vanilla")
            out.append({
                "file": os.path.basename(p), "path": p,
                "model": dep.get("model", {}).get("id", "?"),
                "instance": dep.get("instance", {}).get("type", "?"),
                "opt": opt,
                "container": dep.get("container", {}).get("type", "?"),
                "pipeline": ", ".join(r.get("pipeline", [])),
                "cost": r.get("cost", {}).get("instance_cost_per_hour"),
            })
        except Exception:
            continue
    return out


def _load_results(d="results/matrix"):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
        try:
            with open(p, newline="") as f:
                for row in csv.DictReader(f):
                    for k in ("concurrency", "successful", "failed"):
                        if row.get(k):
                            try: row[k] = int(row[k])
                            except ValueError: pass
                    for k in ("latency_p50", "latency_p99", "rps",
                              "aggregate_output_tok_sec", "tok_per_sec_avg",
                              "output_validation_rate", "ttft_p50"):
                        if row.get(k):
                            try: row[k] = float(row[k])
                            except ValueError: pass
                    rows.append(row)
        except Exception:
            continue
    return rows


def _sparkline(vals):
    blocks = " ▁▂▃▄▅▆▇█"
    if not vals: return ""
    mn, mx = min(vals), max(vals)
    if mn == mx: return blocks[4] * len(vals)
    return "".join(blocks[1 + int((v - mn) / (mx - mn) * 7)] for v in vals)


def _f(v, d=1):
    if v is None or v == "" or v == "None": return "-"
    try: return f"{float(v):.{d}f}"
    except (ValueError, TypeError): return str(v)[:15]


# ── Constants ────────────────────────────────────────────────────────────

TABS = ["Recipes", "Results", "Endpoints", "Benchmark", "Deploy"]


class _VS:
    TABLE = "table"
    SEARCH = "search"
    ACTION = "action"
    OUTPUT = "output"


# ── Main App ─────────────────────────────────────────────────────────────

class BenchmarkApp(App):
    TITLE = "SageMaker Benchmark Suite"

    CSS = """
    #tab-bar { dock: top; height: 1; background: $surface-darken-1; }
    #search-bar { dock: top; height: 3; display: none; }
    #output-log { display: none; height: 1fr; }
    #result-tree { display: none; height: 1fr; }
    #detail { dock: bottom; height: 3; padding: 0 1; }
    #status { dock: bottom; height: 1; background: $surface-darken-2; padding: 0 1; }
    DataTable { height: 1fr; }
    """

    BINDINGS = [
        Binding("left", "prev_tab", "<<", priority=True),
        Binding("right", "next_tab", ">>", priority=True),
        Binding("tab", "next_tab", "", show=False, priority=True),
        Binding("shift+tab", "prev_tab", "", show=False, priority=True),
        Binding("slash", "start_search", "/Filter"),
        Binding("escape", "handle_esc", "Esc", priority=True),
        Binding("r", "refresh", "Refresh", show=False),
        Binding("1", "go_tab(0)", show=False),
        Binding("2", "go_tab(1)", show=False),
        Binding("3", "go_tab(2)", show=False),
        Binding("4", "go_tab(3)", show=False),
        Binding("5", "go_tab(4)", show=False),
    ]

    def __init__(self):
        super().__init__()
        self._tab = 0
        self._vs = _VS.TABLE
        self._recipes: list[dict] = []
        self._results: list[dict] = []
        self._endpoints: list[dict] = []
        self._filtered: list[dict] = []
        self._query = ""
        self._esc_armed = False
        self._esc_timer = None
        self._action_idx = -1
        self._cmd_proc = None

    def compose(self) -> ComposeResult:
        yield Static(self._tab_bar(), id="tab-bar")
        yield Input(placeholder="Type to filter, Esc to close", id="search-bar")
        yield DataTable(id="table")
        yield Tree("Results", id="result-tree")
        yield RichLog(id="output-log", highlight=True, markup=True)
        yield Static("", id="detail")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self):
        t = self.query_one("#table", DataTable)
        t.cursor_type = "row"
        self._populate()
        t.focus()

    # ── Tab bar ──

    def _tab_bar(self):
        segs = []
        for i, name in enumerate(TABS):
            if i == self._tab:
                segs.append(f"[reverse] {i+1} {name} [/reverse]")
            else:
                segs.append(f" {i+1} {name} ")
        return "".join(segs)

    # ── Escape (layered) ──

    def action_handle_esc(self):
        if self._vs == _VS.OUTPUT:
            if self._cmd_proc and self._cmd_proc.poll() is None:
                self._cmd_proc.terminate()
                self._cmd_proc = None
            self._show_main_view()
            return
        if self._vs == _VS.ACTION:
            self._vs = _VS.TABLE
            self.query_one("#detail", Static).update("")
            self._show_status()
            return
        if self._vs == _VS.SEARCH:
            self._vs = _VS.TABLE
            self._query = ""
            sb = self.query_one("#search-bar", Input)
            sb.display = False
            sb.value = ""
            self._populate()
            self._focus_main()
            return
        if self._esc_armed:
            self.exit()
            return
        self._esc_armed = True
        self.query_one("#status", Static).update(
            " [reverse] Press Esc again to quit [/reverse]")
        if self._esc_timer:
            self._esc_timer.stop()
        self._esc_timer = self.set_timer(2.0, self._esc_reset)

    def _esc_reset(self):
        self._esc_armed = False
        self._esc_timer = None
        self._show_status()

    # ── Tab switching ──

    def action_next_tab(self):
        if self._vs != _VS.TABLE: return
        self._switch_tab((self._tab + 1) % len(TABS))

    def action_prev_tab(self):
        if self._vs != _VS.TABLE: return
        self._switch_tab((self._tab - 1) % len(TABS))

    def action_go_tab(self, idx):
        if self._vs != _VS.TABLE: return
        if 0 <= idx < len(TABS):
            self._switch_tab(idx)

    def _switch_tab(self, idx):
        if idx == self._tab: return
        self._tab = idx
        self._query = ""
        self._vs = _VS.TABLE
        self.query_one("#search-bar", Input).display = False
        self.query_one("#tab-bar", Static).update(self._tab_bar())
        self.query_one("#detail", Static).update("")
        self._populate()
        self._focus_main()

    # ── Visibility helpers ──

    def _show_main_view(self):
        """Show correct main widget for current tab, hide others."""
        self._vs = _VS.TABLE
        is_results = TABS[self._tab] == "Results"
        self.query_one("#table", DataTable).display = not is_results
        self.query_one("#result-tree", Tree).display = is_results
        self.query_one("#output-log", RichLog).display = False
        self._focus_main()
        self._show_status()

    def _focus_main(self):
        if TABS[self._tab] == "Results":
            self.query_one("#result-tree", Tree).focus()
        else:
            self.query_one("#table", DataTable).focus()

    # ── Search ──

    def action_start_search(self):
        if self._vs != _VS.TABLE: return
        self._vs = _VS.SEARCH
        sb = self.query_one("#search-bar", Input)
        sb.display = True
        sb.value = self._query
        sb.focus()

    def on_input_changed(self, event: Input.Changed):
        if event.input.id == "search-bar":
            self._query = event.value.strip().lower()
            self._populate()

    # ── Refresh ──

    def action_refresh(self):
        if self._vs != _VS.TABLE: return
        tab = TABS[self._tab]
        if tab == "Recipes": self._recipes.clear()
        elif tab == "Results": self._results.clear()
        elif tab == "Endpoints": self._endpoints.clear()
        self._populate()

    # ── Enter → action menu ──

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        idx = event.cursor_row
        if idx is None: return
        tab = TABS[self._tab]

        if tab == "Recipes" and idx < len(self._filtered):
            self._action_idx = idx
            self._vs = _VS.ACTION
            r = self._filtered[idx]
            self.query_one("#detail", Static).update(
                f"{r['file']}\n"
                f"  e Execute   v Validate   b Benchmark   p Preview   Esc Cancel")
            self.query_one("#status", Static).update(
                f" {r['file']}  |  e:run  v:dry-run  b:bench  p:yaml  Esc:cancel")

        elif tab == "Endpoints" and idx < len(self._filtered):
            self._action_idx = idx
            self._vs = _VS.ACTION
            ep = self._filtered[idx]
            self.query_one("#detail", Static).update(
                f"{ep['name']}\n"
                f"  c Cleanup   s Status   Esc Cancel")
            self.query_one("#status", Static).update(
                f" {ep['name']}  |  c:cleanup  s:status  Esc:cancel")

    # ── Tree node selected (Results) ──

    def on_tree_node_selected(self, event: Tree.NodeSelected):
        """Show detail when a result tree leaf is selected."""
        node = event.node
        if node.data and isinstance(node.data, dict):
            r = node.data
            self.query_one("#detail", Static).update(
                f"C={r.get('concurrency','')}  "
                f"p50={_f(r.get('latency_p50'))}ms  "
                f"p90={_f(r.get('latency_p90',''))}ms  "
                f"p99={_f(r.get('latency_p99'))}ms  "
                f"tok/s={_f(r.get('tok_per_sec_avg'))}  "
                f"Agg={_f(r.get('aggregate_output_tok_sec'))} tok/s")

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted):
        """Update detail on highlight for tree nodes."""
        node = event.node
        if not node.data:
            # Group node — show summary
            if node.children:
                kids = [c.data for c in node.children if c.data and isinstance(c.data, dict)]
                if kids:
                    p50s = [r["latency_p50"] for r in kids if isinstance(r.get("latency_p50"), (int, float))]
                    rpss = [r["rps"] for r in kids if isinstance(r.get("rps"), (int, float))]
                    cs = ",".join(str(r.get("concurrency", "")) for r in kids)
                    line = f"{len(kids)} levels  (C={cs})"
                    if p50s:
                        line += f"  Latency:{_sparkline(p50s)}  RPS:{_sparkline(rpss)}"
                    self.query_one("#detail", Static).update(line)
        elif isinstance(node.data, dict):
            r = node.data
            self.query_one("#detail", Static).update(
                f"C={r.get('concurrency','')}  "
                f"p50={_f(r.get('latency_p50'))}  p99={_f(r.get('latency_p99'))}  "
                f"tok/s={_f(r.get('tok_per_sec_avg'))}  RPS={_f(r.get('rps'),2)}  "
                f"Valid={r.get('output_validation_rate','?')}")

    # ── Action key handler ──

    def on_key(self, event):
        if self._vs != _VS.ACTION: return
        tab = TABS[self._tab]
        key = event.character or ""

        if tab == "Recipes" and self._action_idx < len(self._filtered):
            r = self._filtered[self._action_idx]
            path = r["path"]
            if key == "e":
                self._run_inline(f"python run.py -f {path}")
            elif key == "v":
                self._run_inline(f"python run.py -f {path} --dry-run")
            elif key == "b":
                self._run_inline(f"python run.py -f {path} --only benchmark")
            elif key == "p":
                self._vs = _VS.TABLE
                try:
                    with open(path) as f:
                        self.query_one("#detail", Static).update(f.read()[:500])
                except Exception:
                    pass
                return
            else:
                return
            event.prevent_default()

        elif tab == "Endpoints" and self._action_idx < len(self._filtered):
            ep = self._filtered[self._action_idx]
            if key == "c":
                self._run_inline(f"python run.py --cleanup --endpoint {ep['name']} --region us-west-2")
            elif key == "s":
                self._run_inline(f"python run.py --status --region us-west-2")
            else:
                return
            event.prevent_default()

    # ── Inline command execution ──

    def _run_inline(self, cmd: str):
        self._vs = _VS.OUTPUT
        self.query_one("#table", DataTable).display = False
        self.query_one("#result-tree", Tree).display = False
        log = self.query_one("#output-log", RichLog)
        log.display = True
        log.clear()
        log.write(f"[bold]$ {cmd}[/bold]\n")
        self.query_one("#detail", Static).update("")
        self.query_one("#status", Static).update(" Running...  Esc: stop and return")
        self._exec_cmd(cmd)

    @work(thread=True)
    def _exec_cmd(self, cmd: str):
        try:
            proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            self._cmd_proc = proc
            for line in proc.stdout:
                self.app.call_from_thread(self._log_line, line.rstrip("\n"))
            proc.wait()
            rc = proc.returncode
            self._cmd_proc = None
            self.app.call_from_thread(self._log_line, f"\n[bold]Exit code: {rc}[/bold]")
            self.app.call_from_thread(self._cmd_done, rc)
        except Exception as e:
            self._cmd_proc = None
            self.app.call_from_thread(self._log_line, f"[red]Error: {e}[/red]")
            self.app.call_from_thread(self._cmd_done, 1)

    def _log_line(self, line: str):
        self.query_one("#output-log", RichLog).write(line)

    def _cmd_done(self, rc: int):
        tag = "[green]Done[/green]" if rc == 0 else f"[red]Failed (exit {rc})[/red]"
        self.query_one("#status", Static).update(f" {tag}  |  Esc: return")

    # ── Populate ──

    def _populate(self):
        tab = TABS[self._tab]
        table = self.query_one("#table", DataTable)
        tree = self.query_one("#result-tree", Tree)

        # Toggle visibility
        is_results = tab == "Results"
        table.display = not is_results and self._vs != _VS.OUTPUT
        tree.display = is_results and self._vs != _VS.OUTPUT

        if tab == "Recipes":
            self._pop_recipes(table)
        elif tab == "Results":
            self._pop_results_tree(tree)
        elif tab == "Endpoints":
            self._pop_endpoints(table)
        elif tab in ("Benchmark", "Deploy"):
            self._pop_info(table, tab)

        self._show_status()

    def _pop_recipes(self, table: DataTable):
        if not self._recipes:
            self._recipes = _load_recipes("recipes")
        data = self._recipes
        if self._query:
            data = [r for r in data if any(self._query in str(v).lower() for v in r.values())]
        self._filtered = data
        table.clear(columns=True)
        table.add_column("Recipe", width=38)
        table.add_column("Model", width=22)
        table.add_column("Instance", width=18)
        table.add_column("Opt", width=14)
        table.add_column("Container", width=10)
        table.add_column("Pipeline", width=28)
        if data:
            table.add_rows([
                (r["file"],
                 r["model"].split("/")[-1][:22] if "/" in r["model"] else r["model"][:22],
                 r["instance"], r["opt"], r["container"], r["pipeline"])
                for r in data])

    def _pop_results_tree(self, tree: Tree):
        if not self._results:
            self._results = _load_results("results/matrix")

        # Group by (model, instance, opt, use_case)
        groups: dict[tuple, list[dict]] = {}
        for r in self._results:
            if self._query and not any(self._query in str(v).lower() for v in r.values()):
                continue
            key = (r.get("model", ""), r.get("instance_type", ""),
                   r.get("optimization", ""), r.get("use_case", ""))
            groups.setdefault(key, []).append(r)

        # Sort groups and children
        for key in groups:
            groups[key].sort(key=lambda x: x.get("concurrency", 0))

        tree.clear()
        tree.root.expand()
        flat = []

        # Group by model first, then sub-group by (instance, opt, use_case)
        by_model: dict[str, list[tuple]] = {}
        for key in sorted(groups.keys()):
            model = key[0]
            by_model.setdefault(model, []).append(key)

        for model, keys in by_model.items():
            model_node = tree.root.add(f"{model} ({len(keys)} configs)", expand=False)
            for key in keys:
                _, inst, opt, uc = key
                rows = groups[key]
                peak = rows[-1] if rows else {}
                rps = _f(peak.get("rps"), 2)
                agg = _f(peak.get("aggregate_output_tok_sec"))
                label = f"{inst} {opt} {uc}  RPS={rps} Agg={agg}"
                config_node = model_node.add(label, expand=False)
                for r in rows:
                    c = r.get("concurrency", "?")
                    p50 = _f(r.get("latency_p50"))
                    p99 = _f(r.get("latency_p99"))
                    leaf_rps = _f(r.get("rps"), 2)
                    leaf_agg = _f(r.get("aggregate_output_tok_sec"))
                    config_node.add_leaf(
                        f"C={c}  p50={p50}ms  p99={p99}ms  RPS={leaf_rps}  Agg={leaf_agg}",
                        data=r)
                    flat.append(r)

        self._filtered = flat
        n_groups = len(groups)
        n_rows = len(flat)
        self.query_one("#status", Static).update(
            f" Results: {n_rows} rows, {n_groups} groups  |  "
            f"Enter:expand/collapse  /:filter  </>:tabs  Esc:quit")

    def _pop_endpoints(self, table: DataTable):
        table.clear(columns=True)
        table.add_column("Endpoint", width=45)
        table.add_column("Instance", width=20)
        table.add_column("Status", width=12)
        table.add_column("ICs", width=5)
        if self._endpoints:
            data = self._endpoints
            if self._query:
                data = [e for e in data if self._query in e["name"].lower() or self._query in e["status"].lower()]
            self._filtered = data
            if data:
                table.add_rows([(e["name"], e["instance"], e["status"],
                                 str(len(e["ics"])) if e["ics"] else "-") for e in data])
        else:
            self._filtered = []
            self._fetch_endpoints()

    def _pop_info(self, table: DataTable, tab: str):
        self._filtered = []
        table.clear(columns=True)
        table.add_column("Command", width=15)
        table.add_column("Example", width=80)
        if tab == "Benchmark":
            table.add_rows([
                ("Run", "python run.py -f recipes/RECIPE.yaml --only benchmark --endpoint NAME"),
                ("Full", "python run.py -f recipes/qwen3-32b-g7e-eagle3.yaml"),
                ("Dry run", "python run.py -f recipes/RECIPE.yaml --dry-run"),
                ("Status", "python run.py --status --region us-west-2"),
            ])
        else:
            table.add_rows([
                ("Deploy", "python run.py -f recipes/RECIPE.yaml --only deploy"),
                ("Cleanup", "python run.py --cleanup --endpoint NAME --region us-west-2"),
                ("Full", "python run.py -f recipes/RECIPE.yaml"),
                ("Override", "python run.py -f RECIPE.yaml --image-uri URI --tp 2"),
            ])

    @work(thread=True)
    def _fetch_endpoints(self):
        try:
            import boto3
            sm = boto3.client("sagemaker", region_name="us-west-2")
            eps = []
            for ep in sm.list_endpoints(MaxResults=100).get("Endpoints", []):
                name, status = ep["EndpointName"], ep["EndpointStatus"]
                inst = "?"
                try:
                    c = sm.describe_endpoint_config(EndpointConfigName=name)
                    inst = c["ProductionVariants"][0].get("InstanceType", "?")
                except Exception: pass
                ics = []
                try:
                    r = sm.list_inference_components(EndpointNameEquals=name, MaxResults=10)
                    ics = [x["InferenceComponentName"] for x in r.get("InferenceComponents", [])]
                except Exception: pass
                eps.append({"name": name, "status": status, "instance": inst, "ics": ics})
            self._endpoints = eps
            if TABS[self._tab] == "Endpoints":
                self.app.call_from_thread(self._populate)
                self.app.call_from_thread(self._focus_main)
        except Exception as e:
            self.app.call_from_thread(
                self.query_one("#status", Static).update, f" Endpoints error: {e}")

    # ── Status ──

    def _show_status(self):
        if self._esc_armed:
            return  # Don't overwrite quit message
        tab = TABS[self._tab]
        n = len(self._filtered)
        if tab == "Recipes":
            txt = f"Recipes: {n}/{len(self._recipes)}"
            hint = "  Enter:actions  /:filter  </>:tabs  Esc:quit"
        elif tab == "Results":
            return  # Results tab sets its own status in _pop_results_tree
        elif tab == "Endpoints":
            txt = f"Endpoints: {n}"
            hint = "  Enter:actions  r:refresh  </>:tabs  Esc:quit"
        else:
            txt = tab
            hint = "  </>:tabs  Esc:quit"
        self.query_one("#status", Static).update(f" {txt}{hint}")

    # ── Row highlight (DataTable only — Recipes, Endpoints) ──

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted):
        if self._vs != _VS.TABLE: return
        tab = TABS[self._tab]
        idx = event.cursor_row
        if idx is None or idx >= len(self._filtered): return

        if tab == "Recipes":
            r = self._filtered[idx]
            cost = f"${r['cost']}/hr" if r.get("cost") else "?"
            self.query_one("#detail", Static).update(
                f"{r['model']}  |  {r['instance']} ({cost})  |  {r['opt']}  |  {r['pipeline']}")

        elif tab == "Endpoints":
            ep = self._filtered[idx]
            ics = ", ".join(ep["ics"]) if ep["ics"] else "none"
            self.query_one("#detail", Static).update(
                f"{ep['name']}  |  {ep['instance']}  |  {ep['status']}  |  ICs: {ics}")


if __name__ == "__main__":
    BenchmarkApp().run()
