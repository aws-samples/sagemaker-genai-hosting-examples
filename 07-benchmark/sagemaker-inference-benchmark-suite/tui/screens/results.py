"""Results viewer."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Input, Static


def _f(v, d=1):
    if v is None or v == "" or v == "None":
        return "-"
    try:
        return f"{float(v):.{d}f}"
    except (ValueError, TypeError):
        return str(v)


class ResultsScreen(Screen):
    BINDINGS = [
        ("h", "app.switch_mode('home')", "Home"),
        ("slash", "focus_search", "Search"),
    ]

    DEFAULT_CSS = """
    #rs-search { dock: top; height: auto; padding: 1 0; }
    #rs-search Input { width: 1fr; }
    #rs-detail { height: auto; max-height: 8; padding: 0 1; }
    #rs-status { dock: bottom; height: 1; padding: 0 1; }
    """

    def __init__(self):
        super().__init__()
        self._rows = []
        self._filtered = []
        self._groups = {}

    def compose(self) -> ComposeResult:
        with Horizontal(id="rs-search"):
            yield Input(placeholder="Type to filter...", id="rs-q")
        yield DataTable(id="rs-tbl")
        yield Static("", id="rs-detail")
        yield Static("Loading...", id="rs-status")
        yield Footer()

    def on_mount(self):
        t = self.query_one("#rs-tbl", DataTable)
        t.add_columns("Model", "Instance", "Opt", "Use Case", "C",
                       "p50ms", "p99ms", "RPS", "Agg tok/s")
        t.cursor_type = "row"
        from tui.data_loader import load_results, group_results
        self._rows = load_results("results/matrix")
        self._groups = group_results(self._rows)
        self._filtered = list(self._rows)
        self._fill()

    def _fill(self):
        t = self.query_one("#rs-tbl", DataTable)
        t.clear()
        for i, r in enumerate(self._filtered):
            t.add_row(
                str(r.get("model", ""))[:15],
                str(r.get("instance_type", ""))[:18],
                str(r.get("optimization", ""))[:12],
                str(r.get("use_case", ""))[:15],
                str(r.get("concurrency", "")),
                _f(r.get("latency_p50")), _f(r.get("latency_p99")),
                _f(r.get("rps"), 2), _f(r.get("aggregate_output_tok_sec")),
                key=i,
            )
        self.query_one("#rs-status", Static).update(
            f"{len(self._filtered)}/{len(self._rows)} results  {len(self._groups)} groups")

    def on_input_changed(self, event: Input.Changed):
        if event.input.id != "rs-q":
            return
        q = event.value.lower().strip()
        self._filtered = [r for r in self._rows
                          if not q or any(q in str(v).lower() for v in r.values())]
        self._fill()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted):
        if event.row_key is not None:
            try:
                i = event.row_key.value
                if isinstance(i, int) and i < len(self._filtered):
                    self._detail(self._filtered[i])
            except Exception:
                pass

    def _detail(self, r):
        from tui.data_loader import sparkline
        key = (r.get("model",""), r.get("optimization",""),
               r.get("instance_type",""), r.get("use_case",""))

        lines = [
            f"{r.get('model','?')} | {r.get('instance_type','?')} | {r.get('optimization','?')} | {r.get('use_case','?')}",
            f"C={r.get('concurrency','?')}  p50={_f(r.get('latency_p50'))}  p90={_f(r.get('latency_p90'))}  p99={_f(r.get('latency_p99'))}  avg={_f(r.get('latency_avg'))}ms",
            f"tok/s={_f(r.get('tok_per_sec_avg'))}  RPS={_f(r.get('rps'),2)}  Agg={_f(r.get('aggregate_output_tok_sec'))}  Valid={r.get('output_validation_rate','?')}",
        ]
        ttft = r.get("ttft_p50")
        if ttft and ttft != "None":
            lines.append(f"TTFT p50={_f(ttft)}  p90={_f(r.get('ttft_p90'))}ms")

        if key in self._groups:
            g = self._groups[key]
            p50s = [x.get("latency_p50",0) for x in g if isinstance(x.get("latency_p50"), (int,float))]
            cs = [str(x.get("concurrency","?")) for x in g]
            if p50s:
                lines.append(f"Latency: {sparkline(p50s)}  C={','.join(cs)}")
                rpss = [x.get("rps",0) for x in g if isinstance(x.get("rps"), (int,float))]
                if rpss:
                    lines.append(f"RPS:     {sparkline(rpss)}")

        self.query_one("#rs-detail", Static).update("\n".join(lines))

    def action_focus_search(self):
        self.query_one("#rs-q", Input).focus()
