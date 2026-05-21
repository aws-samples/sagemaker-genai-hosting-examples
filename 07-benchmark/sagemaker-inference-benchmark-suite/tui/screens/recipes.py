"""Recipe browser."""

import os

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Input, RichLog, Static


class RecipesScreen(Screen):
    BINDINGS = [
        ("h", "app.switch_mode('home')", "Home"),
        ("slash", "focus_search", "Search"),
    ]

    DEFAULT_CSS = """
    #rc-search { dock: top; height: auto; padding: 1 0; }
    #rc-search Input { width: 1fr; }
    #rc-body { height: 1fr; }
    #rc-tbl { width: 2fr; }
    #rc-preview { width: 1fr; }
    #rc-status { dock: bottom; height: 1; padding: 0 1; }
    """

    def __init__(self):
        super().__init__()
        self._all = []
        self._filtered = []

    def compose(self) -> ComposeResult:
        with Horizontal(id="rc-search"):
            yield Input(placeholder="Type to filter...", id="rc-q")
        with Horizontal(id="rc-body"):
            yield DataTable(id="rc-tbl")
            yield RichLog(id="rc-preview", highlight=True, markup=True)
        yield Static("", id="rc-status")
        yield Footer()

    def on_mount(self):
        t = self.query_one("#rc-tbl", DataTable)
        t.add_columns("Recipe", "Model", "Instance", "Opt", "Pipeline")
        t.cursor_type = "row"
        from tui.data_loader import load_recipes
        self._all = load_recipes("recipes")
        self._filtered = list(self._all)
        self._fill()

    def _fill(self):
        t = self.query_one("#rc-tbl", DataTable)
        t.clear()
        for i, r in enumerate(self._filtered):
            m = r["model"].split("/")[-1][:20] if "/" in r["model"] else r["model"][:20]
            t.add_row(r["file"], m, r["instance"], r["optimization"], r["pipeline"], key=i)
        self.query_one("#rc-status", Static).update(
            f"{len(self._filtered)}/{len(self._all)} recipes")

    def on_input_changed(self, event: Input.Changed):
        if event.input.id != "rc-q":
            return
        q = event.value.lower().strip()
        self._filtered = [r for r in self._all
                          if not q or any(q in str(v).lower() for v in r.values())]
        self._fill()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted):
        if event.row_key is not None:
            try:
                i = event.row_key.value
                if isinstance(i, int) and i < len(self._filtered):
                    p = self._filtered[i]["path"]
                    preview = self.query_one("#rc-preview", RichLog)
                    preview.clear()
                    with open(p) as f:
                        preview.write(f.read())
            except Exception:
                pass

    def action_focus_search(self):
        self.query_one("#rc-q", Input).focus()
