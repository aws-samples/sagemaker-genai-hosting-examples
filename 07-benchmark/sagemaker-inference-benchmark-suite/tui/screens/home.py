"""Home screen."""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Static


class HomeScreen(Screen):
    BINDINGS = [
        ("b", "app.switch_mode('benchmark')", "Benchmark"),
        ("d", "app.switch_mode('deploy')", "Deploy"),
        ("r", "app.switch_mode('recipes')", "Recipes"),
        ("s", "app.switch_mode('results')", "Results"),
        ("e", "app.switch_mode('endpoints')", "Endpoints"),
        ("q", "app.quit", "Quit"),
    ]

    DEFAULT_CSS = """
    HomeScreen { align: center middle; }
    #menu { width: 55; height: auto; padding: 1 2; }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "SageMaker Benchmark Suite\n"
            "─────────────────────────\n"
            "\n"
            " b  Benchmark   Run live benchmarks\n"
            " d  Deploy      Deploy endpoints\n"
            " r  Recipes     Browse YAML recipes\n"
            " s  Results     View benchmark results\n"
            " e  Endpoints   Manage endpoints\n"
            " q  Quit\n",
            id="menu",
        )
        yield Footer()
