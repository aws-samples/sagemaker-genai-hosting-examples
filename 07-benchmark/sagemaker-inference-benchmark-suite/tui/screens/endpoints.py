"""Endpoint status."""

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Select, Static


class EndpointsScreen(Screen):
    BINDINGS = [
        ("h", "app.switch_mode('home')", "Home"),
        ("r", "refresh", "Refresh"),
    ]

    DEFAULT_CSS = """
    #ep-bar { height: auto; padding: 1 0; }
    #ep-bar Select { width: 22; }
    #ep-detail { height: auto; max-height: 6; padding: 0 1; }
    #ep-status { dock: bottom; height: 1; padding: 0 1; }
    """

    def __init__(self):
        super().__init__()
        self._data = []
        self._timer = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="ep-bar"):
            yield Select([("us-west-2", "us-west-2"), ("us-east-1", "us-east-1")],
                         value="us-west-2", id="rgn")
            yield Button("Refresh", id="ref")
            yield Button("Auto: OFF", id="auto")
            yield Button("Cleanup", id="rm", variant="error")
        yield DataTable(id="ep-tbl")
        yield Static("", id="ep-detail")
        yield Static("Press [r] to refresh", id="ep-status")
        yield Footer()

    def on_mount(self):
        t = self.query_one("#ep-tbl", DataTable)
        t.add_columns("Endpoint", "Instance", "Status", "ICs")
        t.cursor_type = "row"
        self.action_refresh()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "ref":
            self.action_refresh()
        elif event.button.id == "auto":
            b = self.query_one("#auto", Button)
            if self._timer:
                self._timer.stop()
                self._timer = None
                b.label = "Auto: OFF"
            else:
                self._timer = self.set_interval(30, self.action_refresh)
                b.label = "Auto: ON"
        elif event.button.id == "rm":
            self._cleanup()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted):
        if event.row_key is not None:
            try:
                i = event.row_key.value
                if isinstance(i, int) and i < len(self._data):
                    ep = self._data[i]
                    ics = ", ".join(ep["ics"]) if ep["ics"] else "none"
                    self.query_one("#ep-detail", Static).update(
                        f"{ep['name']}  {ep['instance']}  {ep['status']}  ICs: {ics}")
            except Exception:
                pass

    def action_refresh(self):
        self.query_one("#ep-status", Static).update("Loading...")
        self._load(self.query_one("#rgn", Select).value)

    @work(thread=True)
    def _load(self, region):
        try:
            import boto3
            sm = boto3.client("sagemaker", region_name=region)
            eps = []
            for ep in sm.list_endpoints(MaxResults=100).get("Endpoints", []):
                name, status = ep["EndpointName"], ep["EndpointStatus"]
                inst = "?"
                try:
                    c = sm.describe_endpoint_config(EndpointConfigName=name)
                    inst = c["ProductionVariants"][0].get("InstanceType", "?")
                except Exception:
                    pass
                ics = []
                try:
                    r = sm.list_inference_components(EndpointNameEquals=name, MaxResults=10)
                    ics = [x["InferenceComponentName"] for x in r.get("InferenceComponents", [])]
                except Exception:
                    pass
                eps.append({"name": name, "status": status, "instance": inst, "ics": ics})
            self.app.call_from_thread(self._fill, eps, region)
        except Exception as e:
            self.app.call_from_thread(
                lambda: self.query_one("#ep-status", Static).update(f"Error: {e}"))

    def _fill(self, eps, region):
        self._data = eps
        t = self.query_one("#ep-tbl", DataTable)
        t.clear()
        for i, ep in enumerate(eps):
            s = ep["status"]
            sd = {"InService": f"[green]{s}[/green]", "Creating": f"[yellow]{s}[/yellow]",
                  "Failed": f"[red]{s}[/red]"}.get(s, s)
            t.add_row(ep["name"], ep["instance"], sd,
                      str(len(ep["ics"])) if ep["ics"] else "-", key=i)
        self.query_one("#ep-status", Static).update(f"{region}: {len(eps)} endpoints")

    def _cleanup(self):
        t = self.query_one("#ep-tbl", DataTable)
        if t.cursor_row is not None and t.cursor_row < len(self._data):
            ep = self._data[t.cursor_row]
            rgn = self.query_one("#rgn", Select).value
            self._do_rm(ep["name"], rgn)

    @work(thread=True)
    def _do_rm(self, name, region):
        try:
            from scripts.deployer import cleanup
            cleanup(name, region, "standard")
            self.app.call_from_thread(self.action_refresh)
        except Exception as e:
            self.app.call_from_thread(
                lambda: self.query_one("#ep-status", Static).update(f"Error: {e}"))
