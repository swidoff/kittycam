import dataclasses
from dataclasses import dataclass
from typing import List

from kivy.eventmanager import EventManagerBase
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import ListProperty
from kivy.uix.widget import Widget
from shapely import LineString, Polygon

d = 20


@dataclass
class _DrawingState(object):
    first_x: int
    first_y: int
    active_line: Line | None = None
    active_color: Color | None = None
    active_line_valid: bool = True
    completed_lines: List[LineString] = dataclasses.field(default_factory=list)

    def completes_polygon(self, x: int, y: int) -> bool:
        if len(self.completed_lines) < 2:
            return False
        else:
            return (
                self.first_x - d / 2 <= x <= self.first_x + d / 2
                and self.first_y - d / 2 <= y <= self.first_y + d / 2
            )

    def complete_line(self, x: int, y: int):
        self.completed_lines.append(self.update_active_line(x, y))

    def to_polygon(self) -> Polygon:
        return Polygon(shell=[line.coords[0] for line in self.completed_lines])

    def update_active_line(self, x2: int, y2: int) -> LineString | None:
        if self.active_line is None:
            return None
        else:
            [x1, y1] = self.active_line.points[:2]
            self.active_line.points = [x1, y1, x2, y2]
            return LineString([[x1, y1], [x2, y2]])


class DrawPolygons(Widget):
    polygons = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state: _DrawingState | None = None

    def on_touch_down(self, touch):
        if self.state is None:
            self.state = _DrawingState(touch.x, touch.y)
        elif not self.state.active_line_valid:
            return
        elif self.state.completes_polygon(touch.x, touch.y):
            self.state.complete_line(self.state.first_x, self.state.first_y)
            self.polygons = self.polygons + [self.state.to_polygon()]
            self.state = None
            return
        else:
            self.state.complete_line(touch.x, touch.y)

        with self.canvas:
            self.state.active_color = Color((0, 0, 1))
            _ = Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            self.state.active_line = Line(points=(touch.x, touch.y, touch.x, touch.y))

    def on_move(self, x, y):
        if self.state:
            active_line = self.state.update_active_line(x, y)

            if any(active_line.crosses(line) for line in self.state.completed_lines):
                self.state.active_color.rgb = (1, 0, 0)
                self.state.active_line_valid = False
            else:
                self.state.active_color.rgb = (0, 0, 1)
                self.state.active_line_valid = True

    def clear(self):
        self.state = None
        self.polygons = []
        self.canvas.clear()


class DrawPolygonEventManager(EventManagerBase):
    type_ids = ("hover",)

    def __init__(self, paint_widget: DrawPolygons):
        super().__init__()
        self.paint_widget = paint_widget

    def start(self):
        super().start()

    def dispatch(self, etype, me):
        x = int(self.window.width * me.sx)
        y = int(self.window.height * me.sy)
        self.paint_widget.on_move(x, y)

    def stop(self):
        super().stop()
