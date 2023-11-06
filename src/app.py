import dataclasses
from dataclasses import dataclass
from typing import List

import kivy
from kivy.app import App
from kivy.eventmanager import EventManagerBase
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from shapely import LineString, Polygon

kivy.require("2.2.1")
d = 20


@dataclass
class DrawingState(object):
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
        [x1, y1] = self.active_line.points[:2]
        self.completed_lines.append(LineString([[x1, y1], [x, y]]))

    def to_polygon(self) -> Polygon:
        pass

    def update_active_line(self, x2: int, y2: int):
        if self.active_line is None:
            return

        x1 = self.active_line.points[0]
        y1 = self.active_line.points[1]
        self.active_line.points = [x1, y1, x2, y2]
        active_line = LineString([[x1, y1], [x2, y2]])
        if any(active_line.crosses(line) for line in self.completed_lines):
            self.active_color.rgb = (1, 0, 0)
            self.active_line_valid = False
        else:
            self.active_color.rgb = (1, 1, 1)
            self.active_line_valid = True


class MyPaintWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state: DrawingState | None = None
        self.polygons: List[Polygon] = []

    def on_touch_down(self, touch):
        if self.state is None:
            self.state = DrawingState(touch.x, touch.y)
        elif not self.state.active_line_valid:
            return
        elif self.state.completes_polygon(touch.x, touch.y):
            self.state.complete_line(self.state.first_x, self.state.first_y)
            self.polygons.append(self.state.to_polygon())
            self.state = None
            return
        else:
            self.state.complete_line(touch.x, touch.y)

        with self.canvas:
            self.state.active_color = Color((1, 1, 1))
            _ = Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            self.state.active_line = Line(points=(touch.x, touch.y, touch.x, touch.y))

    def on_move(self, x, y):
        if self.state:
            self.state.update_active_line(x, y)

    def clear(self):
        self.state = None
        self.polygons = []
        self.canvas.clear()


class EventManager(EventManagerBase):
    type_ids = ("hover",)

    def __init__(self, paint_widget: MyPaintWidget):
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


class MyApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.painter: MyPaintWidget | None = None

    def build(self):
        parent = Widget()
        self.painter = MyPaintWidget()
        clear_button = Button(text="Clear")
        clear_button.bind(on_release=self.clear_canvas)
        parent.add_widget(self.painter)
        parent.add_widget(clear_button)
        return parent

    def on_start(self):
        super().on_start()
        self.root_window.register_event_manager(EventManager(self.painter))

    def clear_canvas(self, _obj):
        self.painter.clear()


if __name__ == "__main__":
    MyApp().run()
