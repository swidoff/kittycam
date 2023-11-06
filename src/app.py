from typing import List

import kivy
from kivy.app import App
from kivy.eventmanager import EventManagerBase
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from shapely import LineString

kivy.require("2.2.1")


class MyPaintWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.complete = False
        self.first_x: int | None = None
        self.first_y: int | None = None
        self.active_line: Line | None = None
        self.active_color: Color | None = None
        self.active_line_valid: bool = True
        self.completed_lines: List[LineString] = []

    def on_touch_down(self, touch):
        if self.complete:
            return

        if self.first_x is None:
            self.first_x = touch.x
            self.first_y = touch.y

        d = 20
        if self.completed_lines:
            if (
                self.first_x - d / 2 <= touch.x <= self.first_x + d / 2
                and self.first_y - d / 2 <= touch.y <= self.first_y + d / 2
            ):
                self.complete = True
                self.update_active_line(self.first_x, self.first_y)
            elif self.active_line:
                self.update_active_line(touch.x, touch.y)

        if self.active_line is not None:
            if self.active_line_valid:
                [x1, y1, x2, y2] = self.active_line.points
                self.completed_lines.append(LineString([[x1, y1], [x2, y2]]))
            else:
                return

        if not self.complete:
            with self.canvas:
                self.active_color = Color((1, 1, 1))
                _ = Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
                self.active_line = Line(points=(touch.x, touch.y, touch.x, touch.y))

    def on_move(self, x, y):
        if self.active_line is not None and not self.complete:
            self.update_active_line(x, y)

    def update_active_line(self, x2: int, y2: int):
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

    def clear(self):
        self.complete = False
        self.first_x: int | None = None
        self.first_y: int | None = None
        self.active_line: Line | None = None
        self.active_color: Color | None = None
        self.active_line_valid: bool = True
        self.completed_lines: List[LineString] = []
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
        self.root_window.register_event_manager(
            EventManager(self.root_window.children[0].children[-1])
        )

    def clear_canvas(self, obj):
        self.painter.clear()


if __name__ == "__main__":
    MyApp().run()
