from random import random

import kivy

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line
from kivy.eventmanager import EventManagerBase

kivy.require("2.2.1")


class MyPaintWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.complete = False
        self.lines = []

    def on_touch_down(self, touch):
        if self.complete:
            return

        d = 20
        if self.lines:
            first_points = self.lines[0].points
            first_x = first_points[0]
            first_y = first_points[1]
            if (
                first_x - d / 2 <= touch.x <= first_x + d / 2
                and first_y - d / 2 <= touch.y <= first_y + d / 2
            ):
                print("Complete!")
                self.complete = True
                self.update_last_line(first_x, first_y)
                return
            else:
                self.update_last_line(touch.x, touch.y)

        with self.canvas:
            Color((1, 1, 1))
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            line = Line(points=(touch.x, touch.y, touch.x, touch.y))
            self.lines.append(line)

    def on_move(self, x, y):
        if self.lines and not self.complete:
            self.update_last_line(x, y)

    def update_last_line(self, x, y):
        line = self.lines[-1]
        line.points = [line.points[0], line.points[1], x, y]


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
        self.painter.canvas.clear()


if __name__ == "__main__":
    MyApp().run()
