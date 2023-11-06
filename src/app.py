import cv2
import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout

from camera import Camera
from draw import DrawPolygons, DrawPolygonEventManager

kivy.require("2.2.1")


class KittyCam(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        super().__init__(**kwargs)
        self.capture: cv2.VideoCapture | None = None
        self.camera: Camera | None = None
        self.draw_polygons: DrawPolygons | None = None

    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.camera = Camera(capture=self.capture, fps=30)
        self.draw_polygons = DrawPolygons()

        clear_button = Button(text="Clear")
        clear_button.bind(on_release=self.clear_canvas)

        parent = Widget()
        layout = FloatLayout(size=(640, 480))
        polygon_layout = BoxLayout(opacity=0.5)
        polygon_layout.add_widget(self.draw_polygons)

        camera_layout = BoxLayout()
        camera_layout.add_widget(self.camera)

        layout.add_widget(camera_layout)
        layout.add_widget(polygon_layout)

        parent.add_widget(layout)
        parent.add_widget(clear_button)
        return parent

    def on_start(self):
        super().on_start()
        self.root_window.register_event_manager(
            DrawPolygonEventManager(self.draw_polygons)
        )

    def on_stop(self):
        self.capture.release()

    def clear_canvas(self, _obj):
        self.draw_polygons.clear()


if __name__ == "__main__":
    KittyCam().run()
