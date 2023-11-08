import cv2
import kivy
from kivy.app import App
from kivy.config import Config
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout

from camera import Camera
from draw import DrawPolygons, DrawPolygonEventManager

kivy.require("2.2.1")
Config.set("graphics", "width", "640")
Config.set("graphics", "height", "480")


class KittyCam(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        super().__init__(**kwargs)
        self.capture: cv2.VideoCapture | None = None
        self.camera: Camera | None = None
        self.draw_polygons: DrawPolygons | None = None
        self.keyboard = None

    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.camera = Camera(
            capture=self.capture, fps=30, size=(640, 480), class_labels=["person"]
        )
        self.draw_polygons = DrawPolygons(size=(640, 480))

        layout = FloatLayout(size=(640, 480))
        polygon_layout = BoxLayout(opacity=0.5)
        polygon_layout.add_widget(self.draw_polygons)
        camera_layout = BoxLayout()
        camera_layout.add_widget(self.camera)
        layout.add_widget(camera_layout)
        layout.add_widget(polygon_layout)

        self.draw_polygons.bind(polygons=self.camera.setter("polygons"))
        return layout

    def on_start(self):
        super().on_start()
        self.root_window.register_event_manager(
            DrawPolygonEventManager(self.draw_polygons)
        )
        self.keyboard = self.root_window.request_keyboard(
            self.keyboard_closed, self.draw_polygons
        )
        self.keyboard.bind(on_key_down=self.on_keyboard_down)

    def on_keyboard_down(self, _keyboard, keycode, _text, _modifiers):
        if keycode[1] == "escape":
            self.draw_polygons.clear()
        return True

    def keyboard_closed(self):
        self.keyboard.unbind(on_key_down=self.on_keyboard_down)
        self.keyboard = None

    def on_stop(self):
        self.capture.release()

    def clear_canvas(self, _obj):
        self.draw_polygons.clear()


if __name__ == "__main__":
    KittyCam().run()
