import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Set, List, Tuple

import cv2
import numpy as np
import toolz
from kivy.clock import Clock
from kivy.graphics.texture import texture_create
from kivy.properties import ListProperty, NumericProperty, BooleanProperty
from kivy.uix.image import Image
from shapely import Polygon

import coco

s = 0
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

model_dir = Path("../models") / "ssd_mobilenet_v2_coco_2018_03_29"
model_file = model_dir / "frozen_inference_graph.pb"
config_file = model_dir / "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

net = cv2.dnn.readNetFromTensorflow(str(model_file), str(config_file))


class Camera(Image):
    polygons = ListProperty([])
    display = BooleanProperty(True)

    def __init__(
        self,
        capture: cv2.VideoCapture,
        class_labels: Collection[str] | None = None,
        display_fps: int = 30,
        notify_fps: int = 2,
        debounce_seconds: int = 5,
        **kwargs,
    ):
        super(Camera, self).__init__(**kwargs)
        self.fps = {True: display_fps, False: notify_fps}
        self.capture = capture
        self.class_labels = set(class_labels or coco.class_labels)
        self.event = None
        self.display = True
        self.debounce_seconds = debounce_seconds
        self.last_detected_time = 0.0
        self.on_display(self, True)

    def on_display(self, _instance, new_on_display):
        if self.event:
            self.event.cancel()

        if not new_on_display:
            has_frame, frame = self.capture.read()
            if has_frame:
                display_text(frame, "Running!", 10, 50)
                self.texture = cv2_img_to_texture(frame)

        self.event = Clock.schedule_interval(self.update, 1.0 / self.fps[new_on_display])

    def update(self, _dt):
        has_frame, frame = self.capture.read()
        if has_frame:
            objects = detect_objects(frame)
            class_objects = filter_objects(objects, self.class_labels, frame.shape[0], frame.shape[1])
            do_intersect = intersect_region(class_objects, self.polygons, frame.shape[1])

            if self.display:
                display_objects(frame, class_objects, do_intersect)
                self.texture = cv2_img_to_texture(frame)

            if any(do_intersect) and ((now := time.time()) - self.last_detected_time > self.debounce_seconds):
                for i, b in enumerate(do_intersect):
                    if b:
                        print(class_objects[i].label)
                self.last_detected_time = now


def detect_objects(im: np.ndarray, dim: int = 300) -> np.ndarray:
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    return net.forward()


@dataclass
class LabeledRectangle(object):
    x: int
    y: int
    w: int
    h: int
    label: str


def filter_objects(
    objects: np.ndarray,
    label_filter: Set[str],
    rows: int,
    cols: int,
    threshold: float = 0.25,
) -> list[LabeledRectangle]:
    res = []

    # For every Detected Object
    for i in range(objects.shape[2]):
        # Find the class and confidence
        class_id = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        # Check if the detection is of good quality
        class_label = coco.class_labels[class_id]
        if score > threshold and class_label in label_filter:
            # Recover original coordinates from normalized coordinates
            x = int(objects[0, 0, i, 3] * cols)
            y = int(objects[0, 0, i, 4] * rows)
            w = int(objects[0, 0, i, 5] * cols - x)
            h = int(objects[0, 0, i, 6] * rows - y)
            res.append(LabeledRectangle(x, y, w, h, class_label))

    return res


def intersect_region(objects: list[LabeledRectangle], regions: list[Polygon], rows: int) -> list[bool]:
    res = []
    for o in objects:
        oy = rows - o.y
        poly = Polygon(shell=[(o.x, oy), (o.x + o.w, oy), (o.x + o.w, oy - o.h), (o.x, oy - o.h)])
        res.append(any(poly.intersects(region) for region in regions))
    return res


def display_objects(im: np.ndarray, objects: list[LabeledRectangle], intersects: list[bool]):
    for i, o in enumerate(objects):
        if intersects[i]:
            color = (0, 0, 255)
        else:
            color = (255, 255, 255)

        display_text(im, o.label, o.x, o.y)
        cv2.rectangle(im, (o.x, o.y), (o.x + o.w, o.y + o.h), color, 2)


def display_text(im: np.ndarray, text: str, x: int, y: int):
    # Get text size
    text_size = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, THICKNESS)
    dim = text_size[0]
    baseline = text_size[1]

    # Use text size to create a black rectangle
    cv2.rectangle(
        im,
        (x, y - dim[1] - baseline),
        (x + dim[0], y + baseline),
        (0, 0, 0),
        cv2.FILLED,
    )

    # Display text inside the rectangle
    cv2.putText(
        im,
        text,
        (x, y - 5),
        FONT_FACE,
        FONT_SCALE,
        (0, 255, 255),
        THICKNESS,
        cv2.LINE_AA,
    )


def cv2_img_to_texture(frame):
    buf1 = cv2.flip(frame, 0)
    buf = buf1.tostring()
    image_texture = texture_create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
    image_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
    return image_texture
