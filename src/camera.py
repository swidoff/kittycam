import time
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Set

import cv2
import numpy as np
from kivy.clock import Clock
from kivy.graphics.texture import texture_create
from kivy.properties import ListProperty, BooleanProperty
from kivy.uix.image import Image
from shapely import Polygon

import coco

s = 0
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

model_dir = Path("../models") / "ssd_mobilenet_v3_large_coco_2020_01_14"
model_file = model_dir / "frozen_inference_graph.pb"
config_file = model_dir / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
net = cv2.dnn_DetectionModel(str(model_file), str(config_file))
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


class Camera(Image):
    """Samples the camera and detects whether objects with the given labels and confidence >= threshold intersect
    the list of polygon. On detection will trigger the `on_detected` event with a message argument, then silence further
    detections for `debounce_seconds`. Display mode renders the camera at `display_fps`, otherwise shows a static image
    and samples the camera at `notify_fps`.
    """

    polygons = ListProperty([])
    display = BooleanProperty(True)

    def __init__(
        self,
        capture: cv2.VideoCapture,
        class_labels: Collection[str] | None = None,
        display_fps: int = 30,
        notify_fps: int = 2,
        debounce_seconds: int = 15,
        threshold: float = 0.5,
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
        self.threshold = threshold
        self.on_display(self, True)
        self.register_event_type("on_detect")

    # noinspection PyMethodMayBeStatic
    def on_detect(self, msg: str):
        print(msg)

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
            objects = detect_objects(frame, self.class_labels, self.threshold)
            update_object_intersects(objects, self.polygons, frame.shape[0])

            if self.display:
                display_objects(frame, objects)
                self.texture = cv2_img_to_texture(frame)

            if any(o.intersects for o in objects) and (
                (now := time.time()) - self.last_detected_time > self.debounce_seconds
            ):
                for o in objects:
                    if o.intersects:
                        self.dispatch("on_detect", f"{o.label} detected!")
                self.last_detected_time = now


@dataclass
class LabeledRectangle(object):
    x: int
    y: int
    w: int
    h: int
    label: str
    confidence: float
    intersects: bool = False


def detect_objects(im: np.ndarray, label_filter: Set[str], threshold: float = 0.5) -> list[LabeledRectangle]:
    res = []
    rows, cols, _ = im.shape

    classes, confidences, boxes = net.detect(im, confThreshold=threshold, nmsThreshold=0.4)
    if isinstance(classes, tuple) or classes.shape[0] == 0:
        return res

    for class_id, confidence, [x, y, w, h] in zip(classes.flatten(), confidences.flatten(), boxes):
        class_label = coco.class_labels[class_id]
        if confidence > threshold and class_label in label_filter:
            res.append(LabeledRectangle(x, y, w, h, class_label, confidence))

    return res


def update_object_intersects(objects: list[LabeledRectangle], regions: list[Polygon], rows: int):
    if not regions:
        return

    for o in objects:
        oy = rows - o.y
        poly = Polygon(shell=[(o.x, oy), (o.x + o.w, oy), (o.x + o.w, oy - o.h), (o.x, oy - o.h)])
        o.intersects = any(poly.intersects(region) for region in regions)


def display_objects(im: np.ndarray, objects: list[LabeledRectangle]):
    for o in objects:
        if o.intersects:
            color = (0, 0, 255)
        else:
            color = (255, 255, 255)

        display_text(im, f"{o.label} ({o.confidence:2.2%})", o.x, o.y)
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
