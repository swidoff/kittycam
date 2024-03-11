import time
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Set

import cv2
import numpy as np
import plyer
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

# model_dir = Path("../models") / "ssd_mobilenet_v2_coco_2018_03_29"
# model_file = model_dir / "frozen_inference_graph.pb"
# config_file = model_dir / "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
# width = 300
# height = 300
# net = cv2.dnn.readNetFromTensorflow(str(model_file), str(config_file))

model_dir = Path("../models") / "ssd_mobilenet_v3_large_coco_2020_01_14"
model_file = model_dir / "frozen_inference_graph.pb"
config_file = model_dir / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
net = cv2.dnn_DetectionModel(str(model_file), str(config_file))
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# model_dir = Path("../models") / "faster_rcnn_inception_v2_coco_2018_01_28"
# model_file = model_dir / "frozen_inference_graph.pb"
# config_file = model_dir / "faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"
# width = 320
# height = 320
# net = cv2.dnn.readNetFromTensorflow(str(model_file), str(config_file))

# model_dir = Path("../models")
# model_file = model_dir / "efficientdet-d0.pb"
# config_file = model_dir / "efficientdet-d0.pbtxt"
# width = 512
# height = 512
# net = cv2.dnn_DetectionModel(str(model_file), str(config_file))
# net.setInputSize(512, 512)
# net.setInputScale(1.0 / 255)
# net.setInputMean((123.675, 116.28, 103.53))


class Camera(Image):
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
        desktop_notifications: bool = True,
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
        self.desktop_notifications = desktop_notifications
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
            objects = detect_objects(frame, self.class_labels, self.threshold)
            do_intersect = intersect_region(objects, self.polygons, frame.shape[1])

            if self.display:
                display_objects(frame, objects, do_intersect)
                self.texture = cv2_img_to_texture(frame)

            if any(do_intersect) and ((now := time.time()) - self.last_detected_time > self.debounce_seconds):
                for i, b in enumerate(do_intersect):
                    if b:
                        msg = f"{objects[i].label} detected!"
                        plyer.notification.notify("KittyCam Alert", msg)
                self.last_detected_time = now


@dataclass
class LabeledRectangle(object):
    x: int
    y: int
    w: int
    h: int
    label: str
    confidence: float


def detect_objects(im: np.ndarray, label_filter: Set[str], threshold: float = 0.5) -> list[LabeledRectangle]:
    res = []
    rows, cols, _ = im.shape
    # blob = cv2.dnn.blobFromImage(im, 1.0, size=(width, height), mean=(0, 0, 0), swapRB=True, crop=False)
    # net.setInput(blob)
    # objects = net.forward()
    #
    # for i in range(objects.shape[2]):
    #     # Find the class and confidence
    #     class_id = int(objects[0, 0, i, 1])
    #     score = float(objects[0, 0, i, 2])
    #
    #     # Check if the detection is of good quality
    #     class_label = coco.class_labels[class_id]
    #     if score > threshold and class_label in label_filter:
    #         # Recover original coordinates from normalized coordinates
    #         x = int(objects[0, 0, i, 3] * cols)
    #         y = int(objects[0, 0, i, 4] * rows)
    #         w = int(objects[0, 0, i, 5] * cols - x)
    #         h = int(objects[0, 0, i, 6] * rows - y)
    #         res.append(LabeledRectangle(x, y, w, h, class_label))

    classes, confidences, boxes = net.detect(im, confThreshold=threshold, nmsThreshold=0.4)
    if isinstance(classes, tuple) or classes.shape[0] == 0:
        return res

    for class_id, confidence, [x, y, w, h] in zip(classes.flatten(), confidences.flatten(), boxes):
        class_label = coco.class_labels[class_id]
        if confidence > threshold and class_label in label_filter:
            print(class_label, confidence, [x, y, w, h])
            res.append(LabeledRectangle(x, y, w, h, class_label, confidence))

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

        display_text(im, f"{o.label} ({(o.confidence * 100.0):.02}%)", o.x, o.y)
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
