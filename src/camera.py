from pathlib import Path
from typing import Collection, Set, List

import cv2
import numpy as np
from kivy.clock import Clock
from kivy.graphics.texture import texture_create
from kivy.properties import ListProperty
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

    def __init__(
        self,
        capture: cv2.VideoCapture,
        fps: int,
        class_labels: Collection[str] | None,
        **kwargs
    ):
        super(Camera, self).__init__(**kwargs)
        self.capture = capture
        self.class_labels = set(class_labels or coco.class_labels)
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, _dt):
        has_frame, frame = self.capture.read()
        if has_frame:
            objects = detect_objects(frame)
            display_objects(frame, objects, self.class_labels, self.polygons)
            self.texture = cv2_img_to_texture(frame)


def detect_objects(im: np.ndarray, dim: int = 300) -> np.ndarray:
    blob = cv2.dnn.blobFromImage(
        im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False
    )
    net.setInput(blob)
    return net.forward()


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


def display_objects(
    im: np.ndarray,
    objects: np.ndarray,
    label_filter: Set[str],
    polygons: List[Polygon],
    threshold: float = 0.25,
):
    rows = im.shape[0]
    cols = im.shape[1]

    # For every Detected Object
    for i in range(objects.shape[2]):
        # Find the class and confidence
        class_id = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        # Recover original coordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)

        # Check if the detection is of good quality
        if score > threshold and coco.class_labels[class_id] in label_filter:
            display_text(im, coco.class_labels[class_id], x, y)
            oy = rows - y
            object_rectangle = Polygon(
                shell=[(x, oy), (x + w, oy), (x + w, oy - h), (x, oy - h)]
            )
            if any(object_rectangle.intersects(polygon) for polygon in polygons):
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)

            cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)


def cv2_img_to_texture(frame):
    buf1 = cv2.flip(frame, 0)
    buf = buf1.tostring()
    image_texture = texture_create(
        size=(frame.shape[1], frame.shape[0]), colorfmt="bgr"
    )
    image_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
    return image_texture
