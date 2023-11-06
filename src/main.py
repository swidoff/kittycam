import sys
import cv2
import coco
from pathlib import Path
import numpy as np

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

model_dir = Path("models") / "ssd_mobilenet_v2_coco_2018_03_29"
model_file = model_dir / "frozen_inference_graph.pb"
config_file = model_dir / "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

net = cv2.dnn.readNetFromTensorflow(str(model_file), str(config_file))


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


def display_objects(im: np.ndarray, objects: np.ndarray, threshold: float = 0.25):
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
        if score > threshold:
            display_text(im, coco.class_labels[class_id], x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)


source = cv2.VideoCapture(s)

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break

    # label = f"Shape: {frame.shape}"
    # cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    objects = detect_objects(frame)
    display_objects(frame, objects)
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)
