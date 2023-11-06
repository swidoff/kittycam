#!/usr/bin/env bash

mkdir -p models
cd models || exit

if [[ ! -f "tf_text_graph_ssd.py" ]]; then
    curl --output tf_text_graph_ssd.py https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/tf_text_graph_ssd.py
    curl --output tf_text_graph_common.py https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/tf_text_graph_common.py
fi

if [[ ! -d "ssd_mobilenet_v2_coco_2018_03_29" ]]; then
    echo "Installing object detection models"

    curl --output ssd_mobilenet_v2_coco_2018_03_29.tar.gz http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    tar xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    rm ssd_mobilenet_v2_coco_2018_03_29.tar.gz

    python tf_text_graph_ssd.py --input ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb \
        --output ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29.pbtxt \
        --config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config
fi

