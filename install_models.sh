#!/usr/bin/env bash

mkdir -p models
cd models || exit

#if [[ ! -f "tf_text_graph_ssd.py" ]]; then
#    curl --output tf_text_graph_ssd.py https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/tf_text_graph_ssd.py
#    curl --output tf_text_graph_common.py https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/tf_text_graph_common.py
#fi

#if [[ ! -d "ssd_mobilenet_v2_coco_2018_03_29" ]]; then
#    echo "Installing object detection models"
#
#    curl --output ssd_mobilenet_v2_coco_2018_03_29.tar.gz http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
#    tar xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
#    rm ssd_mobilenet_v2_coco_2018_03_29.tar.gz
#
#    python tf_text_graph_ssd.py --input ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb \
#        --output ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29.pbtxt \
#        --config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config

#    python tf_text_graph_ssd.py --input ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb \
#        --output ssd_mobilenet_v2_coco_2018_03_29/test.pbtxt \
#        --config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config
#fi
#
#
#if [[ ! -d "ssd_mobilenet_v2_coco_2018_03_29" ]]; then
    echo "Installing object detection models"

#    curl --output efficientdet_d4_coco17_tpu-32.tar.gz http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz
#    tar xvf efficientdet_d4_coco17_tpu-32.tar.gz
#    rm efficientdet_d4_coco17_tpu-32.tar.gz

#    python tf_text_graph_faster_rcnn.py --input faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8/saved_model/saved_model.pb \
#        --output faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.pbtxt \
#        --config faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8/pipeline.config

#    python tf_text_graph_efficientdet.py --input efficientdet_d4_coco17_tpu-32/saved_model/saved_model.pb \
        --output efficientdet_d4_coco17_tpu-32/efficientdet_d4_coco17_tpu-32.pbtxt \
        --min_level 3 \
        --num_scales 3 \
        --anchor_scale 4.0 \
        --num_classes 90 \
        --width 1024 \
        --height 1024
#fi
