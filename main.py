# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:27:16 2021

@author: gasie
"""


import detVisLib

TEST_IMAGES_PATH = './AFO_SAMPLES/images'
TEST_GT_PATH = './AFO_SAMPLES/gt'
TEST_YOLO_DETS_PATH = './AFO_SAMPLES/yolo_v3_dets'
TEST_DETECTRON_RETINA_DETS_PATH = './AFO_SAMPLES/retina_dets'
TEST_DETECTRON_FRCNN_DETS_PATH = './AFO_SAMPLES/frcnn_dets'
TEST_COCO_ENSEMBLE_DETS_PATH = './AFO_SAMPLES/ensemble_coco_results.json'
TEST_RESULTS_PATH = './AFO_SAMPLES/results'


if __name__ == "__main__":
    classes_dict = {
    0 : 'Small Object',
    1 : 'Large Object',
    }   
    detVis = detVisLib.DetVis(classes_dict)
    detVis.load_images(TEST_IMAGES_PATH)
    detVis.load_detections(TEST_YOLO_DETS_PATH, "darknet", "YOLO")
    detVis.load_detections(TEST_DETECTRON_RETINA_DETS_PATH, "detectron", "RETINA_NET")
    detVis.load_detections(TEST_DETECTRON_FRCNN_DETS_PATH, "detectron", "FASTER_RCNN")
    detVis.load_detections(TEST_COCO_ENSEMBLE_DETS_PATH, "coco", "ENSEMBLE")
    detVis.load_gt(TEST_GT_PATH, "xywh")
    detVis.plot_gt((53, 252, 23), 8)
    detVis.plot_detections("ENSEMBLE", 4, [(255, 221, 0), (189, 46, 255)], 0.25)
    # detVis.plot_detections("YOLO", 4, [(255, 221, 0), (189, 46, 255)], 0.25, no_labels=True)
    # detVis.plot_detections("FASTER_RCNN", 4, [(255, 102, 204), (0, 204, 255)], 0.25, no_labels=True)
    # detVis.plot_detections("RETINA_NET", 4, [(255, 153, 51), (0, 0, 204)], 0.25, no_labels=True)
    detVis.print_images()
    detVis.save_images(TEST_RESULTS_PATH, 'yolo_and_retina_frcnn_no_labels')