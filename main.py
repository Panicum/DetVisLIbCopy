# -*- coding: utf-8 -*-
"""
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
    detVis.load_gt(TEST_GT_PATH, "darknet")
    detVis.plot_gt((53, 252, 23), 8)
    detVis.plot_detections("ENSEMBLE", 4, [(255, 221, 0), (189, 46, 255)], 0.75, no_labels=True)
    detVis.plot_detections("YOLO", 4, [(255, 121, 0), (189, 146, 255)], 0.75, no_labels=True)
    # detVis.plot_detections("FASTER_RCNN", 4, [(255, 102, 204), (0, 204, 255)], 0.75, no_labels=True)
    # detVis.plot_detections("RETINA_NET", 4, [(255, 153, 51), (0, 0, 204)], 0.75, no_labels=True)
    
    #detVis.plot_detections("ENSEMBLE", 4, [(255, 221, 0), (189, 46, 255)], 0.85, iou_calc=True)
    detVis.plot_legend()
    detVis.print_images()
    detVis.save_images(TEST_RESULTS_PATH, '_yolo_and_retina_frcnn_no_labels')
    #detVis.save_images(TEST_RESULTS_PATH, '_ensemble_iou')

    
    ara_classes_dict = {
    0 : 'Person',
    1 : 'Bicycle',
    2 : 'Car',
    3 : 'Motorcycle',
    4 : 'Bus',
    5 : 'Train',
    6 : 'Truck'
    }   
        
    TEST_ARA_IMAGES_PATH = './ARA_SAMPLES/images'
    TEST_ARA_YOLO_PATH = './ARA_SAMPLES/yolo_v4_dets'
    TEST_ARA_RESULTS_PATH = './ARA_SAMPLES/results'
    
    detVisAra = detVisLib.DetVis(ara_classes_dict)
    detVisAra.load_images(TEST_ARA_IMAGES_PATH)
    detVisAra.load_detections(TEST_ARA_YOLO_PATH, "darknet", "YOLOv4")
    detVisAra.plot_detections("YOLOv4",4, conf_thresh = 0.25, no_labels=True)
    detVisAra.plot_legend()
    detVisAra.print_images()
    detVisAra.save_images(TEST_ARA_RESULTS_PATH, '_result')
    
    