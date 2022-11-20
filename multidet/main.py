import argparse
import time

import torch
import numpy as np
import cv2
import detectors as dets

import submodules.sort.sort as sort

def draw(img, detections):
    # TODO placeholder skip if empty
    if len(detections) == 0:
        return img

    for d in detections:
        corners = [(d['corners'].astype(np.int32).reshape(4, 1, 2))]

        # Kind of misusing this, but I want boxes to all have different colors
        withbox = cv2.polylines(
            img,
            corners,
            isClosed=True,
            color = d['color'],
            thickness = 3)

    return withbox

def detect(opt):
    wc = opt.webcam

    cap = cv2.VideoCapture(wc)
    if (not cap.isOpened()):
        print("Could not open camera. Try change the device ID.")
        exit(1)

    apriltags = dets.AprilTagDetector(
        dets.C310_PARAMS,
        'tag36h11',
        0.168
    )
    yolov5 = dets.YoloV5TorchDetector('yolov5n.pt')

    max_age = 10
    min_hits = 3
    iou_thresh = 0.3
    tracker = sort.Sort(max_age=max_age,
                   min_hits=min_hits,
                   iou_threshold=iou_thresh)

    while True:
        t_begin = time.time()
        err, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        at_det = apriltags.detect(frame_gray)
        yolo_det = yolov5.detect(frame)

        all_dets = at_det + yolo_det

        sort_dets = [d['sort_xyxy'] for d in all_dets]
        if len(sort_dets) == 0:
            sort_dets = np.empty((0,5))
        else:
            sort_dets = np.stack(sort_dets)


        trackers = tracker.update(sort_dets)
        print(trackers)

        with_boxes = draw(frame, all_dets)
        cv2.imshow('detector', with_boxes)

        if cv2.pollKey() > -1:
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--webcam', type=int, default=0, help='/dev/videoX number')
    parser.add_argument('--tag-family', type=str, default='tag36h11', help='Apriltag family')
    opt = parser.parse_args()

    detect(opt)
    print('Detector exited.')
    exit(0)
