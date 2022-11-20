import argparse
import time

import torch
import torchvision.ops as ops
import numpy as np
import cv2
import detectors as dets

import submodules.sort.sort as sort

def map_to_sort_id(dets, trackers):
    id_num, _ = trackers.shape
    xyxys = [torch.tensor(x['sort_xyxy'][:4]) for x in dets]

    # Collect all the detections into an (N, 4)
    det_c = torch.stack(xyxys)
    # Get track into an (M, 4)
    trk_c = torch.from_numpy(trackers[..., :4])

    ious = ops.box_iou(det_c, trk_c)

    # N (detections) is on rows, M (sort boxes) on columns.
    # dim=1 reduces horizontally, giving the max column index for each row
    # The index of the max IOU is the index in the sort result that
    # best matches that particular index in the detection result.
    mins = torch.argmax(ious, dim=1)
    for ix, m in enumerate(mins):
        dets[ix]['sort_id'] = trackers[m][-1]


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

        newid = d.get('sort_id', -1)
        cv2.putText(withbox,
            'ID: {}'.format(newid),
            d['corners'][0].astype(np.uint32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            1)

    return withbox

def detect(opt):
    # I recommend that you set the camera to the highest frame rate that yields
    # an acceptable exposure.
    #
    # For instance...
    # v4l2-ctl -d 3 -c exposure_time_absolute=166
    #
    # Use v4l2-ctl --all -d <devicenum> to find the appropriate control
    # There's probably also an auto exposure mode you need to disable.

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
    yolov5 = dets.YoloV5TorchDetector('yolov5s.pt')

    max_age = 60
    min_hits = 3
    iou_thresh = 0.15
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

        # Comes back as an `(x, 5) array` in
        # (xmin, xmax, ymin, ymax, ID)
        trackers = tracker.update(sort_dets)

        # Go back through the array and assign SORT IDs to the boxes
        # based on IOU with SORT boxes.
        if trackers.shape[0] > 0:
            newid = map_to_sort_id(all_dets, trackers)
            # det['sort_id'] = newid

        with_boxes = draw(frame, all_dets)
        cv2.imshow('detector', with_boxes)

        t_end = time.time()
        print("{} iters/s".format(1 / (t_end - t_begin)))
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
