import os
import argparse
import time

import torch
import torchvision.ops as ops
import numpy as np
import cv2

import submodules.sort.sort as sort

import detectors as dets
import draw
import camera

def map_to_sort_id(dets, trackers):
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
        dets[ix]['sort_id'] = int(trackers[m][-1])

def detect(opt):
    # See https://www.kurokesu.com/main/2020/05/22/uvc-camera-exposure-timing-in-opencv/
    try:
        source = int(opt.source)
        camera.config_gain_exposure(source, opt.gain, opt.exposure)
        print("Configured webcam {} as source".format(source))
    except ValueError:
        source = str(opt.source)
        print("Configured file {} as source".format(source))
    except:
        print("Unknown error parsing source")
        exit(1)

    cap = cv2.VideoCapture(source)
    if (not cap.isOpened()):
        print("Could not open source. Try change the device ID.")
        exit(1)

    # These seem to kind of work, GAIN doesn't appear to apply?
    # TODO wrap v4l2-ctl b/c that's more reliable

    apriltags = dets.AprilTagDetector(
        dets.C310_PARAMS,
        opt.tag_family,
        opt.tag_size
    )
    yolov5 = dets.YoloV5TorchDetector(opt.weights)

    min_hits = 1
    tracker = sort.Sort(max_age=opt.max_age,
                   min_hits=min_hits,
                   iou_threshold=opt.iou_thresh)

    while True:
        t_begin = time.time()
        err, frame = cap.read()
        if (not err):
            print("Media source didn't produce frame, stopping...")
            break

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
            map_to_sort_id(all_dets, trackers) # Mutates the dictionary

        with_boxes = draw.draw(frame, all_dets)
        # with_boxes = draw.draw_sort(frame, trackers)
        cv2.imshow('detector', with_boxes)

        t_end = time.time()

        os.system('cls' if os.name == 'nt' else 'clear')
        print("{:.2f} iters/s".format(1 / (t_end - t_begin)))
        print(dets.detections_as_table(all_dets))

        if cv2.pollKey() > -1:
            cap.release()
            cv2.destroyAllWindows()
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=0, help='Media source, anything supported by video capture')
    parser.add_argument('--gain', type=int, default=150, help='Gain to configure with v4l2-ctl')
    parser.add_argument('--exposure', type=int, default=500, help='Exposure time to configure with v4l2-ctl')
    parser.add_argument('--tag_family', type=str, default='tag36h11', help='Apriltag family')
    parser.add_argument('--tag_size', type=float, default=1.0, help='Apriltag size in meters')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Path to YOLO weights file')
    parser.add_argument(
        '--max_age',
        type=int,
        default=30,
        help='Longest time (in frames) SORT will remember an ID without a detection')
    parser.add_argument(
        '--iou_thresh',
        type=float,
        default='0.15',
        help='IOU threshold for SORT. Smaller can track faster movements but reduces accuracy')


    opt = parser.parse_args()

    detect(opt)
    print('Detector exited.')
    exit(0)

if __name__ == '__main__':
    main()
