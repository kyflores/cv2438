import cv2
import sys

def preprocess(frame):
    # HSV threshold
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    (h_l, s_l, v_l) = (0, 0, 0)  # HSV lower bounds
    (h_u, s_u, v_u) = (255, 255, 128)  # HSV upper bounds. 255 is the max 8bit value.
    img = cv2.inRange(img, (h_l, s_l, v_l), (h_u, s_u, v_u))

    # Erode
    img = cv2.erode(
        img,
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),
    )

    # Blur
    img = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=1.0)

    return img

# TODO
def detect(frame):
    # Find Blobs
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = 1
    params.blobColor = (0 if dark_blobs else 255)
    params.minThreshold = 10
    params.maxThreshold = 220
    params.filterByArea = True
    params.minArea = min_area
    params.filterByCircularity = True
    params.minCircularity = circularity[0]
    params.maxCircularity = circularity[1]
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    res = detector.detect(img)
    return res

# Usage:
#   detect.py <image.{jpg | png | ... }
#   detect.py webcam
if __name__ == '__main__':
    if sys.argv[1] == "webcam":
        cap = cv2.VideoCapture(1)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            frame = preprocess(frame)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        frame = cv2.imread(sys.argv[1])
        frame = preprocess(frame)
        cv2.imwrite('out.png', frame)

