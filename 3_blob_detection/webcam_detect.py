import cv2
import numpy as np
import time

def contours(binary_img, min_area=15):
    contour_list, _ = cv2.findContours(binary_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    return [x for x in contour_list if (cv2.contourArea(x) >= min_area)]
    
def draw(img, contour_list):
    width = img.shape[0]
    height = img.shape[1]
    output_img = np.copy(img)
    x_list = []
    y_list = []
    for contour in contour_list:
        cv2.drawContours(output_img, contour, -1, color = (255, 255, 255), thickness = 3)

        rect = cv2.minAreaRect(contour)
        center, size, angle = rect
        center = np.array(center, dtype=np.int32)

        cv2.drawContours(output_img, [cv2.boxPoints(rect).astype(int)], -1, color = (0, 0, 255), thickness = 2)
        cv2.circle(output_img, center = center, radius = 3, color = (0, 0, 255), thickness = -1)

        x_list.append((center[0] - width / 2) / (width / 2))
        x_list.append((center[1] - width / 2) / (width / 2))
        
    return output_img

# Your filtering code here
def preprocess(img):
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grayimg = cv2.inRange(hsvimg, (100, 0, 0), (150, 255, 160))

    grayimg = cv2.GaussianBlur(grayimg, (15,15), 256)
    grayimg = cv2.erode(grayimg, np.ones((7,7)))
    return grayimg

if __name__ == '__main__':
    # If the camera doesn't work try edit this to "0"
    cap = cv2.VideoCapture(0)
    if (not cap.isOpened()):
        print("Could not open camera. Try change the device ID.")
        exit(1)
    
    fnum = 0
    while True:
        t_begin = time.time()
        err, frame = cap.read()
        
        processed = preprocess(frame)
        cts = contours(processed)
        with_labels = draw(frame,  cts)

        t_end = time.time()
        
        print("Frame {}: {:.3f} ms".format(fnum, (t_end - t_begin) * 1000.0))
        fnum = fnum + 1
        
        both = cv2.hconcat((cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR), with_labels))
        cv2.imshow('detector', both)
        
        if cv2.pollKey() > -1:
            cap.release()
            cv2.destroyAllWindows()
            break

    