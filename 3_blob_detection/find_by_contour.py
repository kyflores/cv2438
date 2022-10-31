import cv2
import numpy as np

img = cv2.imread('img.jpg')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


binary_img = cv2.inRange(hsv_img, (45, 50, 80), (150, 255, 255))
contour_list, _ = cv2.findContours(binary_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

width = img.shape[0]
height = img.shape[1]

output_img = np.copy(img)

x_list = []
y_list = []
for contour in contour_list:
    # Ignore small contours that could be because of noise/bad thresholding
    if cv2.contourArea(contour) < 15:
       continue

    cv2.drawContours(output_img, contour, -1, color = (255, 255, 255), thickness = -1)

    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    center = tuple([int(dim) for dim in center]) # Convert to int so we can draw

    # Draw rectangle and circle
    cv2.drawContours(output_img, [cv2.boxPoints(rect).astype(int)], -1, color = (0, 0, 255), thickness = 2)
    cv2.circle(output_img, center = center, radius = 3, color = (0, 0, 255), thickness = -1)

    x_list.append((center[0] - width / 2) / (width / 2))
    x_list.append((center[1] - width / 2) / (width / 2))

cv2.imshow('hi', output_img)
cv2.waitKey(0)
