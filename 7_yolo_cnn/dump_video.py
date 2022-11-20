import cv2
import os
import sys

if __name__ == '__main__':
    fname = sys.argv[1]
    decimate = int(sys.argv[2])
    
    basenm = os.path.basename(fname)
    nm, ext = os.path.splitext(basenm)
    dirnm = os.path.dirname(fname)
    
    cap = cv2.VideoCapture(fname)
    
    more = True
    count = 0 
    os.makedirs(nm, exist_ok=True)
    while(more):
        more, fr = cap.read()
        if fr is None:
            break
        
        if (count % decimate) == 0:
            tmp = "{}/{}_{}.jpg".format(nm, nm, count)
            cv2.imwrite(tmp, fr)
        
        count = count + 1
            
    print("Done, saved {} images".format(count))

