import numpy as np
import cv2

cap = cv2.VideoCapture(0)
count = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27:  # wait for ESC key to exit
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('calibration_pictures/{}.png'.format(count), frame)
        print("image wrote")
        count +=1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()