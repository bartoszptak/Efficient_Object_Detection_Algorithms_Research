import numpy as np

import cv2
from imutils.video import FPS


def FPS_bench(cap, net, size):
    fps = FPS().start()

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        blob = cv2.dnn.blobFromImage(image=frame, scalefactor=1.0, size=(size,size), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)

        out = net.forward()

        fps.update()

    fps.stop()
    return fps