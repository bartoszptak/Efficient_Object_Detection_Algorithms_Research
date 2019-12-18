import numpy as np

import cv2
from utils.yolo_utils import postprocess as yolo_postprocess
from utils.yolo_utils import getOutputsNames as yolo_inputs

def play_camera(cap, net, size, model_type):
    if model_type == 'yolo':
        postprocess = yolo_postprocess
        inputs = yolo_inputs(net)

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        blob = cv2.dnn.blobFromImage(image=frame, scalefactor=1.0/255., size=(size,size), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)

        out = net.forward(inputs)
        postprocess(frame, out, show_boxes=True)

        cv2.imshow('Video', frame)
        cv2.waitKey(1)


