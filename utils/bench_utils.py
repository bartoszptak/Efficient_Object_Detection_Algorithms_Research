import numpy as np

import cv2
from imutils.video import FPS
from utils.yolo_utils import postprocess as yolo_postprocess
from utils.yolo_utils import getOutputsNames as yolo_inputs

def FLOPS_bench(net, size):
    return net.getFLOPS([1, 3, size, size])/10**9

def FPS_bench(cap, net, size, model_type):
    if model_type == 'yolo':
        postprocess = yolo_postprocess
        inputs = yolo_inputs(net)
        
    fps = FPS().start()

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        
        blob = cv2.dnn.blobFromImage(image=frame, scalefactor=1.0/255., size=(size,size), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)

        out = net.forward(inputs)
        _ = postprocess(frame, out, show_boxes=False)

        fps.update()

    fps.stop()
    return fps