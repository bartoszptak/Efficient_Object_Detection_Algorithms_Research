import numpy as np

import cv2
from imutils.video import FPS

def FLOPS_bench(net, size):
    return net.getFLOPS([1, 3, size, size])/10**9

def FPS_bench(cap, net, size, model_type):
    if model_type == 'yolo':
        from utils.yolo_utils import yolo_getOutputsNames, yolo_forward
        outputs = yolo_getOutputsNames(net)
        forward = yolo_forward
    elif model_type == 'vovnet':
        from utils.vovnet_utils import vovnet_getOutputsNames, vovnet_forward
        outputs = vovnet_getOutputsNames(net)
        forward = vovnet_forward
    else:
        print('[LOGS] Net inference not implemented yet')
        raise NotImplementedError
        
    fps = FPS().start()

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        
        frame = forward(net, outputs, frame, size, show_boxes=False)

        fps.update()

    fps.stop()
    return fps