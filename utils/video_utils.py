import numpy as np

import cv2

def play_camera(cap, net, size, model_type):
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

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        frame = forward(net, outputs, frame, size, show_boxes=True)

        cv2.imshow('Video', frame)
        cv2.waitKey(1)


