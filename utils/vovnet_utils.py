import cv2
import numpy as np

from coco_names import coco_names as classes

threshold = 0.6

def vovnet_getOutputsNames(net):
    return [i for i in net.getLayerNames() if i == 'detection_out']

    
def vovnet_postprocess(frame, outs, show_boxes=False):
    det_label = outs[:, 1]
    det_conf = outs[:, 2]
    det_xmin = outs[:, 3] * frame.shape[1]
    det_ymin = outs[:, 4] * frame.shape[0]
    det_xmax = outs[:, 5] * frame.shape[1]
    det_ymax = outs[:, 6] * frame.shape[0]
    results = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])

    if show_boxes:
        vovnet_draw(frame, results)


def vovnet_draw(frame, results):
    for i in range(0, results.shape[0]):
        print(results[i, -1])
        score = results[i, -2]
        if threshold and score < 0.01:
            continue

        # print(score)
        xmin = int(round(results[i, 0]))
        ymin = int(round(results[i, 1]))
        xmax = int(round(results[i, 2]))
        ymax = int(round(results[i, 3]))
        
        # Draw a bounding box.
        cv2.rectangle(frame, (xmin, ymin), (xmax - xmin, ymax - ymin), (0, 0, 255))
        
        # label = '%.2f' % conf
        # label = '%s:%s' % (classes[classId], label)

        # # Display the label at the top of the bounding box
        # labelSize, baseLine = cv2.getTextSize(
        #     label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # top = max(top, labelSize[1])
        # cv2.putText(frame, label, (left, top),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))



def vovnet_forward(net, outputs, frame, size, show_boxes):
    blob = cv2.dnn.blobFromImage(image=frame, scalefactor=1.0/255., size=(size,size), mean=(104, 117, 123), swapRB=True, crop=False)
    net.setInput(blob)

    out = net.forward(outputs)
    vovnet_postprocess(frame, out[0][0][0], show_boxes)
    return frame
