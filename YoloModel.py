from pathlib import Path
import time

import cv2
import numpy as np

from Model import Model


class YoloModel(Model):
    def __init__(self,
                 engine,
                 size=320,
                 config_path='models/YOLOv3/YOLOv3.cfg',
                 weights_path='models/YOLOv3/YOLOv3.weights'):

        self.size = size
        self.confThreshold = 0.5
        self.nmsThreshold = 0.4

        assert size in [
            320, 416, 608], f'Net size {size} not in [320, 416, 608]'
        assert Path(config_path).is_file() and Path(
            weights_path).is_file(), 'Not find config or weights file'

        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

        def setPreferableEngine(engine):
            if engine == 'gpu':
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        setPreferableEngine(engine)

        def getOutputsNames(net):
            layersNames = net.getLayerNames()
            return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        self.outputs = getOutputsNames(self.net)

        self.preprocess_time = 0
        self.inference_time = 0
        self.postprocess_time = 0
        self.count = 0

    def preprocess(self, frames):
        return frames

    def inference(self, frames):
        blob = cv2.dnn.blobFromImages(images=frames, scalefactor=1./255., size=(
            self.size, self.size), mean=(0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        return self.net.forward(self.outputs)

    def predict(self, frames):
        st = time.time()
        frames = self.preprocess(frames)
        self.preprocess_time += (time.time() - st)

        st = time.time()
        outs = self.inference(frames)
        self.inference_time  += (time.time() - st)

        st = time.time()
        a, b, c = outs
        if len(frames) == 1:
            a, b, c = [a], [b], [c]

        boxes = []
        for i in range(len(frames)):
            box = self.postprocess(frames[i], (a[i], b[i], c[i]))
            boxes.append((frames[i], box))

        self.postprocess_time += (time.time() - st)

        self.count += len(frames)

        return boxes

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confThreshold, self.nmsThreshold)

        results = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            results.append((classIds[i], confidences[i],
                            left, top, left + width, top + height))

        return results

    def get_total_FPS(self):
        return self.count/(self.preprocess_time+self.inference_time+self.postprocess_time)

    def get_inference_FPS(self):
        return self.count/self.inference_time