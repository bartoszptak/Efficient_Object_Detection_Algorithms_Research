from pathlib import Path
import os
import time

import cv2
import numpy as np

from Model import Model
from EfficientDet.utils import preprocess_image
from EfficientDet.utils.anchors import anchors_for_shape


class EffModel(Model):
    def __init__(self,
                 engine,
                 size=640,
                 num_classes=20):
        super().__init__()

        self.size = size
        self.score_threshold = 0.5
        self.num_classes = num_classes

        assert size in [
            512, 640], f'Net size {size} not in [512, 640]'

        def setPreferableEngine(engine):
            if engine == 'gpu':
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        setPreferableEngine(engine)
        import tensorflow as tf
        tf.logging.set_verbosity(tf.logging.ERROR)

        from EfficientDet.model import efficientdet

        if size == 512:
            phi = 0
            weighted_bifpn = True
            model_path = 'models/EfficientDet/EfficientDet-d0/EfficientDet-d0.weights'
        else:
            phi = 1
            weighted_bifpn = False
            model_path = 'models/EfficientDet/EfficientDet-d1/EfficientDet-d1.weights'

        assert Path(model_path).is_file(), 'Not find model file'

        _, self.net = efficientdet(phi=phi,
                                   weighted_bifpn=weighted_bifpn,
                                   num_classes=self.num_classes,
                                   score_threshold=self.score_threshold)
        self.net.load_weights(model_path, by_name=True)

    def preprocess(self, frames):
        inputs, anchors, meta = [], [], []

        for frame in frames:
            image = frame[:, :, ::-1]
            h, w = image.shape[:2]

            image, scale, offset_h, offset_w = preprocess_image(
                image, image_size=self.size)
            inputs.append(image)
            anchors.append(anchors_for_shape((self.size, self.size)))
            meta.append((scale, h, w, offset_h, offset_w))

        return [np.array(inputs), np.array(anchors)], meta

    def inference(self, frames):
        boxes, scores, labels = self.net.predict_on_batch(frames)
        return boxes, scores, labels

    def predict(self, frames):
        st = time.time()
        inputs, meta = self.preprocess(frames)
        self.preprocess_time += (time.time() - st)

        st = time.time()
        outs = self.inference(inputs)
        self.inference_time += (time.time() - st)

        boxes = []
        for i in range(len(frames)):
            box = self.postprocess(
                frames[i], ((outs[0][i], outs[1][i], outs[2][i]), meta[i]))
            boxes.append((frames[i], box))

        self.postprocess_time += (time.time() - st)

        self.count += len(frames)

        return boxes

    def postprocess(self, frame, outs):
        (boxes, scores, labels), (scale, h, w, offset_h, offset_w) = outs

        boxes[:, [0, 2]] = boxes[:, [0, 2]] - offset_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - offset_h
        boxes /= scale
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)

        # select indices which have a score above the threshold
        indices = np.where(scores > self.score_threshold)[0]

        # select those detections
        boxes = boxes[indices]
        scores = scores[indices]
        labels = labels[indices]

        results = []
        for box, score, label in zip(boxes, scores, labels):
            xmin = int(round(box[0]))
            ymin = int(round(box[1]))
            xmax = int(round(box[2]))
            ymax = int(round(box[3]))
            results.append((int(label), score,
                            xmin, ymin, xmax, ymax))

        return results
