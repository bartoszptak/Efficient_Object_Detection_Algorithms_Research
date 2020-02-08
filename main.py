import numpy as np
import glob

import click
import cv2

from Model import Model
from utils import draw_all

def load_model(model:str, size:int, engine:str) -> Model:
    if model == 'yolo':
        from YoloModel import YoloModel
        return YoloModel(engine, size)
    elif model == 'eff':
        from EffModel import EffModel
        return EffModel(engine, size)
    else:
        raise NotImplementedError


@click.command()
@click.option('--mode', default='test', help='Run mode [test, benchmark]', required=True)
@click.option('--model', default=None, help='Model type [yolo, efficientdet, vovnet]', required=True)
@click.option('--engine', default='cpu', help='Destination engine [cpu, gpu, jetson]')
@click.option('--batch-size', default='1', help='Batch size of test')

def main(mode, model, engine, batch_size):
    size = 512
    batch_size = int(batch_size)

    net = load_model(model, size, engine)

    if mode == 'test':
        img = cv2.imread('samples/sample.jpg')
        res = net.predict([img])
        draw_all(res, model)

        cv2.imshow('a', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        from glob import glob
        from tqdm import tqdm
        imgs = glob('data/VOCdevkit/VOC2007/JPEGImages/*.jpg')

        for i in tqdm(range(0,1,batch_size)):
            ims = [cv2.imread(im) for im in imgs[i:i+batch_size]]
            res = net.predict(ims)

    print(f'Total imgs: {net.count:.2f}')
    print(f'GFLOPS: {net.get_GFLOPS():.4f}')
    print(f'Total FPS: {net.get_total_FPS():.2f}')
    print(f'Inference FPS: {net.get_inference_FPS():.2f}')

if __name__ == '__main__':
    main()
