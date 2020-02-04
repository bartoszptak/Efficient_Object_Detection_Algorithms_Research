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
@click.option('--size', default=None, help='Size of images', required=True)
@click.option('--engine', default='cpu', help='Destination engine [cpu, gpu, jetson]')
def main(mode, model, size, engine):
    size = int(size)

    net = load_model(model, size, engine)

    img = cv2.imread('samples/sample.jpeg')
    img2 = cv2.imread('samples/sample.jpg')
    res = net.predict([img,img2])

    draw_all(res)

    cv2.imshow('a', img)
    cv2.waitKey(0)
    cv2.imshow('a', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    print(f'{net.get_total_FPS()}')
    print(f'{net.get_inference_FPS()}')

if __name__ == '__main__':
    main()
