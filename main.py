import numpy as np
import glob

import click
import cv2

from bench_utils import FPS_bench#, FLOPS_bench


def load_model(model_type, model_path):
    if model_type == 'yolo':
        print('[LOGS] Load YoloV3 model')
        net = cv2.dnn.readNetFromDarknet(glob.glob(model_path+'/*.cfg')[0], glob.glob(model_path+'/*.weights')[0])
    # elif model_type == 'efficientdet':
    #     print('[LOGS] Load Efficient-det model')
    #     net = cv2.dnn.readNetFromTensorflow(model_path)
    else:
        print('[LOGS] Model option not supported')
        print('[LOGS] Exit')
        exit(0)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    return net

def select_destination_engine(net, engine):
    if engine == 'cpu':
        print('[LOGS] Set preferable target engine to CPU')
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # elif engine == 'gpu':
    #     print('[LOGS] Set preferable target engine to GPU')
    #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    # elif engine == 'jetson':
    #     print('[LOGS] Set preferable target engine to Jetson (GPU FP16)')
    #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
    # elif engine == 'movidious':
    #     print('[LOGS] Set preferable target engine to Movidious (MYRIAD)')
    #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    else:
        print('[LOGS] Engine option not supported')
        print('[LOGS] Exit')
        exit(0)

    return net


@click.command()
@click.option('--model-type', default=None, help='Model type [yolo, efficientdet]', required=True)
@click.option('--model-path', default=None, help='Patch to model directory', required=True)
@click.option('--size', default=None, help='Size of images', required=True)
@click.option('--engine', default='cpu', help='Destination engine [cpu, gpu, jetson, movidious]')
def main(model_type, model_path, size, engine):
    size = int(size)

    net = load_model(model_type, model_path)
    net = select_destination_engine(net, engine)

    #https://www.videvo.net/video/mosque-at-the-roadside/2784/
    cap = cv2.VideoCapture('video/mosque.mp4')

    print('[LOGS] Calculate FPS')
    fps = FPS_bench(cap, net, size)
    print("[LOGS] approx. FPS: {:.2f}".format(fps.fps()))


if __name__ == '__main__':
    main()