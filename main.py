import numpy as np
import glob

import click
import cv2

from utils.bench_utils import FPS_bench, FLOPS_bench
from utils.video_utils import play_camera


def load_model(model_type, model_path, engine):
    if model_type == 'yolo':
        print('[LOGS] Load YoloV3 model')
        net = cv2.dnn.readNetFromDarknet(glob.glob(model_path+'/*.cfg')[0], glob.glob(model_path+'/*.weights')[0])
    elif model_type == 'efficientdet':
        print('[LOGS] Load Efficient-det model')
        net = cv2.dnn.readNetFromTensorflow(glob.glob(model_path+'/*.pb')[0])
    elif model_type == 'vovnet':
        print('[LOGS] Load VoVNet model')
        net = cv2.dnn.readNetFromCaffe(glob.glob(model_path+'/deploy.prototxt')[0], glob.glob(model_path+'/model.caffemodel')[0])
    else:
        print('[LOGS] Model option not supported')
        print('[LOGS] Exit')
        exit(0)

    if engine == 'gpu' or engine=='jetson':
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    return net

def select_destination_engine(net, engine):
    if engine == 'cpu':
        print('[LOGS] Set preferable target engine to CPU')
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    elif engine == 'gpu':
        print('[LOGS] Set preferable target engine to GPU CUDA')
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    elif engine == 'jetson':
        print('[LOGS] Set preferable target engine to Jetson (GPU FP16)')
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    elif engine == 'movidious':
        print('[LOGS] Set preferable target engine to Movidious (MYRIAD)')
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    else:
        print('[LOGS] Engine option not supported')
        print('[LOGS] Exit')
        exit(0)

    return net


@click.command()
@click.option('--mode', default='test', help='Run mode [test, benchmark]', required=True)
@click.option('--model-type', default=None, help='Model type [yolo, efficientdet, vovnet]', required=True)
@click.option('--model-path', default=None, help='Patch to model directory', required=True)
@click.option('--size', default=None, help='Size of images', required=True)
@click.option('--engine', default='cpu', help='Destination engine [cpu, gpu, jetson, movidious]')
def main(mode, model_type, model_path, size, engine):
    size = int(size)

    net = load_model(model_type, model_path, engine)
    print(f'[LOGS] Set network size to {size}x{size}x3')
    net = select_destination_engine(net, engine)

    if mode == 'test':
        print('[LOGS] Run camera')
        cap = cv2.VideoCapture('video/mosque.mp4')
        play_camera(cap, net, size, model_type)
    elif mode == 'benchmark':
        print('[LOGS] Calculate GFLOPS')
        flops = FLOPS_bench(net, size)
        print("[LOGS] GFLOPS: {:.2f}".format(flops))

        #https://www.videvo.net/video/mosque-at-the-roadside/2784/
        cap = cv2.VideoCapture('video/mosque.mp4')

        print('[LOGS] Calculate FPS')
        fps = FPS_bench(cap, net, size, model_type)
        print("[LOGS] Approx. FPS: {:.2f}".format(fps.fps()))
    else:
        print('[LOGS] Run mode option not supported')


if __name__ == '__main__':
    main()