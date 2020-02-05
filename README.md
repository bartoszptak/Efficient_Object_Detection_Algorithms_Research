# Efficient Object Detection Algorithms Research
I would like to compare the models for real time object detection and their performance.
In the future I am going to modify the code so that all networks can use the cv2.dnn module.

# Models
* YOLOv3 [[paper](https://arxiv.org/pdf/1804.02767.pdf)][[code](https://pjreddie.com/darknet/yolo/)]
* EfficientDet [[paper](https://arxiv.org/pdf/1911.09070.pdf)][[code](https://github.com/xuannianz/EfficientDet)]
* (future) VoVNet [[paper](https://arxiv.org/pdf/1904.09730v1.pdf)][code]

# Devices
* Notebook i5-8265U + 12GB RAM + NVIDIA MX230
* PC Core i5-8400 + 16GB RAM + NVIDIA GTX1060 6GB
* NVIDIA Jetson TX2

# Installation guide
* [OpenCV for CPU guide](https://github.com/bartoszptak/Efficient_Object_Detection_Algorithms_Research/blob/master/INSTALLATION_GUIDE.md#opencv-for-cpu-guide)
* [OpenCV for GPU guide](https://github.com/bartoszptak/Efficient_Object_Detection_Algorithms_Research/blob/master/INSTALLATION_GUIDE.md#opencv-for-gpu-guide)

* download modified EfficientDet repos
```
git clone git@github.com:bartoszptak/EfficientDet.git
cd EfficientDet/
mv EfficientDet/* .
rm -r EfficientDet/ inference.py
```

* download models
```
python download_models.py
```

* (optionaly) download and prepare dataset for benchmark
```
mkdir data && cd data
wget "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
tar -vxf VOCtest_06-Nov-2007.tar
cd VOCdevkit/VOC2007/JPEGImages/
# make dataset smaller
rm 00{2..9}*.jpg

```

# Results
INFO: TOTAL FPS = preprocessing + inference + postprocessing)

* ### TOTAL FPS - BATCH_SIZE=1
| Model        | Size | MX230 | GTX1060 | Jetson TX2 |
|--------------|:----:|:-----:|:-------:|:----------:|
| YOLOv3       |  416 |  6.33 | 15.48 |            |
|              |  608 |  3.17 |  7.58 |            |
| EfficientDet |  512 |  5.28 |  14.79 |            |
|              |  640 |  2.77 |  8.51  |            |

* ### TOTAL FPS - BATCH_SIZE=2
| Model        | Size | MX230 | GTX1060 | Jetson TX2 |
|--------------|:----:|:-----:|:-------:|:----------:|
| YOLOv3       |  416 |  6.68 |  16.05 |            |
|              |  608 |  3.21 |  7.75 |            |
| EfficientDet |  512 |  5.87 | 17.17  |            |
|              |  640 |  ERR  | 9.41  |            |

* ### INFERENCE FPS - BATCH_SIZE=1
| Model        | Size | MX230 | GTX1060 | Jetson TX2 |
|--------------|:----:|:-----:|:-------:|:----------:|
| YOLOv3       |  416 |  8.33 |  31.20 |            |
|              |  608 |  4.37 | 16.51 |            |
| EfficientDet |  512 | 11.65 | 33.12 |            |
|              |  640 |  5.86 |  18.79 |            |

* ### INFERENCE FPS - BATCH_SIZE=2
| Model        | Size | MX230 | GTX1060 | Jetson TX2 |
|--------------|:----:|:-----:|:-------:|:----------:|
| YOLOv3       |  416 |  9.01 |  34.61  |            |
|              |  608 |  4.50 |  17.23  |            |
| EfficientDet |  512 | 12.61 |  39.46  |            |
|              |  640 |  ERR  |  21.14  |            |
