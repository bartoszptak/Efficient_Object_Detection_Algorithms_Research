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

### TOTAL FPS - BATCH_SIZE=1
| Model        | Size | MX230 | GTX1060 | Jetson TX2 |
|--------------|:----:|:-----:|:-------:|:----------:|
| YOLOv3       |  416 |  6.33 |         |            |
|              |  608 |  3.17 |         |            |
| EfficientDet |  512 |  5.28 |         |            |
|              |  640 |  2.77 |         |            |

### TOTAL FPS - BATCH_SIZE=2
| Model        | Size | MX230 | GTX1060 | Jetson TX2 |
|--------------|:----:|:-----:|:-------:|:----------:|
| YOLOv3       |  416 |  6.68 |         |            |
|              |  608 |  3.21 |         |            |
| EfficientDet |  512 |  5.87 |         |            |
|              |  640 |  ERR  |         |            |

### INFERENCE FPS - BATCH_SIZE=1
| Model        | Size | MX230 | GTX1060 | Jetson TX2 |
|--------------|:----:|:-----:|:-------:|:----------:|
| YOLOv3       |  416 |  8.33 |         |            |
|              |  608 |  4.37 |         |            |
| EfficientDet |  512 | 11.65 |         |            |
|              |  640 |  5.86 |         |            |

### INFERENCE FPS - BATCH_SIZE=2
| Model        | Size | MX230 | GTX1060 | Jetson TX2 |
|--------------|:----:|:-----:|:-------:|:----------:|
| YOLOv3       |  416 |  9.01 |         |            |
|              |  608 |  4.50 |         |            |
| EfficientDet |  512 | 12.61 |         |            |
|              |  640 |  ERR  |         |            |
