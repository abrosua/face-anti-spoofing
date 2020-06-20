# Face Anti-Spoofing Detection using SSD and MobileNetV2

In this case, we're tasked to solve one problem from any public datasets. Hence, we chose an image classification problem for chest x-ray images (normal or pneumonia). The dataset is provided by [Paul Mooney](https://github.com/paultimothymooney) on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

This is one of the machine learning project assignment of the [bangkit](https://events.withgoogle.com/bangkit/) program by Google, an exclusive machine learning academy led by Google, in collaboration with several Indonesian unicorn startups.


## Getting Started

### Methodology
Convolutional Neural Networks (CNN), is currently the best solution for handling Computer Vision problem. However, harnessing its full potential might be very resourceful. Therefore we decided to use [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) on popular CNN architecture. The model is sorted from ImageNet's image classification [leaderboard](https://paperswithcode.com/sota/image-classification-on-imagenet), since ImageNet is consisted of thousands of classes, thus might provides model with better generalization.

> Note: To simplify the problem, we used the [built-in models](https://www.tensorflow.org/api_docs/python/tf/keras/applications) that are available on TensorFlow Keras, and sorted by the ImageNet leaderboard.

Xception and VGG-16 network are chosen due to its performance and number of parameters. Since we're working on limited resources, "lighter" models are preferable.

To limit our scope of work, we decided to tune the optimizer hyperparameter only (e.g., learning rate, scheduler, etc) as it's the one that arguably impacts the performance the most.

### File Structure

 - `docs`--- supporting documentations
	 - Reference papers.
	 - Presentation slides.
 - `input` --- input directories
	 - `demo`--- test case (image and video format).
	 - `LCC_FASD` --- Face anti-spoofing dataset ([Timoshenko, et al. 2019](https://csit.am/2019/proceedings/PRIP/PRIP3.pdf)).
 - `output` --- training results storage (e.g., trained weights, training history, etc)
 - `pretrain` --- pretrain and trained model storage for inference
 - `train` --- training notebooks directory

### Prerequisites

Install the dependencies from the `requirements.txt`

```
pip install -r requirements.txt
```


## Built With
### Model

1. Single Shot Multibox Detector ([Liu, et al., 2015](https://arxiv.org/abs/1512.02325)); using the OpenCV's built-in SSD face detection model.
2. MobileNetV2 ([Sandler, et al., 2019](https://arxiv.org/abs/1801.04381))

### Modules
* [TensorFlow 2.1.0](https://www.tensorflow.org/)
* [OpenCV](https://opencv.org/)
* [imutils](https://github.com/jrosebr1/imutils/)
* [Google Colab](https://colab.research.google.com/)


## Authors

* **Faber Silitonga** --- [abrosua](https://github.com/abrosua)
* **Mutia Wahyuni** --- [github](https://github.com/)
* **Kevin Angelo** --- [github](https://github.com/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Liveness detection with OpenCV [tutorial](https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/) by Adrian Rosebrock.
* Face anti-spoofing open dataset by [Timoshenko, et al. 2019](https://csit.am/2019/proceedings/PRIP/PRIP3.pdf).
