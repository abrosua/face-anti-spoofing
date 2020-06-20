# Face Anti-Spoofing Detection using SSD and MobileNetV2

Face detection/recognition has been the most popular deep learning projects/researches for these past years. One of its daily application is the face verification feature to [perform tasks](https://support.google.com/pixelphone/answer/9517039?hl=en) on our devices (e.g., unlocking the device, signing in to some apps, confirming our payment, etc). However, this method could be prone to spoof attacks, in which the model could be fooled with the facial photograph of its respective user (i.e., using a printed or digital picture of the user, and many others face-spoofing attacks). Therefore, a facial anti-spoofing detection would be worth to develop for tackling this malicious problem.

This is the final machine learning project assignment of the [Bangk!t](https://events.withgoogle.com/bangkit/) program by Google, an exclusive machine learning academy led by Google, in collaboration with several Indonesian unicorn startups.

### Methodology
Due to the limited training resources (e.g., low computability and limited datasets), we **divided the model** into 2 models (**detector** and **classifier**) with sequential pipeline, as the following:

 - Single Shot Multibox Detector (**SSD**), with the pretrain face detection model, as the **detector**.
 - MobileNetV2, with **transfer learning**, as the **classifier**.

> Note: To simplify the problem, we used the built-in models that are available on OpenCV and TensorFlow Keras respectively.

In this case, we only trained the classifier model and used the detector directly on the inference stage. Moreover, a complete pipeline of facial anti-spoofing detector is recommended fore the future improvements, although it requires a new dataset that also provides non-close-up images (i.e., full or half body) with the corresponding bounding box or facial key points label.

To limit our scope of work, we decided to tune the optimizer hyperparameter only (e.g., learning rate, scheduler, etc) and the `class_weight` as it's the one that arguably impacts the performance the most.

> The dataset is highly imbalanced towards the spoof image, hence making class weight as an important training feature to avoid overfitting.


## Getting Started
### Prerequisites

Install the dependencies from the `requirements.txt`
```
pip install -r requirements.txt
```


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
