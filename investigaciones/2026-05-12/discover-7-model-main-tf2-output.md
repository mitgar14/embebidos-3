## Track B: Descubrimiento

**Tema:** tensorflow object_detection model_main_tf2 step loss log output regex python 3.10

**Resultados:** 52


### 3sidedcube.com

- [TensorFlow Object Detection Training 101 - 3 Sided Cube](https://3sidedcube.com/blog/guide-retraining-object-detection-models-tensorflow)
  > cd ~/pipModel/models [...] oc object_detection [...] python xml_ [...] csv.py - [...] ~/pipModel/annotations/train -out train [...] python xml_ [...] python generate_tfrecord.py --input_csv=train.csv 

### ar5iv.labs.arxiv.org

- [[1712.00726] Cascade R-CNN: Delving into High Quality Object Detection](https://ar5iv.labs.arxiv.org/html/1712.00726)
  > --- | --- [...] relies on a [...] As shown in the left of Figure 4, the distribution of the initial hypotheses, e.g. RPN proposals, is heavily tilted towards low quality. This inevitably induces ineff

### arxiv.org

- [YOLOv12: A Breakdown of the Key Architectural Features](https://arxiv.org/html/2502.14740v1)
  > Darknet-53 [...] | Darknet | [...] [14] | 2020 | Object Detection, Instance Segmentation | Anchor-free detection, SWISH activation, PANet | PyTorch | | YOLOv6 [15] | 2022 | Object Detection, Instance 
- [A PyTorch Library of Turing-Complete Neural Networks](https://arxiv.org/html/2605.08150)
  > state vectors $ [...] }]$ , one per [...] Each vector $ [...] number, the head position, [...] space for intermediate [...] , a position [...] $\beta(t)$ encoding the [...] Transition (feedforward, Le

### blog.tensorflow.org

- [Ecovacs Robotics: the AI robotic vacuum cleaner powered by TensorFlow — The TensorFlow Blog](https://blog.tensorflow.org/2020/01/ecovacs-robotics-ai-robotic-vacuum.html?m=1)
  > Based on the focal loss[7], we designed our loss function with weights to different anchor boxes. Unlike the origin focal loss, which only gives different weights to easy and hard samples, our model a
- [What's new in TensorFlow 2.20 — The TensorFlow Blog](https://blog.tensorflow.org/2025/08/whats-new-in-tensorflow-2-20.html)
  > TensorFlow 2 [...] 20 has been released! For ongoing updates related to the multi-backend Keras, please note that all news and releases, starting with Keras 3.0, are now published directly on keras.io

### borg.csueastbay.edu

- [CS663](http://borg.csueastbay.edu/~grewe/CS663/Mat/TensorFlow/ObjectDetectionApi/Train_Validate_ObjDetectAPI.html)
  > #### Once you prepare the configuration file and your Input TFRecord files for both Training dataset and Validation dataset, you train by invoking the model_main_tf2.py script. Some parameters include

### codeofpaper.com

- [Floorplan-SLAM: A Real-Time, High-Accuracy, and Long-Term Multi-Session Point-Plane SLAM for Efficient Floorplan Reconstruction | Code of Paper](https://codeofpaper.com/paper/2503.00397)
  > using Neural Networks (SSD) on Tensorflow. This repo documents steps and scripts used to train [...] **efficiently** interleave output form multiple pretrained models for various object classes and ha

### cv-tricks.com

- [Training Object Detectors using TensorFlow Object Detection API](https://cv-tricks.com/how-to/training-object-detectors-using-tensorflow-object-detection-api/)
  > We will be working in the Colab environment, so we do not need to break our heads over installing dependencies. We will be using protobufs to compile binaries within the TensorFlow object detection AP

### esri.com

- [Esri UC 2021: Raster analytics and deep learning in ArcGIS Online](https://www.esri.com/arcgis-blog/products/arcgis-online/imagery/esri-uc-2021-raster-analytics-and-deep-learning-in-arcgis-online)
  > ArcGIS Image for ArcGIS Online comes with a collection of raster analysis tools, which includes tools to perform deep learning. The Detect Objects Using Deep Learning and Classify Pixels using Deep Le

### estudogeral.uc.pt

- [](https://estudogeral.uc.pt/bitstream/10316/103124/4/Tese_Carlos%20Neto.pdf)
  > 3.5.1 Loss . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19 [...] Regarding the process of training a neural network model the work was done using TensorFlow [...] Tens

### eujournal.org

- [](https://eujournal.org/index.php/esj/article/view/18335/18184)
  > With all of these components in place, the next step is to incorporate  Tensor [...] and TensorFlow object detection into our code and software;  this will serve as the foundation for training the pre

### geniusjournals.org

- [](https://geniusjournals.org/index.php/erb/article/download/5495/4615/5335)
  > Overall, text-to-image generation is an interdisciplinary field that combines natural language understanding with computer vision, aiming to bridge the semantic gap between textual descriptions and vi

### gilberttanner.com

- [Tensorflow Object Detection with Tensorflow 2: Creating a custom model | Gilbert Tanner](https://gilberttanner.com/blog/tensorflow-object-detection-with-tensorflow-2-creating-a-custom-model/)
  > ## Train model [...] To train the model, execute the following command in the command line: [...] ``` python model_main_tf2.py \     --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_

### github.com

- [research/object_detection/g3doc/tf2_training_and_evaluation.md at master · tensorflow/models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md)
  > ```md # Training and Evaluation with TensorFlow 2  TensorFlow 2.2 Python 3.6 [...] This page walks through the steps required to train an object detection model. It assumes [...] it is recommended tha
- [[object_detection]How to see accuracy and loss? · Issue #8021 · tensorflow/models](https://github.com/tensorflow/models/issues/8021)
  > ## [object_detection]How to see accuracy and loss? [...] **Problem description**： [...] When I use this command to train:  > python model_main.py --pipeline_config_path=training/ssd_inception_v2_coco.
- [README.md at master · TannerGilbert/Tensorflow-Object-Detection-API-Train-Model](https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model/blob/master/README.md)
  > To train the model, execute the following command in the command line: [...] ```bash python model_main_tf2.py --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config --model_di
- [`MODEL_DIR` in TensorFlow Object Detection `model_main_tf2.py` · Issue #10806 · tensorflow/models](https://github.com/tensorflow/models/issues/10806)
  > ## `MODEL_DIR` in TensorFlow Object Detection `model_main_tf2.py` [...] In the following line of `model_main_tf2.py`: https://github.com/tensorflow/models/blob/1afc1e5af7143adf541da55c1142e2e800bc67bb
- [research/object_detection/g3doc/tf2.md at master · tensorflow/models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)
  > ```md # Object Detection API with TensorFlow 2  ## Requirements  Python 3.6 TensorFlow 2.2 Protobuf Compiler >= 3.0  ## Installation [...] You can install the TensorFlow Object Detection API either wi
- [research/object_detection/model_main_tf2.py at d3bd8c0d535783c6cd9d676aaf0b1ba8d869b9c9 · tensorflow/models](https://github.com/tensorflow/models/blob/d3bd8c0d535783c6cd9d676aaf0b1ba8d869b9c9/research/object_detection/model_main_tf2.py)
  > # File: tensorflow/models/research/object_detection/model_main_tf2.py [...] r"""Creates and runs TF2 object detection models.  For local training/evaluation run: PIPELINE_CONFIG_PATH=path/to/pipeline.
- [research/object_detection/model_main_tf2.py at 88499586 · tensorflow/models](https://github.com/tensorflow/models/blob/88499586/research/object_detection/model_main_tf2.py)
  > # File: tensorflow/models/research/object_detection/model_main_tf2.py [...] r"""Creates and runs TF2 object detection models.  For local training/evaluation run: PIPELINE_CONFIG_PATH=path/to/pipeline.
- [Running object detection with: Tensorflow/models/research/object_detection/model_main_tf2.py stops after 2000 steps when there are more than 2000 .jpg files  · Issue #10634 · tensorflow/models](https://github.com/tensorflow/models/issues/10634)
  > ## Running object detection with: Tensorflow/models/research/object_detection/model_main_tf2.py stops after 2000 steps when there are more than 2000 .jpg files [...] Based on model "ssd_mobilenet_v1_f
- [models/research/object_detection/model_main_tf2.py at master](https://github.com/tensorflow/models/blob/master/research/object_detection/model_main_tf2.py)
  > # File: tensorflow/models/research/object_detection/model_main_tf2.py [...] r"""Creates and runs TF2 object detection models.  For local training/evaluation run: PIPELINE_CONFIG_PATH=path/to/pipeline.
- [sourangshupal/Tensorflow2-Object-Detection-Tutorial](https://github.com/sourangshupal/Tensorflow2-Object-Detection-Tutorial)
  > So we are ready to start the training.  **model_main_tf2.py** is the file needed to start the training.  !python model_main_tf2.py --model_dir=training --pipeline_config_path=training/pipeline.config 
- [Understanding pipeline.config · Issue #10597 · tensorflow/models](https://github.com/tensorflow/models/issues/10597)
  > ``` model {   center_net {     num_classes: 90     feature_extractor {       type: "hourglass_104"       channel_means: 104.01361846923828       channel_means: 114.03422546386719       channel_means: 
- [research/object_detection/README.md at master · tensorflow/models](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md)
  > ```md # TensorFlow Object Detection API [...] TensorFlow 2.2 TensorFlow 1.15 Python 3.6 [...] The TensorFlow Object Detection API is an open source framework built on top of TensorFlow that makes it e
- [research/object_detection/g3doc/configuring_jobs.md at master · tensorflow/models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md)
  > The TensorFlow Object Detection API uses protobuf files to configure the training and evaluation process. The schema for the training pipeline can be found in object_detection/protos/pipeline.proto. A
- [Urgent help needed. · Issue #85074 · tensorflow/tensorflow - GitHub](https://github.com/tensorflow/tensorflow/issues/85074)
  > And am using python 3.10.10. [...] Actually I used python 3.12 but I couldn't install tensorflow-io package with python 3.12. Not sure reason yet. [...] So I use python 3.10 now. [...] cd research  in
- [Make steps between logging configurable for object detection training · Issue #10521 · tensorflow/models](https://github.com/tensorflow/models/issues/10521)
  > ## Make steps between logging configurable for object detection training [...] Currently it looks like the number of steps between logging in `object_detection.model_lib_v2.train_loop` is hardcoded to
- [a64bit/tf2-object-detection-api-tutorial](https://github.com/a64bit/tf2-object-detection-api-tutorial)
  > The examples in this repo is tested with python 3.6 and Tensorflow 2.2.0, but it is expected to work with other Tensorflow 2.x versions with python version 3.5 or higher. [...] ```bash out_dir=../mode

### huggingface.co

- [Paper page - Towards Inadequately Pre-trained Models in Transfer Learning](https://huggingface.co/papers/2203.04668)
  > ^{b \times [...] and the residual [...] sum over all [...] = \sum [...] i=1}^K \ [...] , and then determine the minimum $k$ that satisfies [...] ^k / [...] ^K \geq [...] 0.8$ . $\ [...] _m$ preserves 

### kaggle.com

- [SSD MOBILENET V2 | Python 3.10.12 - Kaggle](https://www.kaggle.com/code/happyngoding/ssd-mobilenet-v2-python-3-10-12)
  > SSD MOBILENET V2 | Python 3.10.12 | Kaggle

### openarchive.nure.ua

- [](https://openarchive.nure.ua/bitstreams/b815ddba-9722-4e6d-957e-480c5a10fd95/download)
  > 3.2.4 Встановлення Object Detection API............................................ 34 [...] У минулому створення детектора предметів здавалося трудомісткою та  складною операцією. Тепер за допомогою 

### research.google

- [MediaPipe Iris: Real-time Iris Tracking & Depth Estimation](https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/)
  > Today, we announce the release of MediaPipe Iris, a new machine learning model for accurate iris estimation. Building on our work on MediaPipe Face Mesh, this model is able to track landmarks involvin

### scholar.google.co.in

- [Google Scholar](https://scholar.google.co.in/scholar?cites=12199992810190996739&hl=en&oi=bibs)
  > ### A comprehensive review of object detection with deep learning [...] In the realm of computer vision, Deep Convolutional Neural Networks (DCNNs) have demonstrated excellent performance. Video Proce

### scholar.google.pl

- [Google Scholar](https://scholar.google.pl/scholar?as_sdt=0%2C5&cluster=9754763009665187201&hl=en)
  > ### Deep learning in object detection: A review [...] Object detection continues to play a significant part in computer vision theory, study and practical application. Conventional object detection al

### slideshare.net

- [Deep Learning for Computer Vision: Object Detection (UPC 2016) | PDF](https://www.slideshare.net/slideshow/deep-learning-for-computer-vision-object-detection-upc-2016/64665413)
  > # Deep Learning for Computer Vision: Object Detection (UPC 2016) [...] The document discusses advancements in object detection techniques, focusing on deep convolutional networks and proposals for reg

### stackoverflow.com

- [Tensorflow custom Object Detector: model_main_tf2 doesn't start training](https://stackoverflow.com/questions/66813864/tensorflow-custom-object-detector-model-main-tf2-doesnt-start-training)
  > **Problem summary [...] tensorflow custom object [...] when i follow the [...] ``` model_main_tf2.py  --model_dir=<path1> --pipeline_config_path=<path2> --alsologtostderr ``` [...] The output is shown
- [Tensorflow Output for custom object detection](https://stackoverflow.com/questions/67871629/tensorflow-output-for-custom-object-detection)
  > I'm a newbee in using tensorflow. Why am I getting so many metrics while training custom tensorflow 2.x object detection? [...] ``` Use fn_output_signature instead INFO:tensorflow:Step 100 per-step ti
- [How to print Accuracy and other metrics in Tensorflow 2.x?](https://stackoverflow.com/questions/63966974/how-to-print-accuracy-and-other-metrics-in-tensorflow-2-x)
  > To train the model I use the model\_main\_tf2.py, [...] ``` !python /content/gdrive/My\ Drive/models/research/object_detection/model_main_tf2.py \     --pipeline_config_path={pipeline_file} \     --mo
- [TF2 Object Detection API: model_main_tf2.py - validation loss?](https://stackoverflow.com/questions/64510791/tf2-object-detection-api-model-main-tf2-py-validation-loss/64535040)
  > The problem is, the training loss is shown, and it is decreasing on average, but the validation loss is not. [...] In the`pipeline.config`file, I did input the evaluation TFRecord file (which I assume
- [TF2 Object Detection API: model_main_tf2.py - validation loss?](https://stackoverflow.com/questions/64510791/tf2-object-detection-api-model-main-tf2-py-validation-loss)
  > # TF2 Object Detection API: model_main_tf2.py - validation loss?  - Tags: python, tensorflow, tensorflow2.0, object-detection, object-detection-api - Score: 4 - Views: 5,854 - Answers: 1 - Asked by: Y

### tensorflow-object-detection-api-tutorial.readthedocs.io

- [TensorFlow 2 Object Detection API tutorial ¶](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/2.2.0)
  > Important This tutorial is intended for TensorFlow 2.2, which (at the time of writing this tutorial) is the latest stable version of TensorFlow 2.x. A version for TensorFlow 1.14 can be foundhere. Thi
- [Training Custom Object Detector](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/2.2.0/training.html)
  > Before we begin training our model, let’s go and copy the `TensorFlow/models/research/object_detection/model_main_tf2.py` script and paste it straight into our `training_demo` folder. We will need thi

### tensorflow.org

- [Object Detection | TensorFlow Hub](https://www.tensorflow.org/hub/tutorials/object_detection)
  > Pick an object detection module and apply on the downloaded image. Modules: [...] * **FasterRCNN+InceptionResNet V2**: high accuracy, * **ssd+mobilenet V2**: small and fast. [...] ``` `module\_handle=
- [Object detection with Model Garden | TensorFlow Core](https://www.tensorflow.org/tfmodels/vision/object_detection)
  > ``` `importorbitimporttensorflow\_modelsastfmfromofficial.coreimportexp\_factoryfromofficial.coreimportconfig\_definitionsascfgfromofficial.vision.servingimportexport\_saved\_model\_libfromofficial.vi
- [Training & evaluation with the built-in methods | TensorFlow Core](https://www.tensorflow.org/guide/keras/training_with_built_in_methods)
  > In such cases, you can call`self.add\_loss(loss\_value)`from inside the call method of a custom layer. Losses added in this way get added to the "main" loss during training (the one passed to`compile(

### theseus.fi

- [Examensarbetets titel](https://www.theseus.fi/bitstream/handle/10024/865428/Patabendige_Thumula.pdf?sequence=2&isAllowed=y)
  > Figure 37 YOLO: Classification loss. X-axis: Number of epochs; Y-axis: Loss...................62 [...] Figure 38 YOLO [...] Loss......................63 [...] Figure 39 SSD: classification loss. X-axi

### uniquetrij.medium.com

- [Tensorflow Object Detection: Working Around a NAN Loss | by Trijeet Modak | Medium](https://uniquetrij.medium.com/tensorflow-object-detection-working-around-a-nan-loss-ecf8f66dc09e)
  > Recently I was working on a project that required training of an object detection model in Tensorflow 2.x (version 2.4 to be specific). I was fine-tuning the EfficientDet-D1 checkpoint available at th

### upcommons.upc.edu

- [](https://upcommons.upc.edu/bitstreams/647172b3-a80e-4bad-a5a4-ead6542c468f/download)
  > ='http:// [...] .org/models/obj [...] /tf2/2 [...] 200711/ssd_mobilenet_v2_fpnlite_640x640_coco1 7_tpu-8. [...] Python wget que permite [...] archivos del navegador: [...] 2. Descaremos e instalaremos

### vtechworks.lib.vt.edu

- [](https://vtechworks.lib.vt.edu/bitstream/handle/10919/83200/SatelliteImageFinderReport.pdf%3Fsequence%3D19%26isAllowed%3Dy)
  > The network’s difficult to accurately identify the location of parking lots [...] confirmed by the information presented in Figure 9. This figure represents the localization loss  over [...] of traini

### youtube.com

- [Simple Object Detection in Python using TensorFlow - YouTube](https://www.youtube.com/watch?v=d3BjpLCFFAc)
  > now this is the final step that is model dot fit okay and this that step where the training will begin so now what we'll do will train the this object detection part you can see the training is begin 
