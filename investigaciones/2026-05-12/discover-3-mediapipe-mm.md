## Track B: Descubrimiento

**Tema:** MediaPipe Model Maker MOBILENET_V2_I320 quantization aware training TFLite object detection

**Resultados:** 27


### ai.google.dev

- [Object Detection with TensorFlow Lite Model Maker  |  Google AI Edge  |  Google AI for Developers](https://ai.google.dev/edge/litert/libraries/modify/object_detection)
  > In this colab notebook, you'll learn how to use the TensorFlow Lite Model Maker library to train a custom object detection model capable of detecting salads within images on a mobile device. [...] The
- [MediaPipe Model Maker | Google AI Edge](https://ai.google.dev/edge/mediapipe/solutions/model_maker)
  > MediaPipe Model Maker | Google AI Edge | Google AI for Developers  # MediaPipe Model Maker  MediaPipe Model Maker is a tool for customizing existing machine learning (ML) models to work with your data
- [Object detection model customization guide | Google AI Edge](https://ai.google.dev/edge/mediapipe/solutions/customization/object_detector)
  > ## Model quantization [...] This section of the guide explains how to apply quantization to your model. Model Maker supports two forms of quantization for object detector: [...] 1. Quantization Aware 
- [mediapipe_model_maker.object_detector.ObjectDetector](https://ai.google.dev/edge/api/mediapipe/python/mediapipe_model_maker/object_detector/ObjectDetector)
  > ### `export_model [...] ``` export_model(     model_name: str = 'model.tflite',     quantization_config: Optional[https://ai.google.dev/edge/api/mediapipe/python/mediapipe_model_maker/quantization/Qua

### arxiv.org

- [](https://arxiv.org/pdf/1712.05877)
  > point inference on commonly available integer-only hardware. We also co-design a training procedure to preserve [...] • We provide a quantization scheme (section 2.1) that [...] • We provide a quanti

### beei.org

- [](https://www.beei.org/index.php/EEI/article/download/8131/4043)
  > devices, which are required to deliver real-time processing capabilities. Traditional object detection models are excessively resource-hungry for these environments, making optimization techniques a

### colab.research.google.com

- [Train a custom object detection model with MediaPipe Model Maker](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/tutorials/object_detection/Object_Detection_for_3_dogs.ipynb)
  > Google Colab  Sign in  Google Colab Sign in close Loading... close

### developers.google.com

- [Build and deploy a custom object detection model with TensorFlow Lite (Android)  |  Google for Developers](https://developers.google.com/codelabs/tflite-object-detection-android)
  > In this codelab, you'll learn how to train a custom object detection model using a set of training images with TFLite Model Maker, then deploy your model to an Android app using TFLite Task Library. Y

### discuss.ai.google.dev

- [Decoding of tflite custom object detector output from model ...](https://discuss.ai.google.dev/t/decoding-of-tflite-custom-object-detector-output-from-model-trained-with-mediapipe-mobilenetv2/32206)
  > Decoding of tflite custom object detector output from model trained with mediapipe (MobileNetV2) - General Discussion - Google AI Developers Forum  Decoding of tflite custom object detector output fro
- [MediaPipe: massive accuracy loss with quantization-aware training](https://discuss.ai.google.dev/t/mediapipe-massive-accuracy-loss-with-quantization-aware-training/23177)
  > MediaPipe: massive accuracy loss with quantization-aware training - General Discussion - Google AI Developers Forum

### ejurnal.seminar-id.com

- [](http://ejurnal.seminar-id.com/index.php/tin/article/download/8453/4160/)
  > berbahaya dan beracun), kemudian melakukan pelatihan algoritma SSD MobileNetV2 menggunakan framework MediaPipe library [...] mediapipe-model-maker, dan pengembangan aplikasi android dengan integrasi a

### export.arxiv.org

- [](https://export.arxiv.org/pdf/2303.05016v1.pdf)
  > (for Intel CPUs only), TensorFlow Lite (TFLite), ONNX, and [...] MobileNetV [...] TFLite [...] of DNN inference frameworks, including OpenVINO [5] (for Intel CPUs [...] ), TensorFlow Lite [...] (TFLi

### gilberttanner.com

- [TFLite Object Detection with TFLite Model Maker](https://gilberttanner.com/blog/tflite-model-maker-object-detection/)
  > TFLite Object Detection with TFLite Model Maker      The TensorFlow Lite Model Maker library is a high-level library that simplifies the process of training a TensorFlow Lite model using a custom data

### github.com

- [mediapipe/model_maker/python/vision/object_detector/object_detector_demo.py at master · google-ai-edge/mediapipe](https://github.com/google/mediapipe/blob/master/mediapipe/model_maker/python/vision/object_detector/object_detector_demo.py)
  > """Demo for making an object detector model by MediaPipe Model Maker.""" [...] .model_maker. [...] .vision import object [...] def define_flags() -> None:   """Define flags for the object detection mo
- [mediapipe/model_maker/python/vision/object_detector/model_spec.py at master · google-ai-edge/mediapipe](https://github.com/google/mediapipe/blob/master/mediapipe/model_maker/python/vision/object_detector/model_spec.py)
  > MOBILENET_V [...] 256_FILES = file_util.DownloadedFiles(     'object_detector [...] 'https://storage.googleapis.com [...] ssd_coco/ [...] 56_ckpt.tar.gz',     is_folder=True, ) [...] MOBILENET_V2_I320
- [Object detection task on custom model failed on Raspberry Pi · Issue #4744 · google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe/issues/4744)
  > Object detection on MOBILENET_V2_I320 model [...] model_path = 'model_int8.tflite' #Custom model, post-training quantization for int8 with                                  # media pipe modelmaker on G
- [Mobilenetv2 detector inference · Issue #4836 · google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe/issues/4836)
  > ## Mobilenetv2 detector inference [...] - State: [...] platform:android, type:modelmaker, task:object detection [...] kuaashish [...] I have trained the mobilenetv2 model for the detection task, I am 
- [mediapipe/model_maker/python/vision/object_detector/object_detector.py at master · google-ai-edge/mediapipe](https://github.com/google/mediapipe/blob/master/mediapipe/model_maker/python/vision/object_detector/object_detector.py)
  > object detector model.""" [...] model = model_lib.ObjectDetectorModel(         model_spec= [...] ._model_spec,         model_options=self._model_options,         num_classes=self._num_classes,     ) [
- [Could not use custom model in object detection · Issue #22 · google-ai-edge/mediapipe](https://github.com/google/mediapipe/issues/22)
  > ## Could not use custom model in object detection [...] Recently I would like to replace model `ssdlite_object_detection.tflite `by my custom model, which is trained with **ssd_mobilenetv2_coco**(floa
- [tensorflow/lite/g3doc/models/modify/model_maker/object_detection.ipynb at master · tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models/modify/model_maker/object_detection.ipynb)
  > 4] are [...] family of mobile/IoT-friendly object detection models derived [...] 8       | 49            | 30.55%               |\n",         "| [...] Lite2 | 7.2       | 69            | 33.97%       
- [mediapipe/model_maker/python/vision/object_detector/__init__.py at master · google-ai-edge/mediapipe](https://github.com/google/mediapipe/blob/master/mediapipe/model_maker/python/vision/object_detector/__init__.py)
  > """MediaPipe Model Maker Python Public API For Object Detector."""  from mediapipe.model_maker.python.vision.object_detector import dataset from mediapipe.model_maker.python.vision.object_detector imp
- [mediapipe-samples/examples/customization/object_detector.ipynb ...](https://github.com/googlesamples/mediapipe/blob/main/examples/customization/object_detector.ipynb)
  > Model quantization is a model modification technique that can reduce the model size and improve the speed of predictions with only a relatively minor decrease in accuracy.\n",         "\n",         "T
- [mediapipe/docs/solutions/models.md at master - GitHub](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md?plain=1)
  > # File: google-ai-edge/mediapipe/docs/solutions/models.md  - Repository: google-ai-edge/mediapipe | Cross-platform, customizable ML solutions for live and streaming media. | 34K stars | C++ - Branch: 
- [mediapipe/model_maker/python/vision/object_detector/model_options.py at master · google-ai-edge/mediapipe](https://github.com/google/mediapipe/blob/master/mediapipe/model_maker/python/vision/object_detector/model_options.py)
  > # File: google-ai-edge/mediapipe/mediapipe/model_maker/python/vision/object_detector/model_options.py [...] """Configurable model options for object detector models."""  import dataclasses  @dataclass

### iris.unito.it

- [](https://iris.unito.it/bitstream/2318/2067155/1/intellisys24.pdf)
  > Our study targets popular NN architectures such as ResNetV1 and V2, [...] MobileNetV1 and V2, and introduces a custom-designed model, examining their suitability to TinyML constraints. [...] AR-10 [.

### medium.com

- [Mediapipe Model Maker vs Tensorflow Model Maker | by Elven Kim](https://medium.com/@elvenkim1/mediapipe-model-maker-vs-tensorflow-model-maker-0f5220498d83)
  > Mediapipe Model Maker vs Tensorflow Model Maker | by Elven Kim | Medium  Sitemap  Open in app  Sign up  Sign in  Medium Logo  Get app  Write  Search  Sign up  Sign in  # Mediapipe Model Maker vs Tenso

### sayak.dev

- [Optimizing MobileDet for Mobile Deployments – Sayak Paul](https://sayak.dev/posts/mobiledet-optimization.html)
  > Fair question. After all, there are so many great examples and tutorials that show how to use the post-training quantization APIs in TFLite to perform the model conversion. MobileDet models in the TFO
