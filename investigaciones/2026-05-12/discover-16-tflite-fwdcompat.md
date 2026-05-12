## Track B: Descubrimiento

**Tema:** TFLite forward compatibility TF 2.5 op_version validation flatbuffer schema Jetson Nano 2024 2025 2026

**Resultados:** 28


### android.googlesource.com

- [tensorflow/lite/schema/upgrade_schema.py - platform/external/tensorflow - Git at Google](https://android.googlesource.com/platform/external/tensorflow/%2B/83d4e9bebe90f6b939719acc6a2a0ab1e43437d1/tensorflow/lite/schema/upgrade_schema.py)
  > | """Upgrade script to move from pre-release schema to new schema. | [...] tensorflow/lite/schema/upgrade_schema [...] in.json out.json | [...] bazel run tensorflow/lite/schema/upgrade_schema -- in.bi

### arxiv.org

- [](https://www.arxiv.org/pdf/2010.08678v2)
  > model file (.tflite); [...] conversion, the model file can be [...] deployed to a client device [...] .g., a mobile or embedded system [...] and run locally using [...] TensorFlow Lite interpreter. [

### blog.tensorflow.org

- [What's new in TensorFlow 2.19 — The TensorFlow Blog](https://blog.tensorflow.org/2025/03/whats-new-in-tensorflow-2-19.html)
  > GuwtY/s1600/Tensorflow- [...] 81% [...] 9.png [...] March 13, 2025 — *Posted by the TensorFlow team*TensorFlow 2.19 has been released! Highlights of this release include changes to the C++ API in Lite

### docs.nvidia.com

- [TensorFlow For Jetson Platform - NVIDIA Docs](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html)
  > This document describes [...] key features, software enhancements and improvements, and known issues regarding NVIDIA TensorFlow on the Jetson platform. See Installing TensorFlow for Jetson Platform f

### forums.developer.nvidia.com

- [Tensorflow compatibilty issue - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/tensorflow-compatibilty-issue/303968)
  > Tensorflow compatibilty issue - Jetson Nano - NVIDIA Developer Forums
- [Installing tflite_interpreter on Jetson Nano - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/installing-tflite-interpreter-on-jetson-nano/264497)
  > Installing tflite_interpreter on Jetson Nano - Jetson Nano - NVIDIA Developer Forums  Installing tflite\_interpreter on Jetson Nano - Jetson Nano - NVIDIA Developer Forums
- [How to run tflite model on Jetson nano - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/how-to-run-tflite-model-on-jetson-nano/116240)
  > How to run tflite model on Jetson nano - Jetson Nano - NVIDIA Developer Forums
- [Jetson Nano EOL Compatibility Solutions - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/jetson-nano-eol-compatibility-solutions/269290)
  > Jetson Nano EOL Compatibility Solutions - Jetson Systems / Jetson Nano - NVIDIA Developer Forums  # Jetson Nano EOL Compatibility Solutions  Robotics & Edge Computing Jetson Systems Jetson Nano  ubunt
- [GPU support for tflite - Jetson Nano - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/gpu-support-for-tflite/156477)
  > I am using Jetpack 4.4DP(yet to flash 4.4) and Tensorflow 2.2.0 version(nvidia provided one). [...] I am able to run Tensorflow model in GPU. [...] But if i try to run tflite model it is not using the
- [How can i run TFLite model using GPU support on Jetson Nano?](https://forums.developer.nvidia.com/t/how-can-i-run-tflite-model-using-gpu-support-on-jetson-nano/202629)
  > How can i run TFLite model using GPU support on Jetson Nano? - Jetson Nano - NVIDIA Developer Forums
- [Running tflite models on Orin Nano - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/running-tflite-models-on-orin-nano/299669)
  > Running tflite models on Orin Nano - Jetson Orin Nano - NVIDIA Developer Forums
- [Official TensorFlow for Jetson Nano! - Page 7](https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-nano/71770?page=7)
  > Official TensorFlow for Jetson Nano! - Page 7 - Jetson Nano - NVIDIA Developer Forums  Official TensorFlow for Jetson Nano! - Page 7 - Jetson Nano - NVIDIA Developer Forums

### fossies.org

- [Release 2.19.0](https://fossies.org/linux/tensorflow/RELEASE.md)
  > * `LiteRT`, [...] a.`tf.lite`: [...] * The public constants`tflite::Interpreter:kTensorsReservedCapacity`and`tflite::Interpreter:kTensorsCapacityHeadroom`are now const [...] references, rather than`co

### github.com

- [Update flatbuffers to 24.3.25 · c17d64d · tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/commit/c17d64df85a83c1bd0fd7dcc0b1230812b0d3d48)
  > ## Update flatbuffers to 24.3.25 [...] | File | Status | Add | Del | | --- | --- | --- | --- | | tensorflow [...] modified | +1 | -1 | | tensorflow/ [...] gpu/cl/compiled_program_cache_generated.h | m
- [Will Tflite support flatbuffers v23.5.26 in the future ?(in this version, flatbuffers already support 64 bit field) · Issue #62413 · tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/issues/62413)
  > ## Will Tflite support flatbuffers v23.5.26 in the future ?(in this version, flatbuffers already support 64 bit field) [...] I already see that Tesnorflow Lite can support models > 2GB in schema Versi
- [Breaking change of TFLite model definition in TensorFlow 2.4.0: OperatorCode.BuiltinCode · Issue #46663 · tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/issues/46663)
  > ## Breaking change of TFLite model definition in TensorFlow 2.4.0: OperatorCode.BuiltinCode [...] **This change breaks software stacks that depend on the built TFLite model parser, e.g. tvm, tflite2on
- [flatbuffers version missmatch · Issue #3979 · google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT/issues/3979)
  > ## flatbuffers version missmatch [...] tflite/CMakeLists.txt  have this code: [...] ```CMake if(NOT TENSORFLOW_SOURCE_DIR)   message(STATUS "Downloading TensorFlow repository...")   FetchContent_Decla
- [Android Tensorflow tflite error version 2.17. Didn't find op for builtin opcode 'FULLY_CONNECTED' version '12'. An older version of this builtin might be supported · Issue #80736 · tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/issues/80736)
  > ## Android Tensorflow tflite error version 2.17. Didn't find op for builtin opcode 'FULLY_CONNECTED' version '12'. An older version of this builtin might be supported [...] - Author: @therohanchoudhar
- [Convert TFlite buffer created using TF1 to TF2 TFlite buffer · Issue #66281 · tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/issues/66281)
  > ## Convert TFlite buffer created using TF1 to TF2 TFlite buffer [...] , comp:lite, [...] -05 [...] T01 [...] Context: I have a bunch of TFlite files that were created using schema.fbs before `version`
- [Add another test for ABI backwards compatibility. · 49da99b · tensorflow/tf-build-actions](https://github.com/tensorflow/tf-build-actions/commit/49da99b0a844125dbf9004821438c89fc0c101f6)
  > ## Add another test for ABI backwards compatibility. [...] This one is designed to catch ABI breakage from possible future changes to the FlatBuffer compiler's proto to FlatBuffer schema conversion (e
- [docs/increasing_dependencies_versions.md at main · NXP/eiq-onnx2tflite](https://github.com/NXP/eiq-onnx2tflite/blob/main/docs/increasing_dependencies_versions.md)
  > # Upgrade onnx2 [...] ONNX and [...] As part of the maintenance activities the converter must keep aligned with ONNX and TensorFlow Lite evolution. Therefore, both the ONNX and TensorFlow Lite version
- [Are you using old TFLite binary with newer model?Registration failed. · Issue #47857 · tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/issues/47857)
  > ## Are you using old TFLite binary with newer model?Registration failed. [...] - Author: @aliceruget - State: closed (completed) - Labels: stat:awaiting response, type:build/install, stale, comp:lite,
- [build: update flatbuffers dependency to v23.5.26 · Pull Request #2274 · tensorflow/tflite-micro](https://github.com/tensorflow/tflite-micro/pull/2274)
  > ## build: update flatbuffers dependency to v23.5.26 [...] Update the third_party flatbuffers library to v23.5.26, the current version in upstream TF. Synchronize the override BUILD and build_defs.bzl 
- [tensorflow_lite_support/metadata/metadata_schema.fbs at master · tensorflow/tflite-support](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs)
  > // The Metadata schema is versioned by the Semantic versioning number, such as // MAJOR.MINOR.PATCH. It tracks the schema changes according to the rules below: //  * Bump up the MAJOR number when maki
- [Need to fetch version from TFLite Model · Issue #50138 · tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/issues/50138)
  > I have downloaded / created tfLite Model. [...] How do i know the version of the tfLite Model from just the flatbuffer model(*.tflite ) file.. [...] Can i fetch the version information from this model

### tensorflow.org

- [TensorFlow version compatibility](https://www.tensorflow.org/guide/versions)
  > Because of this, we use a different version number for TensorFlow Lite (`TFLITE\_VERSION\_STRING`in`tensorflow/lite/version.h`, and`TfLiteVersion()`in`tensorflow/lite/c/c\_api.h`) than for TensorFlow 

### zenodo.org

- [Tools to convert ONNX files (NCHW) to TensorFlow format (NHWC)](https://zenodo.org/records/18645133)
  > Published February 15, 2026 | Version 2.0.11 [...] This PR extends`flatbuffer_direct` for quantized ONNX graphs and documents the updated support matrix. [...] Added`flatbuffer_direct`-only preprocess
- [ultralytics/yolov5: v6.1 - TensorRT, TensorFlow Edge TPU and OpenVINO Export and Inference](https://zenodo.org/records/6222936)
  > This release incorporates new features and bug fixes (**271 PRs**from**48 contributors**) since our lastreleasein October 2021. It addsTensorRT,Edge TPUandOpenVINOsupport, and provides retrained model
