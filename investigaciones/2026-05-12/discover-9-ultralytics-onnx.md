## Track B: Descubrimiento

**Tema:** Ultralytics 8.4 8.5 release notes breaking changes ONNX export opset 11 YOLOv8 2026

**Resultados:** 28


### arxiv.org

- [Ultralytics YOLO Evolution: An Overview of YOLO26, YOLO11, YOLOv8, and YOLOv5 Object Detectors for Computer Vision and Pattern Recognition](https://arxiv.org/html/2510.09653v3)
  > On the systems side, YOLO26 leans into portability and quantization. By removing DFL and NMS, the exported graphs map cleanly to ONNX, TensorRT, CoreML, and TFLite with fewer custom kernels, facilitat

### community.ultralytics.com

- [New Release: Ultralytics v8.4.0 - Discussion](https://community.ultralytics.com/t/new-release-ultralytics-v8-4-0/1747)
  > New Release: Ultralytics v8.4.0 - Discussion - Ultralytics

### docs.ultralytics.com

- [ONNX Export for YOLO26 Models - Ultralytics YOLO Docs](https://docs.ultralytics.com/integrations/onnx/)
  > # ONNX Export for YOLO26 Models [...] can deliver up to [...] YOLO26 models to ONNX format streamlines deployment and ensures [...] performance across various environments [...] This guide will show y
- [Model Export with Ultralytics YOLO](https://docs.ultralytics.com/modes/export)
  > The ultimate goal of training a model [...] to deploy it for real-world applications. Export mode in Ultralytics YOLO26 offers a versatile range of options for exporting your trained model to differen
- [Ultralytics Docs: Home](https://docs.ultralytics.com/)
  > Introducing Ultralytics YOLO26, the latest version of the acclaimed real-time object detection and image segmentation model. YOLO26 is built on deep learning and computer vision advancements, featurin
- [Explore Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)
  > YOLOv8 was released by Ultralytics on January 10, 2023, offering cutting-edge performance in terms of accuracy and speed. Building upon the advancements of previous YOLO versions, YOLOv8 introduced ne

### github.com

- [v8.4.20 - `ultralytics 8.4.20` Remove redundant hardcoded `tuner_callbacks` (#23772)](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.20)
  > # Release: ultralytics/ultralytics v8.4.20 [...] - Repository: ultralytics/ultralytics | Ultralytics [...] 🚀 | 5 [...] stars | Python - Name: v8.4.20 - `ultralytics 8.4.20` Remove redundant hardcoded 
- [Support Exporting YOLOv8 to ONNX with IR Version 8 · Issue #19498 · ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/issues/19498)
  > I'm trying to export a YOLOv8 model to ONNX with an intermediate representation (IR) version of 8. By default, the export produces a model with IR version 9, which is incompatible with my deployment e
- [v8.4.8 - `ultralytics 8.4.8` Support `max_det` and `agnostic_nms` for end2end (#23396)](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.8)
  > # Release: ultralytics/ultralytics v8.4.8 [...] - Repository: ultralytics/ultralytics | Ultralytics YOLO 🚀 | 56K stars | Python - Name: v8.4.8 - `ultralytics 8.4.8` Support `max_det` and `agnostic_nms
- [v8.4.1 - `ultralytics 8.4.1` Re-enable NCNN exports for ARM64 (#23211)](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.1)
  > # Release: ultralytics/ultralytics v8.4.1 [...] - Repository: ultralytics/ultralytics | Ultralytics YOLO 🚀 | 56K stars | Python - Name: v8.4.1 - `ultralytics 8.4.1` Re-enable NCNN exports for ARM64 (#
- [v8.4.13 - `ultralytics 8.4.13` Retry smaller batch on training CUDA OOM (#23590)](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.13)
  > # Release: ultralytics/ultralytics v8.4.13 [...] - Repository: ultralytics/ultralytics | Ultralytics YOLO 🚀 | 56K stars | Python - Name: v8.4.13 - `ultralytics 8.4.13` Retry smaller batch on training 
- [v8.4.38 - `ultralytics 8.4.38` Unify args naming for standalone export functions (#24120)](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.38)
  > # Release: ultralytics/ultralytics v8.4.38 [...] - Repository: ultralytics/ultralytics | Ultralytics YOLO 🚀 | 56K stars | Python - Name: v8.4.38 - `ultralytics 8.4.38` Unify args naming for standalone
- [v8.4.21 - `ultralytics 8.4.21` Fix Rockchip RKNN export path (#23806)](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.21)
  > # Release: ultralytics/ultralytics v8.4.21 [...] - Repository: ultralytics/ultralytics | Ultralytics YOLO 🚀 | 57K stars | Python - Name: v8.4.21 - `ultralytics 8.4.21` Fix Rockchip RKNN export path (#
- [Comparing v8.4.0...v8.4.1 · ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/compare/v8.4.0...v8.4.1)
  > # Compare: ultralytics/ultralytics v8.4.0...v8.4.1 [...] - 8c4cd25: Update YOLO26-Pose/OBB metrics (#23178) (Jing Qiu, 2026-01-14) - 2f5c440: Update docs (#23181) (Jing Qiu, 2026-01-14) [...] - d8dbea
- [Comparing v8.4.19...v8.4.21 · ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/compare/v8.4.19...v8.4.21)
  > # Compare: ultralytics/ultralytics v8.4.19...v8.4.21 [...] - c54c138: Fix RKNN exports to support YOLO26 models (#23802) (Lakshantha Dissanayake, 2026-03-05) [...] - d0f665f: `ultralytics 8.4.20` Remo
- [`ultralytics 8.2.15` get_latest_opset() compat for `torch<1.13.0` · Pull Request #12652 · ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/pull/12652)
  > ## `ultralytics 8.2.15` get_latest_opset() compat for `torch<1.13.0` [...] Torch 1.13 introduced significant changes in onnx/init.py. In earlier Torch versions, it is not possible to directly retrieve
- [Releases · ultralytics/ultralytics - GitHub](https://github.com/ultralytics/ultralytics/releases)
  > v8.4.48 - `ultralytics 8.4.48` Fix Platform training edge cases (#24431) Latest [...] Ultralytics`v8.4.48` is a stability-focused release that mainly improves training reliability on Ultralytics Platf
- [Exporting the operator silu to ONNX opset version 11 is not supported](https://github.com/ultralytics/ultralytics/issues/644)
  > ## Exporting the operator silu to ONNX opset version 11 is not supported [...] PyTorch: starting from runs\detect\train\weights\best.pt with output shape (1, 84, 8400) (6.2 MB)  ONNX: starting export 
- [Yolov11 export ONNX format, opset-related issues #16839 - GitHub](https://github.com/ultralytics/ultralytics/issues/16839)
  > # Issue: ultralytics/ultralytics #16839  - Repository: ultralytics/ultralytics | Ultralytics YOLO 🚀 | 55K stars | Python  ## Yolov11 export ONNX format, opset-related issues  - Author: @zhoujun0715 - 
- [Comparing v8.4.24...v8.4.25 · ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/compare/v8.4.24...v8.4.25)
  > , what changes for your [...] formats support it [...] +If you're upgrading to [...] 26 from an earlier model like YOLOv8 or YOLO11, one of the biggest changes you'll notice is the removal of Non-Maxi
- [Export - Ultralytics YOLOv8 Docs #2386 - GitHub](https://github.com/orgs/ultralytics/discussions/2386)
  > # Organization: Ultralytics (@ultralytics)  Democratizing Vision AI  - Website: https://ultralytics.com - Email: hello@ultralytics.com - Twitter: @ultralytics - Location: United States of America - Fo
- [yolov8 export as onnx issue (8.0.43) #1097 - GitHub](https://github.com/ultralytics/ultralytics/issues/1097)
  > ## yolov8 export as onnx issue (8.0.43) [...] It appears that something might've changed with the latest yolov8.0.43 as by running the script: [...] ``` yolo export \ model=yolov8m.pt \ imgsz=640 \ fo

### ultralytics.com

- [YOLOv8: 1 Year of Innovation - Ultralytics](https://www.ultralytics.com/blog/ultralytics-yolov8-turns-one-a-year-of-breakthroughs-and-innovations)
  > ytics YOLOv8 [...] Breakthroughs and Innovations [...] We’re bringing YOLOv8 closer to you! Our documentation is now available in 11 languages, with 200+ docs pages, and is continuously expanding to s

### zenodo.org

- [Ultralytics YOLO](https://zenodo.org/records/17054310)
  > Published September 4, 2025 | Version v8.3.193 [...] Ultralytics 8.3.193 boosts long-video and large-batch inference performance by auto-enabling TorchVision NMS, streamlines checkpoint loading with a
- [Ultralytics YOLO](https://zenodo.org/records/15682027)
  > Published June 17, 2025 | Version v8.3.156 [...] This release enhances model export reliability—especially for TensorRT INT8 quantization—improves data handling for training and calibration, and bring
- [Ultralytics YOLO](https://zenodo.org/records/14900382)
  > Published February 20, 2025 | Version v8.3.78 [...] This release,`v8.3.78`, brings an exciting new model to the family: YOLO12 🚀, featuring an attention-centric design for superior accuracy and effici
- [Ultralytics YOLO](https://zenodo.org/records/15862938)
  > Published July 11, 2025 | Version v8.3.164 [...] This release delivers a critical fix to YOLO detection validation metrics, improves dataset flexibility, enhances export reliability, and polishes docu
- [Ultralytics YOLO](https://zenodo.org/records/14862620)
  > Published February 13, 2025 | Version v8.3.75 [...] The v8.3.75 release includes robust updates for improved model export compatibility, user experience, and error handling across platforms, alongside
