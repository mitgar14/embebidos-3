## Track B: Descubrimiento

**Tema:** Roboflow Python SDK 1.3.8 dataset location parameter bug fixed 2026

**Resultados:** 24


### blog.roboflow.com

- [Roboflow Changelog: February 2023](https://blog.roboflow.com/changelog-february-2023)
  > - Launched model weights uploads for YOLOv5 Object Detection - Launched model weights uploads for YOLOv7 Instance Segmentation - Bug Fixes and Improvements to the Deploy Tab - Improved status messages
- [Recap: Roboflow's 12 Days of #Shipmas](https://blog.roboflow.com/shipmas-2023/)
  > We coined the 1 [...] Days of #Shipmas and shipped 12 new features in 12 days. New updates were released to improve model-assisted labeling, model training, Roboflow Universe, our annotation tools, th

### deepwiki.com

- [Dataset Upload | roboflow/roboflow-python | DeepWiki](https://deepwiki.com/roboflow/roboflow-python/4.1-dataset-upload)
  > . It covers both single image [...] and bulk dataset uploads with [...] . For information on [...] The `upload` method accepts several parameters for customization: [...] |Parameter|Type|Description| 

### discuss.roboflow.com

- [Can't Download my Dataset - Community Help - Roboflow](https://discuss.roboflow.com/t/cant-download-my-dataset/11880)
  > So , My Major problem is that I gathered a very very big dataset for my graduation project and research use but when the Size got so big I can’t download the current Version and I do not know where is
- [Dataset download stuck issue - Community Help - Roboflow](https://discuss.roboflow.com/t/dataset-download-stuck-issue/11989)
  > Dataset download stuck issue - 🤝 Community Help - Roboflow  Resources  Categories  Tags  # Dataset download stuck issue  You have selected 0 posts.  35 views  2 / 6  Mar 25  I am having an issue expor
- [Dataset new version not working! - Community Help - Roboflow](https://discuss.roboflow.com/t/dataset-new-version-not-working/8282)
  > Dataset new version not working! - 🤝 Community Help - Roboflow  - More - README - Topics  Resources  - YouTube - Research Credits - Weekly Webinar - Templates - Docs - Quickstart  Categories  - All ca
- [Dataset Download Stuck - Community Help - Roboflow](https://discuss.roboflow.com/t/dataset-download-stuck/11625)
  > Dataset Download Stuck - 🤝 Community Help - Roboflow

### docs.roboflow.com

- [Upload a Dataset | Developer Reference | Roboflow Docs](https://docs.roboflow.com/developer/python-sdk/upload-a-dataset)
  > The `upload_dataset` method lets you upload a dataset to a workspace into a new project or to one that already exists within Roboflow. [...] To upload a dataset using the Python SDK, use the following
- [Create a Dataset Version | Developer Reference | Roboflow Docs](https://docs.roboflow.com/developer/python-sdk/create-a-dataset-version)
  > # Create a Dataset Version [...] To create a Dataset version, use the `project.generate_version()` method. [...] ```python import roboflow  rf = roboflow.Roboflow(api_key=YOUR_API_KEY_HERE)  project =

### github.com

- [roboflow/roboflow-python](https://github.com/roboflow/roboflow-python)
  > - Stars: 554 - Forks: 119 - Watchers: 554 - Open issues: 61 - Primary language: Python - Languages: Python (97.6%), Shell (2.3%), Makefile - License: Apache License 2.0 (Apache-2.0) - Topics: computer
- [docs/index.md at 74885a27 · roboflow/roboflow-python](https://github.com/roboflow/roboflow-python/blob/74885a27/docs/index.md)
  > # upload a dataset workspace.upload_dataset(     dataset_path="./dataset/",     num_workers=10,     dataset_format="yolov8", # supports yolov8, yolov5, and Pascal VOC     project_license="MIT",     pr
- [Incorrect Data Path in YOLOv8 Dataset Configuration · Issue #240 · roboflow/roboflow-python](https://github.com/roboflow/roboflow-python/issues/240)
  > ## Incorrect Data Path in YOLOv8 Dataset Configuration [...] While using the Roboflow Python client to download a YOLOv8 dataset for a tennis ball tracking project, I encountered an issue with the gen
- [Roboflow - ultralytics error with dataset path · Issue #306 · roboflow/notebooks](https://github.com/roboflow/notebooks/issues/306)
  > ## Roboflow - ultralytics error with dataset path [...] Yolov8 custom dataset, but using the python api. Dataset path is apparently incorrect. I know I've had this issue in the past [...] ```python rf
- [Dataset not in Roboflow · Issue #183 · roboflow/notebooks](https://github.com/roboflow/notebooks/issues/183)
  > > Hello @Shouq95! Thank you for your comment. You can upload your dataset to Roboflow to use the Roboflow data download features in our notebooks. You can also replace anywhere that `dataset.location`
- [Issue with relative paths in data.yaml file when trying to train yolo custom model · Issue #333 · roboflow/roboflow-python](https://github.com/roboflow/roboflow-python/issues/333)
  > ## Issue with relative paths in data.yaml file when trying to train yolo custom model [...] I am having an issue where if I make my data.yaml file use relative paths, I get the error: [...] ```   Runt
- [.download() re-downloads the same version even if it already exists on disk · Issue #108 · roboflow/roboflow-python](https://github.com/roboflow/roboflow-python/issues/108)
  > ## .download() re-downloads the same version even if it already exists on disk [...] The default behavior should be to use the local copy since versions are frozen in time. Perhaps we could provide a 

### ijcaonline.org

- [](https://ijcaonline.org/archives/volume187/number48/parvaiz-2025-ijca-925794.pdf)
  > Roboflow [...] The entire pipeline was trained and evaluated on the Roboflow Indian Number Plate Dataset, a big dataset of real-world traffic images. The code was run on Google Colab, making the cod

### pypi.org

- [roboflow v1.3.1](https://pypi.org/project/roboflow/1.3.1/)
  > # upload a dataset workspace.upload_dataset(     dataset_path="./dataset/",     num_workers=10,     dataset_format="yolov8", # supports yolov8, yolov5, and Pascal VOC     project_license="MIT",     pr

### roboflow.github.io

- [Roboflow Python](https://roboflow.github.io/roboflow-python/)
  > # upload a dataset workspace.upload_dataset(     dataset_path="./dataset/",     num_workers=10,     dataset_format="yolov8", # supports yolov8, yolov5, and Pascal VOC     project_license="MIT",     pr

### universe.roboflow.com

- [Detecting Filipino Foods Object Detection Dataset by Thesis](https://universe.roboflow.com/thesis-mthq0/detecting-filipino-foods)
  > ## About Detecting Filipino Foods Dataset [...] A description for this project has not [...] ## Use Free Chicken (Fried Chicken), Fish (Pan Fried Tilapia) and White Rice (Boiled Rice) Detection API [.
- [Paper Object Detection Model by Trash](https://universe.roboflow.com/trash-aaqf1/paper-b2zvp)
  > ## Use Free ', \ and Recyclable glass Detection API [...] ``` pip install inference-sdk ``` [...] ``` # 1. Import the library from inference_sdk import InferenceHTTPClient [...] # 2. Connect to your w
- [FruitShop Object Detection Model by paperSaur+Fruity](https://universe.roboflow.com/papersaur-fruity/fruitshop)
  > ## About FruitShop Model [...] A description for this project has not been published yet. [...] ## Use Free Fruit Detection API [...] ``` pip install inference-sdk ``` [...] ``` # 1. Import the librar
- [Pikachu Object Detection Dataset by Doo Nung](https://universe.roboflow.com/doo-nung-fiioq/pikachu-hc3dw)
  > # Pikachu Computer Vision Dataset [...] ## About Pikachu Dataset [...] A description for this project has not been published yet [...] ## Use Free Object Detection API [...] ``` pip install inference-

### upcommons.upc.edu

- [](https://upcommons.upc.edu/bitstreams/3d92cf6c-2b3c-4f85-9be8-916a4ed8923b/download)
  > ería Robof [...] Roboflow es una plataforma integral para la gestión de datos de visión por  computadora que facilita [...] etiquetado, el preprocesamiento, el aumento de datos  y el entrenamiento de 
