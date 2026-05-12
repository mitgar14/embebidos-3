## Track B: Descubrimiento

**Tema:** roboflow-python SDK dataset download location parameter bug data.yaml subfolder versioned

**Resultados:** 17


### discuss.roboflow.com

- [Data.yaml not find with all datasets when importing into Google Colab](https://discuss.roboflow.com/t/data-yaml-not-find-with-all-datasets-when-importing-into-google-colab/1514)
  > Data.yaml not find with all datasets when importing into Google Colab - 🤝 Community Help - Roboflow  Data.yaml not find with all datasets when importing into Google Colab - 🤝 Community Help - Roboflow
- [Urgent -- Full Dataset Not Downloading - Community Help - Roboflow](https://discuss.roboflow.com/t/urgent-full-dataset-not-downloading/4984)
  > Urgent -- Full Dataset Not Downloading - 🤝 Community Help - Roboflow
- [Can't Download my Dataset - Community Help - Roboflow](https://discuss.roboflow.com/t/cant-download-my-dataset/11880)
  > So , My Major problem is that I gathered a very very big dataset for my graduation project and research use but when the Size got so big I can’t download the current Version and I do not know where is

### docs.roboflow.com

- [Export Data | Developer Reference | Roboflow Docs](https://docs.roboflow.com/developer/rest-api/export-data)
  > To create a ZIP file of a dataset for export from the Python SDK, begin by retrieving a specific version from a project: [...] ``` version = project.version(version_number) [...] Then download the dat
- [Download a Dataset | Developer Reference - Roboflow Docs](https://docs.roboflow.com/developer/command-line-interface/download-a-dataset)
  > # Download a Dataset  If your project has a saved version, you can download the images and annotations for that version using the command line.  ## Command  ```bash roboflow version download <workspac

### github.com

- [data.yaml file has different references for image paths · Issue #125 · roboflow/roboflow-python](https://github.com/roboflow/roboflow-python/issues/125)
  > ## data.yaml file has different references for image paths [...] When a dataset is downloaded via the API, the data.yaml file has inconsistent image paths. For example they look like: [...] ``` test: 
- [Fix for v8>=8.0.29 breaking changed to dataset loader  · Pull Request #113 · roboflow/roboflow-python](https://github.com/roboflow/roboflow-python/pull/113)
  > ## Fix for v8>=8.0.29 breaking changed to dataset loader [...] PR solves problems related to https://github.com/ultralytics/ultralytics/issues/873. Along with the `8.0.30` release, the YOLOv8 team cha
- [Roboflow - ultralytics error with dataset path · Issue #306 · roboflow/notebooks](https://github.com/roboflow/notebooks/issues/306)
  > ## Roboflow - ultralytics error with dataset path [...] Yolov8 custom dataset, but using the python api. Dataset path is apparently incorrect. I know I've had this issue in the past [...] I download t
- [.download() re-downloads the same version even if it already exists on disk · Issue #108 · roboflow/roboflow-python](https://github.com/roboflow/roboflow-python/issues/108)
  > ## .download() re-downloads the same version even if it already exists on disk [...] The default behavior should be to use the local copy since versions are frozen in time. Perhaps we could provide a 
- [Issue with relative paths in data.yaml file when trying to train yolo custom model · Issue #333 · roboflow/roboflow-python](https://github.com/roboflow/roboflow-python/issues/333)
  > ## Issue with relative paths in data.yaml file when trying to train yolo custom model [...] I am having an issue where if I make my data.yaml file use relative paths, I get the error: [...] ```   Runt
- [tests/test_version.py at 74885a27 · roboflow/roboflow-python](https://github.com/roboflow/roboflow-python/blob/74885a27/tests/test_version.py)
  > class TestDownload(unittest.TestCase):     def setUp(self):         super().setUp()         self.api_url = "https://api.roboflow.com/test-workspace/test-project/4/coco"         self.version = get_vers
- [roboflow/roboflow-python](https://github.com/roboflow/roboflow-python)
  > # upload a dataset workspace.upload_dataset(     dataset_path="./dataset/",     num_workers=10,     dataset_format="yolov8", # supports yolov8, yolov5, and Pascal VOC     project_license="MIT",     pr
- [Incorrect Data Path in YOLOv8 Dataset Configuration · Issue #240 · roboflow/roboflow-python](https://github.com/roboflow/roboflow-python/issues/240)
  > While using the Roboflow Python client to download a YOLOv8 dataset for a tennis ball tracking project, I encountered an issue with the generated `data.yaml` file. The file's paths were incorrectly se
- [Missing dataset name in test split path in data.yaml · Issue #82 · roboflow/notebooks](https://github.com/roboflow/notebooks/issues/82)
  > ## Missing dataset name in test split path in data.yaml [...] When using the Roboflow data import, the dataset name is not written to the path of the dataset test split in the `data.yaml` file. [...] 
- [Dataset missing path error · Issue #69 · roboflow/notebooks](https://github.com/roboflow/notebooks/issues/69)
  > FileNotFoundError:  Dataset '/home/elin/model_training/yolov8/ClassAction-1/data.yaml' not found ⚠️, missing paths ['/home/elin/model_training/yolov8/datasets/ClassAction-1/valid/images'] [...] rf = R

### roboflow.github.io

- [Versions - Roboflow Python](https://roboflow.github.io/roboflow-python/core/version/)
  > model_format=None [...] :param model_ [...] to use for [...] :param location: An optional [...] :param overwrite: An optional flag to prevent dataset overwrite [...] ): An optional path for saving [..

### stackoverflow.com

- [Load dataset from Roboflow in colab](https://stackoverflow.com/questions/69594288/load-dataset-from-roboflow-in-colab)
  > I'm trying to retreive a roboflow project dataset in google colab. It works for two of the dataset versions, but not the latest I have created (same project, version 5). [...] ``` from roboflow import
