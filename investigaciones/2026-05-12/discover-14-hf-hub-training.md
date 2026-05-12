## Track B: Descubrimiento

**Tema:** Hugging Face Hub upload model TensorBoard training pipeline checkpoints incremental 2026

**Resultados:** 35


### arxiv.org

- [Flow with FlorDB: Incremental Context Maintenance for the Machine Learning Lifecycle](https://arxiv.org/html/2408.02498v1)
  > The scope of FlorDB111 https://github.com/ucbrise/flor has been significantly expanded to encompass the entire ML lifecycle, covering complete pipelines and their regular execution, rather than just i

### discuss.huggingface.co

- [Continuing model training takes seconds in next round - Transformers](https://discuss.huggingface.co/t/continuing-model-training-takes-seconds-in-next-round/17549)
  > I’m currently working with the huggingface framework to train a binary classifier. I saved the newly trained model and wanted to use the checkpoint for incremental learning (what I basically want to d

### github.com

- [src/huggingface_hub/_tensorboard_logger.py at main · huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/_tensorboard_logger.py)
  > """Contains a logger to push training logs to the Hub, using Tensorboard.""" [...] class HFSummaryWriter(_RuntimeSummaryWriter):     """     Wrapper around the tensorboard's `SummaryWriter` to push tr
- [src/huggingface_hub/cli/upload.py at 0b55fb46 · huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub/blob/0b55fb46/src/huggingface_hub/cli/upload.py)
  > # Upload filtered directory (example: tensorboard logs except for the last run)     hf upload my-cool-model ./model/training /logs --include "*.tfevents.*" --exclude "*20230905*" [...] hf upload [...]
- [Release code and artifacts for Entropy Dynamics RFT paper on Hugging Face · Issue #503 · agentscope-ai/Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT/issues/503)
  > I saw your comment on the paper page that the code is "coming soon". It'd be great to make the code and any checkpoints or datasets resulting from your research available on the 🤗 hub, to improve thei
- [Upload models over to the Hugging Face Hub! · Issue #45 - GitHub](https://github.com/WongKinYiu/yolov9/issues/45)
  > # Issue: WongKinYiu/yolov9 #45  - Repository: WongKinYiu/yolov9 | Implementation of paper - YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information | 9K stars | Python  ## Uplo
- [Upload models to the Hugging Face Hub · Issue #1 - GitHub](https://github.com/atomicarchitects/equiformer_v2/issues/1)
  > # Issue: atomicarchitects/equiformer_v2 #1  - Repository: atomicarchitects/equiformer_v2 | [ICLR 2024] EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations | 336
- [docs/hub/models-uploading.md at main · huggingface/hub-docs](https://github.com/huggingface/hub-docs/blob/main/docs/hub/models-uploading.md)
  > To upload models to the Hub, you'll need to create an account at Hugging Face. Models on the Hub are Git-based repositories, which give you versioning, branches, discoverability and sharing features, 
- [docs/source/en/guides/upload.md at 9e46a06f · huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub/blob/9e46a06f/docs/source/en/guides/upload.md)
  > Sharing your files and work is an important aspect of the Hub. The `huggingface_hub` offers several options for uploading your files to the Hub. You can use these functions independently or integrate 
- [docs/hub/tensorboard.md at main · huggingface/hub-docs](https://github.com/huggingface/hub-docs/blob/main/docs/hub/tensorboard.md)
  > ```md # Using TensorBoard [...] TensorBoard provides tooling for tracking and visualizing metrics as well as visualizing models. All repositories that contain TensorBoard traces have an automatic tab 
- [docs/source/en/model_sharing.md at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_sharing.md)
  > The Hugging Face Hub is a platform for sharing, discovering, and consuming models of all different types and sizes. We highly recommend sharing your model on the Hub to push open-source machine learni
- [docs/source/en/using-diffusers/push_to_hub.md at main · huggingface/diffusers](https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/push_to_hub.md)
  > # Sharing pipelines and models [...] Share your pipeline or models and schedulers on the Hub with the [`~diffusers.utils.PushToHubMixin`] class. This class: [...] 1. creates a repository on the Hub 2.

### huggingface.co

- [Uploading models · Hugging Face](https://huggingface.co/docs/hub/en/models-uploading)
  > To upload models to the Hub, you'll need to create an account at Hugging Face. Models on the Hub are Git-based repositories, which give you versioning, branches, discoverability and sharing features, 
- [TensorBoard logger · Hugging Face](https://huggingface.co/docs/huggingface_hub/main/package_reference/tensorboard)
  > TensorBoard is well integrated with the Hugging Face Hub. The Hub automatically detects TensorBoard traces (such as `tfevents`) when pushed to the Hub which starts an instance to visualize them. To ge
- [Using TensorBoard · Hugging Face](https://huggingface.co/docs/hub/main/en/tensorboard)
  > TensorBoard provides tooling for tracking and visualizing metrics as well as visualizing models. All repositories that contain TensorBoard traces have an automatic tab with a hosted TensorBoard instan
- [Sharing · Hugging Face](https://huggingface.co/docs/transformers/v4.57.0/model_sharing)
  > Hugging Face Hub [...] and consuming models of all [...] your model on the Hub [...] This guide will show you how to share a model to the Hub from Transformers. [...] ## Uploading a model [...] There 
- [Sharing pretrained models - Hugging Face LLM Course](https://huggingface.co/learn/nlp-course/chapter4/3)
  > If you have played around with the `Trainer` API to train a model, the easiest way to upload it to the Hub is to set `push_to_hub=True` when you define your `TrainingArguments`: [...] When you call `t
- [Integrate any ML framework with the Hub](https://huggingface.co/docs/huggingface_hub/main/guides/integrations)
  > 1. Push to Hub: implement a method to upload a model to the Hub. This includes the model weights, as well as the model card and any other relevant information or data necessary to run the model (for e
- [Train a diffusion model · Hugging Face](https://huggingface.co/docs/diffusers/tutorials/basic_training)
  > Before you begin, make sure you have 🤗 Datasets installed to load and preprocess image datasets, and 🤗 Accelerate, to simplify training on any number of GPUs. The following command will also install T
- [Checkpointing · Hugging Face](https://huggingface.co/docs/accelerate/v0.26.1/usage_guides/checkpoint)
  > # Checkpointing [...] When training a PyTorch model with 🤗 Accelerate, you may often want to save and continue a state of training. Doing so requires saving and loading the model, optimizer, RNG gener
- [Upload files to the Hub · Hugging Face](https://huggingface.co/docs/huggingface_hub/v0.16.2/en/guides/upload)
  > # Upload files to the Hub [...] Sharing your files and work is an important aspect of the Hub. The`huggingface_hub` offers several options for uploading your files to the Hub. You can use these functi
- [Uploading models · Hugging Face](https://huggingface.co/docs/hub/models-uploading)
  > To upload models to the Hub, you'll need to create an account at Hugging Face. Models on the Hub are Git-based repositories, which give you versioning, branches, discoverability and sharing features, 
- [Push files to the Hub · Hugging Face](https://huggingface.co/docs/diffusers/v0.25.1/en/using-diffusers/push_to_hub)
  > # Push files to the Hub [...] 🤗 Diffusers provides a PushToHubMixin for uploading your model, scheduler, or pipeline to the Hub. It is an easy way to store your files on the Hub, and also allows you t
- [Using TensorBoard - Hugging Face](https://huggingface.co/docs/hub/tensorboard)
  > # Using TensorBoard  TensorBoard provides tooling for tracking and visualizing metrics as well as visualizing models. All repositories that contain TensorBoard traces have an automatic tab with a host
- [Introducing Storage Buckets on the Hugging Face Hub](https://huggingface.co/blog/storage-buckets)
  > Hugging Face Models and Datasets repos are great for publishing final artifacts. But production ML generates a constant stream of intermediate files (checkpoints, optimizer states, processed shards, l
- [Introducing Buckets: S3-like storage on the Hub](https://huggingface.co/changelog/introducing-storage-buckets)
  > Buckets bring mutable, non-versioned object storage to the Hub, available for users and organizations using your existing storage plan. Upload training checkpoints, intermediate artifacts, logs and pr
- [Summer at Hugging Face](https://huggingface.co/blog/summer-at-huggingface)
  > ### TensorBoard Integration [...] In late June, we launched a TensorBoard integration for all our models. If there are TensorBoard traces in the repo, an automatic, free TensorBoard instance is launch
- [Upload files to the Hub - Hugging Face](https://huggingface.co/docs/huggingface_hub/guides/upload)
  > Sharing your files and work is an important aspect of the Hub. The `huggingface_hub` offers several options for uploading your files to the Hub. You can use these functions independently or integrate 
- [Sharing - Hugging Face](https://huggingface.co/docs/transformers/en/model_sharing)
  > Hugging Face Hub [...] and consuming models of all [...] your model on the Hub [...] This guide will show you how to share a model to the Hub from Transformers. [...] ## Uploading a model [...] There 
- [Models compatible with the TensorBoard library - Hugging Face](https://huggingface.co/models?library=tensorboard&sort=modified)
  > Models compatible with the TensorBoard library – Hugging Face                           500B","label":"> 500B","type":"num_parameters"}]}},"orderedInferenceProviders":["groq","novita","cerebras","samb
- [Paper page - Rewriting Pre-Training Data Boosts LLM Performance in Math and Code](https://huggingface.co/papers/2505.02881)
  > context, and reformatting [...] Within a fixed 50 billion token [...] budget, continual pre-training of Llama-3.1-8B with SwallowCode boosts pass@1 by +17.0 on HumanEval and +16.1 on HumanEval+ compar

### marktechpost.com

- [Hugging Face Releases ml-intern: An Open-Source AI Agent that Automates the LLM Post-Training Workflow](https://www.marktechpost.com/2026/04/21/hugging-face-releases-ml-intern-an-open-source-ai-agent-that-automates-the-llm-post-training-workflow/)
  > manual effort from ML researchers and engineers. It operates as [...] loop that mirrors [...] an ML researcher [...] Hugging Face Papers, reading methodology sections, traversing citation graphs, and 

### scalarlm.com

- [Save Models to Hugging Face](https://www.scalarlm.com/save-fine-tuned-model-to-hugging-face/)
  > By default, ScalarLM saves fine-tuned model checkpoints locally inside the job directory on the server (e.g.`checkpoint_16.pt`). This guide shows you how to automatically push those checkpoints to the

### theneuralbase.com

- [Model upload and versioning | Huggingface Api Advanced Course | The Neural Base](https://theneuralbase.com/huggingface-api/learn/advanced/model-upload-and-versioning/)
  > Upload trained models to Hugging Face Hub with automatic versioning and revision control using the HfApi client. [...] Sharing models requires understanding how Hub versioning works: commits, revision

### zenodo.org

- [HF-NLP10K: A Dataset and Metadata Analysis of 10,000+ Hugging Face NLP Models](https://zenodo.org/records/15682522)
  > ,Limitations,Use Cases, [...] ization,Training Data,Comparison,Evaluation Metrics via Model Card,Base Model,Fine-Tuning,Private,Pipeline_tag, [...] _name,License,Region,Language,Language Count, [...] 
