## Track B: Descubrimiento

**Tema:** Vast.ai container TensorFlow 2.15 PyTorch 2 CUDA 12 Python 3.10 dual stack 2026

**Resultados:** 31


### blog.tensorflow.org

- [What's new in TensorFlow 2.15](https://blog.tensorflow.org/2023/11/whats-new-in-tensorflow-2-15.html)
  > — *Posted by the TensorFlow team*TensorFlow 2.15 has been released! Highlights of this release (and 2.14) include a much simpler installation method for NVIDIA CUDA libraries for Linux, oneDNN CPU per

### cloud.google.com

- [Prebuilt containers for Vertex AI serverless training  |  Google Cloud Documentation](https://cloud.google.com/vertex-ai/docs/training/pre-built-containers)
  > Vertex AI provides Docker container images that you run as prebuilt containers for serverless training. These containers, which are organized by machine learning (ML) framework and framework version, 

### dev.to

- [The Zero-Trust Docker Pipeline: Securing GPU/AI Container Images from Build to Production](https://dev.to/pavan_madduri/the-zero-trust-docker-pipeline-securing-gpuai-container-images-from-build-to-production-50g2)
  > chain, npm, development headers — stays in the build stage. Here's the pattern I use for keda-gpu-scaler: ``` # === Build Stage === FROM golang:1.22-bookworm AS builder WORKDIR /app COPY go.mod go.sum

### docs.cloud.google.com

- [Choose a container image | Deep Learning Containers](https://docs.cloud.google.com/deep-learning-containers/docs/choosing-container)
  > Each container image provides a Python 3 environment and includes the selected data science framework (such as PyTorch or TensorFlow), Conda, the NVIDIA stack for GPU images (CUDA, cuDNN, NCCL2), and 
- [Prebuilt containers for Vertex AI serverless training](https://docs.cloud.google.com/vertex-ai/docs/training/pre-built-containers)
  > Vertex AI provides Docker container images that you run as prebuilt containers for serverless training. These containers, which are organized by machine learning (ML) framework and framework version, 

### docs.nvidia.com

- [TensorFlow Release 24.09 - NVIDIA Docs](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-24-09.html)
  > The NVIDIA container [...] of TensorFlow, release 24.09, is available on [...] This container image includes the complete source of the NVIDIA version of TensorFlow in `/opt/tensorflow`. It is prebuil

### docs.vast.ai

- [Choosing a Template - Vast.ai Documentation – Affordable GPU Cloud Marketplace](https://docs.vast.ai/guides/instances/choosing/templates)
  > > Select the right template for your Vast.ai instance. Templates define your Docker image, launch mode, and initialization settings. [...] Templates are saved configurations that define how your insta
- [Template Settings - Vast.ai Documentation – Affordable GPU Cloud Marketplace](https://docs.vast.ai/guides/templates/template-settings)
  > ## Docker Repository And Environment [...] This is where you define the Docker image you want to run, along with any options we want to pass into the container. [...] Here is where you can define the 
- [vastai create instance - Vast.ai Documentation – Affordable GPU Cloud Marketplace](https://docs.vast.ai/cli/reference/create-instance)
  > # vastai create instance [...] vastai create instance ID [...] OPTIONS] [--args ...] [...] docker container image to launch [...] is recommended for this [...] Creates an instance from an offer ID (wh
- [Overview - Vast.ai Documentation – Affordable GPU Cloud Marketplace](https://docs.vast.ai/instances-help)
  > > Instances are Docker containers that give you exclusive GPU access for training, inference, and development. Pay by the second, connect via SSH or Jupyter. [...] Instances are containerized environm
- [Creating and Using Templates with API - Vast.ai Documentation – Affordable GPU Cloud Marketplace](https://docs.vast.ai/api-reference/creating-and-using-templates-with-api)
  > | Field | Type | Description | | --- | --- | --- | | `name` | string | Required. Human-readable name for the template | | `image` | string | Required. Docker image path (e.g., `vastai/pytorch`) | | `t
- [Technical FAQ - Vast.ai Documentation – Affordable GPU Cloud Marketplace](https://docs.vast.ai/documentation/reference/faq/technical)
  > # Technical FAQ [...] > Docker configuration, performance, and advanced topics [...] ## Docker Configuration [...] ### What Docker options can I use? [...] Add Docker run arguments in the template con
- [Docker Execution Environment - Vast.ai Documentation – Affordable GPU Cloud Marketplace](https://docs.vast.ai/documentation/instances/docker-environment)
  > # Docker Execution Environment [...] > Learn how Vast.ai Docker instances handle resource allocation, environment variables, networking, ports, and CLI usage. [...] Vast.ai instances run as Linux Dock
- [PyTorch - Vast.ai Documentation – Affordable GPU Cloud Marketplace](https://docs.vast.ai/pytorch)
  > # Running PyTorch on Vast.ai: A Complete Guide [...] This guide walks you through setting up and running PyTorch workloads on Vast.ai, a marketplace for renting GPU compute power. Whether you're train
- [CUDA - Vast.ai Documentation – Affordable GPU Cloud Marketplace](https://docs.vast.ai/cuda)
  > # CUDA Programming on Vast.ai [...] This guide walks you through setting up and running CUDA applications on Vast.ai's cloud platform. You'll learn how to set up a CUDA development environment, connec

### github.com

- [Dockerfile at main · vast-ai/base-image](https://github.com/vast-ai/base-image/blob/main/Dockerfile)
  > ``` # Choose a base image.  Sensible options include ubuntu:xx.xx, nvidia/cuda:xx-cuddnx ARG BASE_IMAGE [...] RUN \     # Update libnccl for Blackwell GPUs     set -euo pipefail && \     if [[ "$BASE_
- [README.template.md at main · vast-ai/base-image](https://github.com/vast-ai/base-image/blob/main/README.template.md)
  > This is a simple demo template running only the base docker image.   All included software is enabled by default. [...] This is our foundational Docker base image, designed to serve as the starting po
- [Add cuda12 variant of tensorflow-notebook (#2100) · b9553a8 · jupyter/docker-stacks](https://github.com/jupyter/docker-stacks/commit/b9553a8e5d33a8d59eac52b0d8790d3f46f6f03c)
  > ## Add cuda12 variant of tensorflow-notebook (#2100) [...] Add cuda12 variant for tensorflow-notebook  - Reduce size of CPU version of tensorflow-notebook  - [...] x86_64-scipy [...] +  x86_64-tensorf
- [awsteiner/foundation](https://github.com/awsteiner/foundation)
  > Repo for docker images combining Torch and TensorFlow on Ubuntu/openSUSE/Arch some of which have nvcc/CUDA support. [...] Repository for constructing Docker images combining Torch and TensorFlow, some
- [tensorflow[and-cuda] 2.15.0/2.15.1 compatibility with jax[cuda12 ...](https://github.com/tensorflow/tensorflow/issues/68290)
  > ## tensorflow[and-cuda] 2.15.0/2.15.1 compatibility with jax[cuda12] [...] - Author: @attaluris - State: closed (completed) - Labels: stat:awaiting response, type:bug, stale, TF 2.15 - Assignees: @sus
- [GitHub - ai-dock/pytorch: PyTorch docker images for use in GPU ...](https://github.com/ai-dock/pytorch)
  > # Repository: ai-dock/pytorch  PyTorch docker images for use in GPU cloud and local environments. Includes AI-Dock base for authentication and improved user experience.  - Stars: 15 - Forks: 4 - Watch
- [Vast.ai Base Docker Image - GitHub](https://github.com/vast-ai/base-image)
  > A feature-rich base image designed for GPU computing on Vast.ai. This image extends large, commonly-used base images to maximize Docker layer caching benefits, resulting in faster instance startup tim
- [vast-ai/vast-cli](https://github.com/vast-ai/vast-python)
  > Vast.ai python and cli api client [...] - Stars: 191 - Forks: 83 - Watchers: 7 - Open issues: 34 - Primary language: Python - Languages: Python (98.8%), Shell (1.2%) - License: MIT License (MIT) - Top
- [build.sh at main · vast-ai/base-image](https://github.com/vast-ai/base-image/blob/main/build.sh)
  > BUILD_CONFIGS=(     "stock-22|ubuntu:22.04|stock-ubuntu22.04-py\${py_version_tag}|linux/amd64,linux/arm64|3.7|3.14"     "stock-24|ubuntu:24.04|stock-ubuntu24.04-py\${py_version_tag}|linux/amd64,linux/

### hal.science

- [](https://hal.science/hal-04728894v1/file/9.pdf)
  > a choice between the three leading frameworks: PyTorch [...] [1], TensorFlow [2], and JAX [3], which have limited crosscompatibility. This means that if an RL environment library [...] 3.1 PyTorch [.

### hub.docker.com

- [vastai/tensorflow - Docker Image](https://hub.docker.com/r/vastai/tensorflow/tags)
  > Tensorflow docker image built on cuda 10. [...] ``` docker pull vastai/tensorflow:2.19.0-cuda-12.4.1 ``` [...] ``` docker pull vastai/tensorflow:2.16.1-cuda-12.4.1 ``` [...] docker pull vastai/tensorf
- [vastai/pytorch - Docker Image](https://hub.docker.com/r/vastai/pytorch/tags)
  > ## vastai/pytorch [...] Pytorch 1.0 rc0 with cuda 10.0. [...] ``` docker pull vastai/pytorch:cuda-13.0.2-auto ``` [...] ``` docker pull vastai/pytorch:cuda-12.9.1-auto ``` [...] ``` docker pull vastai

### huggingface.co

- [Setting Up a Stable GPU Environment for PyTorch and TensorFlow](https://huggingface.co/blog/daya-shankar/stable-pytorch-tensorflow-gpu-environment)
  > Path A: Containers first (most stable). [...] Docker is already mainstream for dev workflows, with 59% of professional developers reporting they use it. Containers let you pin the whole user space, in

### math.ens.psl.eu

- [](https://www.math.ens.psl.eu/~feydy/geometric_data_analysis_draft.pdf)
  > Our PyTorch/NumPy routines fully support automatic differentiation and scale [...] up to millions of samples in seconds. They generally outperform baseline GPU [...] implementations with x10 to x1,000

### vast.ai

- [Run PyTorch, TensorFlow on GPU Rentals | AI/ML Frameworks](https://vast.ai/use-cases/ai-ml-frameworks)
  > Run PyTorch, TensorFlow on GPU Rentals | AI/ML Frameworks | Vast.ai  ## AI/ML Frameworks  Execute leading frameworks rapidly on scalable GPU infrastructure.  ### Built for This  - Run popular ML frame
- [April 2026 Product Update](https://vast.ai/article/april-2026-product-update)
  > It's been a busy few months at Vast.ai, with updates spanning performance and reliability improvements, new templates and guides, and a major step forward for serverless deployment workflows. [...] ##
