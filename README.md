# 🧠 AI Lab - Docker Image for Machine Learning & Data Science

[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com/)
[![Debian](https://img.shields.io/badge/Debian-12-blue.svg)](https://www.debian.org/)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-enabled-orange)](https://jupyter.org/)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA%20%2F%20ROCm-green)](https://developer.nvidia.com/cuda-zone)

---

## 📘 Overview

**AI Lab** is a ready-to-use Docker image providing a full environment for data science, machine learning, and deep learning. It includes support for:

- **Training on CPU or GPU (CUDA / ROCm)**
- **Jupyter Notebook + nbextensions + ipywidgets**
- **Visualization and statistical modeling**
- **Time series and date/time analysis**
- **Structured and unstructured data modeling**
- **Support for NetCDF, Excel, CSV, and more**

---

## 🚀 Key Features

- ✅ Ready for **TensorFlow**, **PyTorch**, **Ultralytics**
- ✅ Includes tools like `Transformers`, `Optuna`, `MLFlow`, `Ray`, `DVC`
- ✅ Full support for `ipywidgets` interactive elements
- ✅ Full **LaTeX & Markdown rendering** in Jupyter
- ✅ Reads `.csv`, `.xlsx`, `.xls`, `.nc` (NetCDF), `HDF` files
- ✅ Includes statistical and time-series packages

---

## 📚 Included Packages

A comprehensive and organized list of essential **Python packages** used in **Data Science**, **Machine Learning**, **Deep Learning**, **Computer Vision**, **Natural Language Processing**, **Data Visualization**, **Model Deployment**, **Database Interaction**, and more.

---

### 🧮 Data Science & Machine Learning

#### 🔹 Core Libraries

| Package          | Description                                                                                                                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **numpy**        | Fundamental package for scientific computing in Python. Provides support for large, multi-dimensional arrays and matrices with optimized performance under the hood using C/Fortran. Essential for numerical operations. |
| **pandas**       | Powerful data manipulation library built on NumPy. Introduces DataFrame and Series objects for structured data analysis, including tools for cleaning, transforming, and exploring datasets.                             |
| **scikit-learn** | Comprehensive machine learning library offering classification, regression, clustering, and preprocessing tools. Features a consistent API for model training, evaluation, and pipelines.                                |

---

#### 🔥 Deep Learning

| Package        | Description                                                                                                                                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **tensorflow** | End-to-end platform for ML and deep learning developed by Google. Supports both high-level Keras API and low-level customization. Ideal for production systems and scalable ML solutions.                   |
| **torch**      | Flexible deep learning framework with dynamic computation graphs, preferred in research environments. Developed by Facebook’s AI Research lab. Offers GPU acceleration and an intuitive Pythonic interface. |

---

#### 🧠 Specialized ML

| Package          | Description                                                                                                                                                                                                           |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **transformers** | State-of-the-art NLP library by Hugging Face. Includes thousands of pretrained models like BERT, GPT, T5 for text classification, translation, summarization, and generation. Compatible with PyTorch and TensorFlow. |
| **ultralytics**  | High-performance computer vision library specializing in object detection and segmentation using YOLO models. Known for real-time inference speed and production-ready deployment tools.                              |

---

### 📈 Data Visualization

#### 📊 Basic Plotting

| Package        | Description                                                                                                                                                                     |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **matplotlib** | Foundational plotting library for creating static, animated, and interactive visualizations. Highly customizable and serves as the base for many other visualization libraries. |
| **seaborn**    | High-level statistical visualization library built on matplotlib. Simplifies creation of visually appealing plots for categorical data, distributions, and regression trends.   |

#### 💡 Interactive Visualization

| Package       | Description                                                                                                                                                                      |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **plotly**    | Library for creating interactive, publication-quality charts. Supports 3D plots, geographic maps, and financial charts. Integrates with Dash for building analytical dashboards. |
| **streamlit** | Fast framework to build interactive web apps from Python scripts. Ideal for turning ML models and data exploration tools into shareable interfaces with minimal effort.          |

---

### ⚙️ Utilities & Preprocessing

#### 🖼️ Data Processing

| Package           | Description                                                                                                                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **opencv-python** | Open Source Computer Vision library. Contains algorithms for image/video processing, facial recognition, augmented reality, and more. Used across robotics, surveillance, and medical imaging. |
| **Pillow**        | Friendly fork of PIL for image processing. Supports opening, modifying, and saving various image formats. Useful for basic image transformations and filters.                                  |

---

#### 🛠️ Workflow Tools

| Package    | Description                                                                                                                                                     |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **tqdm**   | Lightweight progress bar for loops and iterables. Helps monitor long-running operations with minimal overhead. Supports Jupyter and custom formatting.          |
| **joblib** | Lightweight pipelining library for caching expensive computations and parallel execution. Commonly used in scikit-learn for model persistence and optimization. |

---

### 🗃️ Databases & Storage

#### 🗄️ SQL Databases

| Package             | Description                                                                                                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **psycopg2-binary** | PostgreSQL adapter for Python. Implements DB API 2.0 and supports advanced PostgreSQL features like asynchronous notifications and COPY commands. Binary version avoids compilation requirements. |
| **PyMySQL**         | Pure-Python MySQL client that doesn’t require external dependencies. Implements Python DB API v2.0 with support for prepared statements and connection pooling.                                   |

#### 📦 NoSQL Databases

| Package     | Description                                                                                                                                                            |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **pymongo** | Official MongoDB driver for Python. Enables working with documents and collections using an intuitive API. Supports aggregation, GridFS, and change streams.           |
| **redis**   | Python interface to Redis, a fast key-value store. Supports transactions, pub/sub messaging, Lua scripting, and is widely used for caching and real-time applications. |

---

### 🤖 Model Deployment & APIs

#### 🌐 Web Frameworks

| Package     | Description                                                                                                                                                                                          |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **fastapi** | Modern, high-performance web framework for building APIs using Python 3.7+. Based on type hints with automatic data validation and OpenAPI/Swagger documentation. Excellent for deploying ML models. |
| **gradio**  | Easy-to-use library for creating interactive UI around ML models. Great for demos, testing, and sharing models with non-technical users. Supports inputs like images, audio, and video.              |

#### 🏭 MLOps

| Package     | Description                                                                                                                                                                                                 |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **mlflow**  | Open source platform for managing the end-to-end ML lifecycle: experiment tracking, reproducible runs, and model deployment. Works with multiple frameworks.                                                |
| **bentoml** | Framework for serving, deploying, and monitoring ML models. Packages models with dependencies and provides high-performance serving with adaptive batching. Integrates with Kubernetes and cloud platforms. |

---

### 📁 File Formats & I/O

#### 📦 Specialized Formats

| Package     | Description                                                                                                                                                                           |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **pyarrow** | Python implementation of Apache Arrow. Provides a fast, language-independent columnar memory format for analytics. Enables efficient data sharing between Python, R, Spark, and more. |
| **h5py**    | Python interface to HDF5 binary data format. Efficient for storing and manipulating large numerical datasets. Supports hierarchical organization of data in groups and datasets.      |

#### 🔄 Data Transfer

| Package    | Description                                                                                                                                                             |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **kaggle** | Official Kaggle API client. Enables downloading datasets and competition entries programmatically. Useful for automating data science workflows.                        |
| **wandb**  | Weights & Biases library for tracking experiments, metrics, and hyperparameters during model training. Provides team dashboards and integrates with most ML frameworks. |

---

### 🎮 Reinforcement Learning

| Package | Description                                                                                                                                                                            |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **gym** | Toolkit for developing and comparing reinforcement learning algorithms. Provides standardized environments ranging from classic control problems to Atari games. Maintained by OpenAI. |

---

### 📝 Other Useful Packages

#### 🧱 3D Processing

| Package    | Description                                                                                                                                                      |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **open3d** | Modern library for 3D data processing. Supports point clouds, RGB-D images, and meshes. Includes algorithms for registration, reconstruction, and visualization. |

#### 📝 NLP Utilities

| Package      | Description                                                                                                                                         |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **tiktoken** | Fast BPE tokenizer used by OpenAI models. Optimized for tokenizing text for GPT and similar transformer-based models. Extremely fast and efficient. |

#### 🧰 System Tools

| Package   | Description                                                                                                                                                                          |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **blosc** | High-performance compression library for binary data. Multi-threaded and faster than traditional compressors for numerical arrays. Often used as a backend for scientific computing. |

---

## 🧪 Building the Image

Clone the repo and build:

```bash
docker build -t ai-lab .
```

---

## ▶️ Running the Container

### 💻 CPU Mode (default)

```bash
docker run -d -e TZ=America/Recife \
    -p 8888:8888 \
    -v $(pwd)/workspace:/workspace ai-lab
```

### ⚡ GPU Mode - CUDA (NVIDIA)

```bash
docker run -d --gpus all \
  -e TZ=America/Recife \
  -e ENABLE_GPU=yes \
  -e GPU_TYPE=CUDA \
  -p 8888:8888 -v $(pwd)/notebooks:/workspace ai-lab
```

### 🔷 GPU Mode - ROCm (AMD)

```bash
docker run -d \
  -e TZ=America/Recife \
  -e ENABLE_GPU=yes \
  -e GPU_TYPE=ROCM \
  -p 8888:8888 -v $(pwd)/notebooks:/workspace ai-lab
```

---

## docker-compose.yml

```
version: '3'

services:
  ai-lab:
    build: .
    container_name: ai-lab
    ports:
      - "8888:8888"
    volumes:
      - ./workspace:/workspace
    environment:
      ENABLE_GPU: ${ENABLE_GPU:-no}
      GPU_TYPE: ${GPU_TYPE:-CUDA}
    profiles:
      - cpu
      - gpu-cuda
      - gpu-rocm
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

# Configuración por perfil
profiles:
  cpu:
    description: "Ejecución en modo CPU (por defecto)"
  gpu-cuda:
    description: "Ejecución con soporte GPU NVIDIA (CUDA)"
    services:
      ai-jupyter:
        runtime: nvidia
        environment:
          ENABLE_GPU: "yes"
          GPU_TYPE: "CUDA"
          TZ: "America/Recife"

  gpu-rocm:
    description: "Ejecución con soporte GPU AMD (ROCm)"
    services:
      ai-jupyter:
        environment:
          ENABLE_GPU: "yes"
          GPU_TYPE: "ROCM"
          TZ: "America/Recife"

```

### Run as

### 💻 CPU Mode (default)

```bash
docker compose --profile cpu up --build
```

### ⚡ GPU Mode - CUDA (NVIDIA)

```bash
docker compose --profile gpu-cuda up --build
```

### 🔷 GPU Mode - ROCm (AMD)

```bash
docker compose --profile gpu-rocm up --build
```

---

## 📓 Jupyter Access

Once running, open your browser:

```
http://localhost:8888
```

> No token or password required.

---

## 🧩 Extensions Enabled

- `ipywidgets`: sliders, buttons, inputs
- `toc2`: table of contents
- `code_prettify`: code formatter
- `LaTeX`: full math support (`$\alpha + \beta = \gamma$`)

---

## 🧪 Recommended Notebook Tests

Run demo.ipynb at http://localhost:8888 or http://localhost:8888

---

## 📄 License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). You are free to use, share, and adapt the material, provided that appropriate credit is given to the original author.

---

📦 Software Repository and Citation Notice
This software is published under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
You are free to use, modify, and distribute this code for any purpose, provided that proper credit is given to the original author.

Please cite this software using the following reference:

Author: H. L. Varona
Title: AI Lab: Docker Image for Machine Learning & Data Science
Zenodo DOI: https://doi.org/10.5281/zenodo.15353983

BibTeX citation format:

```latex
@software{hlvarona-ailab-v1,
  author       = {H. L. Varona},
  title        = {AI Lab: Docker Image for Machine Learning & Data Science},
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {110.5281/zenodo.15353983}
}
```

---

## 👤 Author

**HL Varona**
📧 [humberto.varona@gmail.com](mailto:humberto.varona@gmail.com)
🔧 Project: `VaronaTech`
