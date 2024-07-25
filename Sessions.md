# **<ins>ibm-skills-ai-colab-sessions</ins>**
> ## IBM Skills Build AI Fundamentals - Colab - Sessions

### **Objectives**
<center style=font-size:18px><strong><ins>A. Portfolio focused Project Based Learning</ins>

<ins>B. Local v Remote Deployment/Runtimes</ins>

<ins>C. Self Directed Configuration of VSCode and Python Locally</ins>
</strong></center>

---
>  ...
---

## **<ins>A. Portfolio focused Project Based Learning</ins>**

*Artefacts from Live Technical Sessions in the form of*:

- `Session 1`: Python Fundamentals (for beginners and new to Python). **(2024.06.19)**
  - [<ins>`🖇️ Session1.ipnyb`</ins>](./notebooks-labs/Session1.ipynb "Folder: notebooks-labs: Session 1: Python Fundamentals"): <br>
    **CoLab** Run -> [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10ZTbzzrTTrYpzu4xFcN88gV_Rai4Y-Rj?usp=sharing "Session 1: Google Colab: Session2_Regression_Clustering_Classification_Recommender.ipnyb"):
    - NB: *Was familar with Python Fundamentals from previous software engineering efforts and courses.*
      - i) Lists, Tuples, and Dictionaries
      - ii) Basic Python Operations
      - iii) Flow Control Structoures
      - iv) Handling errors
      - v) Functions
    - Recommended Activities
      1.   Code with Mosh [Complete Python Mastery ](https://codewithmosh.com/p/python-programming-course-beginners)
      2.   Practice Katas, for example, [Code Wars](https://www.codewars.com/), [CodeSignal](https://learn.codesignal.com/)
- `Session 2`: Machine Learning Models and Methodologies Fundamentals. **(2024.07.02)**
  - [<ins>`🖇️ Session2.ipnyb`</ins>](./notebooks-labs/Session2.ipynb "Folder: notebooks-labs: Session 2: Machine Learning Fundamentals") <br>
      **CoLab** Run -> [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FW5-OGD2jegulfkF8afRptkZ3cEakL--?usp=sharing "Session 2: Google Colab: Session2_Regression_Clustering_Classification_Recommender.ipnyb"):
    - i) Regressions
    - ii) Classifications 
    - iii) Clustering 
    - iv) Recommender Systems 
- `Session 3`: Generative AI Lab **(2024.07.16)** 
  - [<ins>`🖇️  Session3_VAE.ipnyb`</ins>](./notebooks-labs/Session3_VAE.ipynb "Folder: notebooks-labs: Session 3: Assumed GitHub as Host, Use Google Colabl as Host"): <br>
    **CoLab** Run -> [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eD7pRKmhVFl0nfwzsoIy9RTtoPMVkZPW?usp=sharing "Session 3: Google Colab: Session3_VAE.ipynb")
    - i) Load Datasets
    - ii) Encoders
    - iii) VAE Sampling
    - iv) Decoders
    - v) VAE Model
    - vi) VAE Loss
    - vii) Model Training
    - viii) Display Images (func) 
    <br><br>
  - [<ins>`🖇️ Session3_Transformers.ipnyb`</ins>](./notebooks-labs/Session3_FineTuning_BERTandGPT.ipynb "Folder: notebooks-labs: Session 3: Assumed GitHub as Host, Use Google Colabl as Host"): <br> 
    **CoLab** Run -> [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19YcHhGy__BdZp3iDGeiytA0ZfCrfiPCY?usp=sharing "Session 3: Google Colab: Session3_VAE.ipynb")
    - i) Setups/Imports
    - ii) Load Datasets
    - iii) Load Transformer Model (BERT)
    - iv) Training Params
    - v) Trainer
    - vi) Model Evaluation
    - vii) Predictions
    <br><br>

---
>  ...
---

## **<ins>B. Local v Remote Deployment/Runtimes</ins>**

Learning and Execution of Notebook and Remote Environments for Deployment/Runtimes

### Local Setup & Limitations

- [`🖇️ Setup.md`](./Setup.md "Setting up local environment for Juypter")
  - Effectively, my local system does not have the compute for running Session 3 models. 
  - Defaults to free compute limitations of Google Colab or alternate dedicated platforms (e.g. IBM WatsonX).

### Remote Notebook Workspaces

#### [Google Colab](https://colab.research.google.com/)

- Authorise GitHub as a service so to save your Colab notebook files. 
  - If you save with results of execution, these the are saved.
- Certain amount of free compute credits are available for light workloads only or until threashold is reached.
- Launch CoLab buttons are attatched to `.ipnyb` files when saved/committed from Colab direct into GitHub.

> - GoTo: https://colab.research.google.com/
> - Help: https://colab.research.google.com/notebooks/basic_features_overview.ipynb

#### Python Libraries

##### *`Session 1`*:

- Standard Libraries, Python 3.10

##### *`Session 2`*: 

- Standard Libraries, Python 3.10
- Regression
    - NumPy: Random Seeds etc.
    - MatplotLib: PyPlot
- Classifications
    - SciKit-Learn: Data Algorithms, Analysis, Modeling.
        - Datasets: Irises
        - Linear Models: LogisticRegression
        - Clustering: KMeans, Agglomerative Clustering, DBSCAN
- Recommender Systems:
    - SciKit-Learn:
        - Metrics: Pairwise / Cosine Similarity
    - Pandas: Data Structures

##### *`Session 3`*:

- Standard Libraries, Python 3.10
- Variational AutoEncoder (VAE)
    - TensorFlow: TF, Layers (PyTorch)
    - NumPy: Random Seeds etc.
    - MatplotLib: PyPlot
    - Dataset: MINST 
        - (Keras Dataset): https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
- Tuning Transformers: BERT / GPT
    - <small>`! pip install -U`</small> the following:
        - Accelerate: v: 0.32.1
            - Depends: Satisfied on Google Colab.
                - numpy, packaging, psutil, pyyaml, torch, huggingface-hub, safetensors, filelock, typing-extensions, sympy, networkx, jinja2, fsspec<br>
                triton, requests, tqdm, MarkupSafe, charset-normalizer, idna, urllib3, certifi, mpmath
            - NVidia: 
                - nvidia-cuda-nvrtc-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-cupti-cu12, nvidia-cudnn-cu12, nvidia-cublas-cu12, nvidia-cufft-cu12<br>
                nvidia-curand-cu12, nvidia-cusolver-cu12, nvidia-cusparse-cu12, nvidia-nccl-cu12, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12
        - Transformers: v: 4.42.4
            - Depends: 
                - filelock, huggingface-hub, numpy, packaging, pyyaml, regex, requests, safetensors, tokenizers, tqdm, fsspec, typing-extensions <br>
                charset-normalizer, idna, urllib3, certifi.
            - Imports: Libraries:
                - BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
    - <small>`! pip install -p`</small> the following:
        - Torch: v: 2.3.0+cu121
        - Datasets: load_dataset, load_metric

---
>  ...
---

## **<ins>C. Machine Learning Methods & Approaches</ins>** 

- `Session 2`
- `Session 3`

### 1. Unsupervised Learning: `Session 2`

### *Regression*

#### NumPy
>  NumPy, short for "Numerical Python," is a powerful library used in Python programming for numerical and scientific computing. 

NumPy like a supercharged version of Python's built-in list data structure, designed to handle large amounts of data more efficiently.

### Matplotlib 

> Matplotlib is a powerful library in Python used for creating visualizations, such as graphs and charts. 

MatplotLibs is particularly useful for data scientists, engineers, and anyone who needs to visualize data to understand and communicate trends, patterns, and insights

### *Classification*

#### SciKit_Learn

> Scikit-learn is a popular Python library for machine learning, offering simple and efficient tools for data analysis and modeling

SciKit_Learn (`sklearn`) provides a wide range of algorithms for classification, regression, clustering, and dimensionality reduction. 
- It integrates well with other scientific libraries like NumPy and pandas
- As such, makes it easy to build and evaluate machine learning models. 
- Is widely used for its 
    - ease of use, 
    - comprehensive documentation, and 
    - versatility in handling different machine learning tasks.

### Clustering

1. K-Means Clustering
2. Hierarchical Clustering
3. DBSCAN

#### K-Means Clustering

> K-Means is an unsupervised machine learning algorithm that partitions a dataset into k distinct clusters based on similarities, aiming to minimize the sum of squared distances between data points and their assigned cluster centroids

It minimizes within-cluster variances (squared Euclidean distances), facilitating partitioning by mean rather than Euclidean distances.

#### Hierarchical Clustering

> Hierarchical Clustering is an unsupervised machine learning algorithm that groups unlabeled data points into a hierarchy of clusters based on their similarity. An analytical method that seeks to build a hierarchy of clusters by either merging or splitting them based on data observations.

It builds a cluster hierarchy in the form of a tree-like structure called a dendrogram, where each merge or split is represented by a node
- Agglomerative (Bottom-up) - Starting small, think of this as starting with one feature as its own group
- Divisive (Top-down) - Starting big, think of this as starting with the whole box of features as one big group

#### DBSCAN

> DBSCAN is an unsupervised clustering algorithm that groups together closely packed data points based on their density, while identifying points in low-density regions as outliers or noise.

- DBSCAN is known as Density-Based Spatial Clustering of Applications with Noise.

It operates by defining clusters as areas where a minimum number of points (minPts) exist within a specified radius (epsilon) around each point, allowing it to detect clusters of arbitrary shapes and effectively handle noise in datasets

### Recommender Systems

---

### 2. Generative AI: VAE: `Session 3`

These sessions needs to be run on <small>[![GoogleColab](https://img.shields.io/badge/Google-CoLab-blue?logo=googlecolab&logoColor=white)](https://colab.research.google.com/ "Free Compute Credits across CPU/GPU")</small> if local system compute are not configured or specified for GPU loads.

#### TensorFlow

![TensorFlow](https://img.shields.io/badge/TensorFlow-Python-FF6F00?logo=tensorflow&logoColor=white): [Website](https://github.com/tensorflow/tensorflow) | [Guide](https://www.tensorflow.org/guide/keras "High Level API for TensorFlow") | [GitHub](https://github.com/keras-team/keras) | [PyPi](https://pypi.org/project/keras/)

> TensorFlow is an end-to-end open source platform for machine learning and it is easy to create ML models that can run in any environment. 

- It has a comprehensive, flexible ecosystem of tools, libraries, and community resources to  build and deploy ML-powered applications.
    - Lite lirbaries for mobile and edge devices
    - Browser libraries
    - ML models & datasets
    - Developer tools for model evaluation, performance optimisation and productising ML workflows.

##### TensorFlow & Keras 3 (<small>Source:<sup>PyPi</sup></small>)

![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white): [Website](keras.io/) | [Guide](https://www.tensorflow.org/guide/keras "High Level API for TensorFlow") | [Getting Started](https://keras.io/getting_started/) | [GitHub](https://github.com/keras-team/keras) | [PyPi](https://pypi.org/project/keras/)

> Keras is a multi-backend deep learning framework, with support for JAX, TensorFlow, and PyTorch

-  It provides an approachable, highly-productive interface for solving machine learning (ML) problems, with a focus on modern deep learning.
- Build and train models for computer vision, natural language processing, audio processing, timeseries forecasting, recommender systems, etc.
- To use keras, you should also install the backend of choice: `tensorflow`, `jax`, or `torch`.
- NB:  Note that `tensorflow` is required for using certain Keras 3 features: certain preprocessing `layers` as well as `tf.data` pipelines.
    - Keras 3 is intended to work as a drop-in replacement for `tf.keras` (when using the TensorFlow backend).

#### Others <small><sup>See above</sup></small>

- NumPy
- MatployLib: PlyPlot

### 3. Generative AI: Tuning Transformers: `Session 3`

#### Accelerate
![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Accelerate-blue): [Website](https://huggingface.co/docs/accelerate/en/quicktour "HuggingFace Accelerate") | [GitHub](https://github.com/huggingface/accelerate/tree/main "HuggingFace Accelerate: GitHub.com") | [PyPi](https://pypi.org/project/accelerate/ "HuggingFace Accelerate: PyPi.org") 

> HuggingFace's 🤗 library that enables the same PyTorch code to be run across any distributed configuration. 

- It's run your *raw* PyTorch training script on any kind of device.
- Accerlate was created for PyTorch users who like to write the training loop of PyTorch models 
    - ... but are reluctant to write and maintain the boilerplate code needed to use multi-GPUs/TPU/fp16.
- Accelerate abstracts exactly and only the boilerplate code related to multi-GPUs/TPU/fp16.

#### Transformers

![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-blue): [Website](https://huggingface.co/docs/transformers/en/index "HuggingFace Transformers") | [GitHub](https://github.com/huggingface/transformers/tree/main "HuggingFace Transformers: GitHub.com") | [PyPi](https://pypi.org/project/transformers/ "HuggingFace Transformers: PyPi.org")

> HuggingFace provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and share them on HuggingFace's mode; hub.

-  It provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio.

**UseCases** (<small>Source:<sup>PyPi</sup></small>)

These models can be applied on:

- 📝 Text, for tasks like text classification, information extraction, question answering, summarization, translation, and text generation, in over 100 languages.
- 🖼️ Images, for tasks like image classification, object detection, and segmentation.
- 🗣️ Audio, for tasks like speech recognition and audio classification.
- Transformer models can also perform tasks on several modalities combined, such as
    - Table question answering, 
    - Ooptical character recognition, 
    - Information extraction from scanned documents, 
    - Video classification, and 
    - Visual question answering.

---
> <center>...</center>
---