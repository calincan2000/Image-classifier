# Project image classifier from scratch that will identify different species of flowers
![image classifier output](https://github.com/calincan2000/Image-classifier/blob/master/inference_example.png)

Going forward, AI algorithms will be incorporated into more and more everyday applications. 
For example, you might want to include an image classifier in a smart phone app. 
To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. 
A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. 
You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. 
In practice you'd train this classifier, then export it for use in your application. 
We'll be using this dataset of 102 flower categories, you can see a few examples below.


### The project is broken down into multiple steps:

Load and preprocess the image dataset
Train the image classifier on your dataset
Use the trained classifier to predict image content
We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. 
Here your network will be learning about flowers and end up as a command line application. 
But, what you do with your new skills depends on your imagination and effort in building a dataset. 
For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. 
Go build your own dataset and make something new.

### Getting Started

1. These notebook require PyTorch v0.4 or newer, and torchvision. The easiest way to install PyTorch and torchvision locally is by following the instructions on the PyTorch site which can be found on [link ](https://pytorch.org/get-started/locally/) . Choose the stable version, your appropriate OS and Python versions, and how you'd like to install it. You'll also need to install numpy and jupyter notebooks, the newest versions of these should work fine. Using the conda package manager is generally best for this,[conda install numpy jupyter notebook]

   If you haven't used conda before, [please read the documentation](https://conda.io/en/latest/) to learn how to create environments and install packages. I suggest installing Miniconda instead of the whole Anaconda distribution. The normal package manager pip also works well. If you have a preference, go with that.

   PyTorch uses a library called [CUDA](https://developer.nvidia.com/cuda-zone) to accelerate operations using the GPU. If you have a GPU that CUDA supports, you'll be able to install all the necessary libraries by installing PyTorch with conda. 

2. If you can't use a local GPU, you can use cloud platforms such as AWS, GCP, and FloydHub to train your networks on a GPU.[The project can be oppend also using  Google Colab](https://colab.research.google.com/) or using  [Kaggle Kernels](https://www.kaggle.com)
3. How to reproduce the results can be found in [Jupyter Notebook  file](https://github.com/unhcr/Jetson/blob/master/Finding-the-Nexus/FindTheNexusDeepLearning/VHI%20and%20Displacements%20from%20Somanlia.ipynb) the same dataset split between training and testing for predicting and checking the prediction

---
