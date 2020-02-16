# Project image classifier from scratch that will identify different species of flowers
![image classifier output](https://github.com/calincan2000/Image-classifier/blob/master/output_5_0.png)
![image classifier output](https://github.com/calincan2000/Image-classifier/blob/master/inference_example.png)

## Image-Classifier
In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice, you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories.

When you've completed this project, you'll have an application that can be trained on any set of labelled images. Here your network will be learning about flowers and end up as a command line application. 


## The project is broken down into multiple steps:

Load and preprocess the image dataset
Train the image classifier on your dataset
Use the trained classifier to predict image content
We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. 
Here your network will be learning about flowers and end up as a command line application. 
But, what you do with your new skills depends on your imagination and effort in building a dataset. 
For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. 
Go build your own dataset and make something new.

## Getting Started

1. These notebook require PyTorch v0.4 or newer, and torchvision. The easiest way to install PyTorch and torchvision locally is by following the instructions on the PyTorch site which can be found on [link ](https://pytorch.org/get-started/locally/) . Choose the stable version, your appropriate OS and Python versions, and how you'd like to install it. You'll also need to install numpy and jupyter notebooks, the newest versions of these should work fine. Using the conda package manager is generally best for this,[conda install numpy jupyter notebook]

   If you haven't used conda before, [please read the documentation](https://conda.io/en/latest/) to learn how to create environments and install packages. I suggest installing Miniconda instead of the whole Anaconda distribution. The normal package manager pip also works well. If you have a preference, go with that.

   PyTorch uses a library called [CUDA](https://developer.nvidia.com/cuda-zone) to accelerate operations using the GPU. If you have a GPU that CUDA supports, you'll be able to install all the necessary libraries by installing PyTorch with conda. 

2. If you can't use a local GPU, you can use cloud platforms such as AWS, GCP, and FloydHub to train your networks on a GPU.[The project can be oppend also using  Google Colab](https://colab.research.google.com/) or using  [Kaggle Kernels](https://www.kaggle.com)
3. How to reproduce the results can be found in Jupyter Notebook  file the same dataset split between training and testing for predicting and checking the prediction

---

## Data
The data used specifically for this assignment are a flower database(.json file). It is not provided in the repository as it's larger than what github allows.<br/>
The data need to comprised of 3 folders:
1. test
2. train 
3. validate<br/>

Generally the proportions should be 70% training 10% validate and 20% test.

Inside the train, test and validate folders there should be folders bearing a specific number which corresponds to a specific category, clarified in the json file. For example if we have the image x.jpg and it is a lotus it could be in a path like this /test/5/x.jpg and json file would be like this {...5:"lotus",...}. 

## GPU/CPU
As this project uses deep CNNs, for training of network you need to use a GPU. However after training you can always use normal CPU for the prediction phase.
