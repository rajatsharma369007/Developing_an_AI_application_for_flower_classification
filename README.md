# Developing an AI application for flower classification

## Introduction
In this repository, I have implemented a command line application to train the model and then using the trained model you can input a flower image to predict a flower category. I have used method of transfer learning here.

## Dataset
I have used [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) having 102 flower categories. 

## Installation
* Numpy  
<code>conda install numpy</code>
* Matplotlib  
<code>conda install matplotlib</code>
* Pil  
<code>conda install -c anaconda pil</code>
* Pytorch  
<code>conda install torch</code>

## How to run scripts
__For train.py__  
Argument to be passed:
* data directory  
<code>--data_dir</code>
* saved checkpoint directory  
<code>--save_dir</code>
* gpu training  
<code>--gpu</code>
* architecture  
<code>--arch</code>
* learning rate  
<code>--learning_rate</code>
* hidden units  
<code>--hidden_units</code>
* number of epochs  
<code>--epochs</code>

__For predict.py__  
Argument to be passed:
* input image directory  
<code>--image</code>
* checkpoint directory  
<code>--check_point</code>
* gpu usage  
<code>--gpu</code>
* top k classes with probability  
<code>--top_k</code>
* category to name  
<code>--category_to_name</code>

first run:  
<code>python train.py --data_dir dir_path --gpu True --learning_rate 0.001</code>

after successful training run:  
<code>python predict.py --image image_name.jpg --checkpoint checkpoint.pth --gpy True</code>

## Issue/Bug
Please open issues on github to report bugs or make feature requests.

## Contribution
If you are interested in improving the code, please open an issue first to describe the task you are planning to do. For small fixes (a few lines of change) feel free to open pull requests directly.

