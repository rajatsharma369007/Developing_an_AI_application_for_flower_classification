import numpy as np
import json
import torch
import argparse
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# arguments to be passed
parser = argparse.ArgumentParser(description='Prediction of flower class')
parser.add_argument('--image', type=str, default='flowers/test/25/image_06593.jpg', help='image to be classified')
parser.add_argument('--check_point', type=str, default='checkpoint.pth', help='load the checkpoint')
parser.add_argument('--gpu', type=bool, default=False, help='to use GPU')
parser.add_argument('--top_k', type=int, default=5, help='top K classes with probabilities')
parser.add_argument('--category_to_name', type=str, default='cat_to_name.json', help='json mapping of the category to name')

args = parser.parse_args()

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(file_path):
    check_point = torch.load(file_path)
    model = models.vgg19(pretrained=True)
    model.to(device)
    model.classifier = check_point['classifier']
    model.load_state_dict(check_point['state_dict'])
    model.class_to_idx = check_point['class_to_idx']
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    image = Image.open(image)
    width = image.size[0]
    height = image.size[1]
    aspect = width/height
    
    if width <= height:
        image = image.resize((256,int(256/aspect)))
    else:
        image = image.resize((int(256*aspect),256))
    
    mid_w = image.size[0]/2
    mid_h = image.size[1]/2
    
    crop = image.crop((mid_w-112, mid_h-112, mid_w+112, mid_h+112))
    
    np_image = np.asarray(crop)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = (np_image - mean)/std
    image= image.transpose((2, 0, 1))
    
    return torch.from_numpy(image)

def predict(image_dir, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    image = process_image(image_dir)
    image = image.unsqueeze(0).float()
    
    device = torch.device('cuda:0' if args.gpu else 'cpu')
    
    image = image.to(device)
    
    model = load_checkpoint(model)
    
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
        
    prob, index = torch.topk(ps, top_k)
    pb = np.array(prob.data[0])
    Index = np.array(index.data[0])
    
    with open(args.category_to_name, 'r') as f:
        cat_to_name = json.load(f)
        
    idx_to_class = {idx:cla for cla,idx in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in Index]
    labels = [cat_to_name[cla] for cla in classes]

    return pb,labels

# TODO: Print the top K classes along with corresponding probabilities
probability, classes = predict(args.image, args.check_point, args.top_k)
print('Left: Possible Type   Right: Probability')
for pb, cla in zip(probability, classes):
    print("%20s: %f" % (cla, pb))
