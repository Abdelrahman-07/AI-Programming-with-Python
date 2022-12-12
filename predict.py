import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns
import PIL
import json
import time
import argparse

def get_input_args():
    parser = argparse.ArgumentParser(description = 'use the network yoou trained to get predictions!')
    parser.add_argument('--path', type = str, default = './checkpoint_1.pth', help='Choose the directory where you saved the neural network.')
    parser.add_argument('--image_path', type = str, default = 'flowers/test/33/image_06454.jpg', help = 'Choose an image to predict the flower type')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'file path for the flower category names')
    parser.add_argument('--classes_probs', type = int, default = 5, help = 'number of displayed predicted types for the flower image')
    parser.add_argument('--gpu', action = 'store_true', help = 'this enables gpu learning')
    input_arguments = parser.parse_args()
    return  input_arguments

def check_gpu(in_args):
    """ this function checks if a gpu is available for accelerated learning. it takes the input args as an input and returns the device"""
    if in_args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def load_checkpoint(path):
    
    """ 
    this function takes a path as its input and returns a model saved at the soecified path
    """
   
    # load network information
    checkpoint = torch.load(path)
    model = models.vgg16(pretrained = True)
    classifier = nn.Sequential(OrderedDict([('hidden_1', nn.Linear(25088, 4096)),
                                         ('dropout', nn.Dropout(p=0.5)),
                                         ('relu_1', nn.ReLU()),
                                         ('hidden_2', nn.Linear(4096,102)),
                                         ('output', nn.LogSoftmax(dim = 1))]))
    model.classifier = classifier
    for params in model.parameters():
        params.requires_grad = False
    model.class_to_idx = checkpoint['class_to_index']
    model.load_state_dict(checkpoint['state_dict'], strict = False)
    return model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = PIL.Image.open(image) 
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    np_image = np.array(transform(image))
    
    return np_image



def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # turn off gradient becuase we are not training
    with torch.no_grad():
        model.to(device)
        # process the image:
        image = process_image(image_path)
        # convert image into tensor:
        image = torch.from_numpy(image)
        # Returns a new tensor with a dimension of size one inserted at the specified position:
        image.unsqueeze_(0)
        # convert the image into float. I used cpu because the expected type is torch.FloatTensor not torch.cuda.FloatTensor :
        image = image.float()
        image.type(torch.FloatTensor)
     
        # enter evaluation mode:
        model.eval();
        # run the image through the network:
        outputs = model(image.to(device))
        # get the top probaabilities and classes:
        probs, classes_p = torch.exp(outputs).topk(topk)
        # turn probabilities into lists:
        probs = probs.tolist()[0]
        # turn categories into lists:
        classes_p = classes_p.tolist()[0]
        # convert from these indices to the actual class labels:
        index = {value: key for key, value in model.class_to_idx.items()}
        classes = [index[i] for i in classes_p]
        #Returns the tensors as a (nested) list/ matrix:
        return probs, classes
    
    
def label_mapping(directory):

    with open(directory, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
    
def main():
    # get the input args:
    in_args = get_input_args()
    # check if a gpu is available
    device = check_gpu(in_args)
    # get the checkpoint path:
    path = in_args.path
    # load the checkpoint:
    model = load_checkpoint(path)
    # get the image path:
    image_path = in_args.image_path
    # process the image:
    process_image(image_path)
    # get the actual flower name:
    flower_label = image_path.split('/')[2]
    directory = in_args.category_names
    cat_to_name = label_mapping(directory)
    flower_name = cat_to_name[str(flower_label)]
    # get the predicted classes and their probabilities:
    topk = in_args.classes_probs
    top_p, top_class = predict(image_path, model, topk, device)
    # get the predicted classes names:
    flower_types= [cat_to_name[str(classes)] for classes in top_class]
    
    # print the results
    print("Actual flower name:")
    print(flower_name)
    print("Predicted flower types:")
    print(flower_types)
    print("Probability:")
    print(top_p)
main()