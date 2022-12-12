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
    """ this function gets the input arguments from the user"""
    parser = argparse.ArgumentParser(description = 'Train a neural network however you like!')
    parser.add_argument('--data_directory', default = 'flowers', help = "Specify the data directory")
    parser.add_argument('--save_dir', type = str, default = './checkpoint_1.pth', help='Choose a directory to save the neural network.')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'CNN model archeticture')
    parser.add_argument('--learning_rate', type = float, default = 0.0001, help = 'Learning rate for your model')
    parser.add_argument('--epochs', type = int, default = 15, help = 'number of epochs')
    parser.add_argument('--hidden_units', type = int, default = 4096, help = 'number of hidden units')
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

def get_data(data_dir):
    """ this function gets the data path and returns the path for the training, validation, and test data"""
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    return train_dir, valid_dir, test_dir

def data_transform(means, std_dev, train_dir, valid_dir, test_dir):
    """ this function takes the path of the data and returns a transformed version of the data"""
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(means, 
                                                            std_dev)])

# no need to perform randomization on validation/test samples; only need to normalize
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(means, 
                                                            std_dev)])

    test_transforms  = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(means, 
                                                            std_dev)])
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_datasets  = datasets.ImageFolder(test_dir,  transform = test_transforms)
    return train_datasets, valid_datasets, test_datasets


def data_loader(train_datasets, valid_datasets, test_datasets):
    """ this function takes the transformed datasets and returns data loaders"""
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64)
    test_loader  = torch.utils.data.DataLoader(test_datasets,  batch_size = 64)
    return train_loader, valid_loader, test_loader 


def label_mapping(directory):
    """ this function gets the actual names of the flower types"""
    with open(directory, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def build_classifier(arch, hidden_units, learning_rate):
    """ this function builds our classifier and returns a model with our classifier"""
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        #freezing our feature parameters
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([('hidden_1', nn.Linear(25088, hidden_units)),
                                         ('dropout', nn.Dropout(p=0.5)),
                                         ('relu_1', nn.ReLU()),
                                         ('hidden_2', nn.Linear(hidden_units,102)),
                                         ('output', nn.LogSoftmax(dim = 1))]))
        
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units >= 1024:
            hidden_units = 512
        classifier = nn.Sequential(OrderedDict([('hidden_1', nn.Linear(1024, hidden_units)),
                                         ('dropout', nn.Dropout(p=0.5)),
                                         ('relu_1', nn.ReLU()),
                                         ('hidden_2', nn.Linear(hidden_units,102)),
                                         ('output', nn.LogSoftmax(dim = 1))])) 
        
    elif arch == 'inception_v3':
        if hidden_units >= 2048:
            hidden_units = 1024
        model = models.inception_v3(pretrained = True)
        # freezing our feature parameters
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([('hidden_1', nn.Linear(2048, hidden_units)),
                                         ('dropout', nn.Dropout(p=0.5)),
                                         ('relu_1', nn.ReLU()),
                                         ('hidden_2', nn.Linear(hidden_units,102)),
                                         ('output', nn.LogSoftmax(dim = 1))])) 
        
    else:
        raise ValueError('Model arch error.')
        
    # changing the classifier in the original model to our classifier above
    model.classifier = classifier
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return model, classifier, criterion, optimizer
    
def train_model(model, criterion, optimizer, epochs, train_loader, valid_loader, device):
    """ this function trains the neural network."""
    
    # move the model to our preferred device
    model.to(device)
    # this will help us print data every 10 loops
    print_every = 10
    # overall training loss
    training_loss = 0
    # loop counter
    steps = 0

    for epoch in range(epochs):
        for images, labels in train_loader:
            # move the images and labels to our preferred device
            images, labels = images.to(device), labels.to(device)
            # increment the loop counter by one
            steps+= 1
            # Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()
            # forward propagation to get the probabilities on the log scale
            log_ps = model.forward(images)
            # calculate the loss
            loss = criterion(log_ps, labels)
            # backward propagation
            loss.backward()
            # take a step to lower the loss
            optimizer.step()
            # add the computed loss to the total tarianing loss
            training_loss += loss.item()

            # if 10 loops have passed:
            if steps % print_every == 0:
                # set the model to evaluation mode
                model.eval()
                # prediction accuracy
                accuracy = 0
                # loss on the validation set
                valid_loss = 0
                # turn off the gradient because we are testing
                with torch.no_grad():
                    # forward propagate, calculate the loss and prediction accuracy
                    for images, labels in valid_loader:
                        images, labels = images.to(device), labels.to(device)

                        log_ps = model.forward(images)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        # calculte accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                # print the results as the network trains        
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {training_loss/print_every:.3f}.. "
                      f"validation loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"validation accuracy: {accuracy/len(valid_loader):.3f}")
                # set the training loss to zero again
                training_loss = 0
                # set the model to training mode
                model.train();
    return model
        
def test_model(model, test_loader, device):
    """ this function tests our model"""
    model.to(device);
    accuracy = 0
    test_loss = 0
    with torch.no_grad():
        model.eval();
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)

            # calculte accuracy
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Test accuracy: {accuracy/len(test_loader):.3f}")

def set_checkpoint(model, directory, train_datasets,optimizer):
    """ this function savees a checkpoint for the model so we don't need to trian the model every time we use it"""
    if type(directory) == type(None):
        print('Checkpoint directory was not specified. Checkpoint cannot be saved')
    model.class_to_idx = train_datasets.class_to_idx
    # information about our network
    checkpoint = {
                  'features': model.features,
                  'classifier': model.classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_index': model.class_to_idx
                 }

    torch.save(checkpoint, directory)
    print('checkpoint saved at {}'.format(directory))
    return checkpoint


def main():
    # get the input arguments:
    in_args = get_input_args()
    # check if gpu is available for training
    device = check_gpu(in_args)
    # get the data path
    data_dir = in_args.data_directory
    # get the path for the training, validation, and test data
    train_dir, valid_dir, test_dir = get_data(data_dir)
    # set the means and the standard deviations
    means = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]
    # get the training, validation, and test  transformed datasets
    train_datasets, valid_datasets, test_datasets = data_transform(means, std_dev, train_dir, valid_dir, test_dir)
    # get the data loaders
    train_loader, valid_loader, test_loader = data_loader(train_datasets, valid_datasets, test_datasets)
    # label mapping
    cat_to_name = label_mapping('cat_to_name.json')
    # get the used archeticture
    arch = in_args.arch
    # get the number of hidden units
    hidden_units = in_args.hidden_units
    # get the learning rate
    learning_rate = in_args.learning_rate
    # get the number of epochs
    epochs = in_args.epochs
    # build the classifier
    model, classifier, criterion, optimizer = build_classifier(arch, hidden_units, learning_rate)
    # train the network
    trained_model = train_model(model, criterion, optimizer, epochs, train_loader, valid_loader, device)
    # test the model
    test_model(model, test_loader, device)
    # save the checkpoint
    save_dir = in_args.save_dir
    checkpoint = set_checkpoint(model, save_dir, train_datasets, optimizer)
main()