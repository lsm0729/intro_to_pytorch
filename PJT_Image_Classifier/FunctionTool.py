import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import json
import numpy as np


def data_preprocess(data_dir='./flowers'):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    print('train_dir: ' ,train_dir)
    print('valid_dir: ', valid_dir)
    print('test_dir: ' , test_dir)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    data_loaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=32),
        'test': DataLoader(image_datasets['test'], batch_size=32)
    }
    print("done data preprocess!")
    return image_datasets,data_loaders




def build_model(device,dropout=0.5, hidden_layer=4096, lr=0.001, arch='vgg13'):
    if arch =='vgg13':
        model=models.vgg13(pretrained=True)
        input_layer=25088
    elif arch =='vgg16':
        model=models.vgg16(pretrained=True)
        input_layer=25088
    elif arch =='vgg19':
        model=models.vgg19(pretrained=True)
        input_layer=25088
    elif arch =='vgg11':
        model=models.vgg11(pretrained=True)
        input_layer=25088
    elif arch =='densenet121':
        model=models.densenet121(pretrained=True)
        input_layer=1024
    elif arch =='alexnet':
        model=models.alexnet(pretrained=True)
        input_layer=9216
    else :
        print("this arch {} is not supported with this script. run with vgg13".format(arch))
        print("suppoted arch list : vgg11, vgg13, vgg16, vgg19, densenet121, alexnet")
        model=models.vgg13(pretrained=True)
        input_layer=25088

    for param in model.parameters():
        param.requires_grad = False

    Classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_layer, hidden_layer)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_layer, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = Classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    if torch.cuda.is_available() and device:
        print("model runs with cuda!")
        model.to('cuda')
    print("Model build success!!")

    return model, criterion, optimizer


def train_model(model, criterion, optimizer,device,data_loaders, epochs=5):

    print("=====================train start============================")
    if torch.cuda.is_available() and device:
        print("model runs with cuda!")

    step=0
    for e in range(epochs):

        running_loss = 0

        for images, labels in data_loaders['train']:
            if torch.cuda.is_available() and device :
                images, labels = images.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step+=1

            if step%5==0:
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    model.eval()
                    for images, labels in data_loaders['valid']:
                        images, labels = images.to('cuda'), labels.to('cuda')
                        logps = model.forward(images)
                        valid_loss += criterion(logps, labels)
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                model.train()

                print(
                    "Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Train Loss: {:.3f}.. ".format(running_loss / len(data_loaders['train'])),
                    "Valid Loss: {:.3f}.. ".format(valid_loss / len(data_loaders['valid'])),
                    "Valid Accuracy: {:.3f}..".format(accuracy / len(data_loaders['valid']))
                )


    print("===================== train end ============================")
    return model


def test_model(model, device, data_loaders):
    accuracy = 0
    model.eval()
    print("start evaluate with test set")
    if torch.cuda.is_available() and device:
        print("model runs with cuda!")
        model.to('cuda')

    with torch.no_grad():
        for images, labels in data_loaders['test']:
            if torch.cuda.is_available() and device :
                images, labels = images.to('cuda'), labels.to('cuda')         
            logps = model(images)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(
        "Test set Accuracy: {:.3f}".format(accuracy / len(data_loaders['test']))
    )
def save_chkpoint(model, save_dir,image_datasets,hidden_layer,lr,epochs,arch):
    checkpoint = {
        'state_dict'  : model.state_dict(),
		'classifier'  : model.classifier,
        'class_to_idx': image_datasets['train'].class_to_idx,
        'arch'        : arch
    }
    torch.save(checkpoint, save_dir)
    print("sace checkpoint done")


def load_model(path):

    checkpoint = torch.load(path)
    arch = checkpoint['arch']
    print("arch is : {}".format(arch))
    if arch =='vgg13':
        model=models.vgg13(pretrained=True)
    elif arch =='vgg16':
        model=models.vgg16(pretrained=True)
    elif arch =='vgg19':
        model=models.vgg19(pretrained=True)
    elif arch =='vgg11':
        model=models.vgg11(pretrained=True)
    elif arch =='densenet121':
        model=models.densenet121(pretrained=True)
    elif arch =='alexnet':
        model=models.alexnet(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']     
    model.load_state_dict(checkpoint['state_dict'])   
    model.class_to_idx = checkpoint['class_to_idx']
    print("load model done!")
    
    return model


def label_mapping(dir='cat_to_name.json'):
    with open(dir, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name



def predict(image_path, model, topk, device, cat_to_name):
    ''' 
	Predict the class (or classes) of an image using a trained deep learning model.
    '''
    

    image = Image.open(image_path)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])    
    img = transform(image)  
  
    img = img.unsqueeze_(0)
    img = img.float()
    
    model.eval()
    if torch.cuda.is_available() and device:
        print("model runs with cuda!")
        model.to('cuda')
        img=img.to('cuda')
    
    with torch.no_grad(): 
        output = model.forward(img)
        
    probs_ = F.softmax(output.data,dim=1)    
    probs, classes = probs_.topk(topk)
    
    top_k = [cat_to_name[str(x+1)] for x in np.array(classes)[0]]
    
    print("predict result for image : {}".format(image_path))
    for f_type, p in zip(top_k,probs[0]):  
        print("{:<22} {: .3f}".format(f_type,float(p)))
        