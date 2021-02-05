# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 2021

@author: Tom

Ref: https://calebrob.com/ml/imagenet/ilsvrc2012/2018/10/22/imagenet-benchmarking.html
"""
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import time
import argparse
from torch.utils.data import DataLoader
from dataloader import ImageNetDataset

def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    ''' Create Model '''
    if(FLAGS.model == 'vgg'):
        model = models.vgg16_bn(pretrained=True, progress=True)
    if(FLAGS.model == 'resnet'):
        model = models.resnet34(pretrained=True, progress=True)
    model.to(device)
    model.eval()
    
    #print("Using model")
    #print(model)
        
    ''' Test Network '''
    run_test(model, FLAGS, device)

def run_test(model, FLAGS, device):
    batch_size = FLAGS.batch_size
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = ImageNetDataset(FLAGS.img_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    correct = 0
    for i, batch in enumerate(dataloader):
        '''
        batch['image'] is a batch of images (X)
        batch['target'] is a batch of labels (Y)
        '''
        batch['image'] = batch['image'].to(device)
        batch['target'] = batch['target'].to(device)

        predictions = model.forward(batch['image']).argmax(dim=1, keepdim=True)
        
        correct += predictions.eq(batch['target'].view_as(predictions)).sum().item()
        
        print(i, "Result:", correct, "/", (i+1)*batch_size)
        
    print("Result:", correct, "/", (i+1)*batch_size)
    print("Accuracy:", correct/((i+1)*batch_size))

if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--model',
                        type=str, 
                        default='resnet',
                        help='(resnet or vgg) Select VGG or ResNet model.')
    parser.add_argument('--PGD',
                        type=str,
                        default='off',
                        help='(on or off) Whether to use adversarial images created with PGD or original imagenet images.')
    parser.add_argument('--epsilon',
                        type=int, 
                        default=2,
                        help='(2 to 10) Max allowed perturbation')
    parser.add_argument('--img_dir',
                        type=str, 
                        default='../../Data/ImageNet2012/ILSVRC2012_img_val/',
                        help='Directory location of ImageNet validation images')
    parser.add_argument('--batch_size',
                        type=int, 
                        default=10,
                        help='Batch size for testing network')

    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    print("FLAGS", FLAGS)

    #Keep track of run time
    start = time.time()
    run_main(FLAGS)
    end = time.time()
    print("Elapsed Time")
    print(end - start)