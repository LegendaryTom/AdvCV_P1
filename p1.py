# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 2021

@author: Tom

Ref: https://calebrob.com/ml/imagenet/ilsvrc2012/2018/10/22/imagenet-benchmarking.html
"""
import torch
import torch.nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
import time
import argparse
from torch.utils.data import DataLoader
from dataloader import ImageNetDataset
import matplotlib.pyplot as plt
from PIL import Image

batch_size=8
#Taken from: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def run_main(FLAGS):
    batch_size=FLAGS.batch_size
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

    ''' Test Network '''
    if(FLAGS.PGD=='off'):
        run_test(model, FLAGS, device)
    if(FLAGS.PGD=='on'):

        results = []
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # do_normalize = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        inv = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) #Note, this modifies the tensor in-place
        dataset = ImageNetDataset(FLAGS.img_dir, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        for i, batch in enumerate(dataloader):
            sample = batch['image'].to(device)
            target = batch['target'].squeeze(1).to(device)

            sample.requires_grad = True
            prediction = model.forward(sample)
            loss = torch.nn.CrossEntropyLoss()
            output = loss(prediction.float(), target)
            
            model.zero_grad()
            output.backward()

            for e in range(2,11):
                #Since the image needs to be normalized, we also divide step_size and epsilon be normalized just like the image
                adv_sample = projected_gradient_descent(model, sample, target, loss, num_steps=e*2, step_size=1/255.0, eps=e/255.0, eps_norm='inf', step_norm='inf', clamp=(-255, 255))
                adv_sample.to(device)
                adv_prediction = model.forward(adv_sample)

                #Save adversarial images to disk in PNG format so we get discrete pixel values
                saved_adv_sample = torch.empty(batch_size, 3, 224, 224).to(device)
                cloned_adv_sample = adv_sample.clone().to(device)
                for n in range(batch_size):
                    image_path = "temp/saved_adv.PNG"
                    save_image(inv(cloned_adv_sample[n]), image_path)
                    saved_adv_sample[n] = transform(Image.open(image_path).convert('RGB'))
                
                #Predict saved adversarial images
                saved_adv_prediction = model.forward(saved_adv_sample)
                
                #Output Results
                for n in range(batch_size):
                    image_number = (i*batch_size)+n+1
                    sample_pred = prediction.argmax(dim=1, keepdim=True)[n].item() #Model's prediction on original image
                    adv_pred = adv_prediction.argmax(dim=1, keepdim=True)[n].item() #Model's prediction on adversarial image
                    adv_distance = (((sample[n]-adv_sample[n])**2).mean()**(1/2)).item() #Distance between original and adversarial image
                    
                    saved_adv_pred = saved_adv_prediction.argmax(dim=1, keepdim=True)[n].item() #Model's prediction on saved adversarial image
                    saved_adv_distance = (((sample[n]-saved_adv_sample[n])**2).mean()**(1/2)).item() #Distance between original and saved adversarial image
                    result = [FLAGS.model, str(image_number), str(e), str(target[n].item()), str(sample_pred), str(adv_pred), str(saved_adv_pred), str(round(adv_distance, 5)), str(round(saved_adv_distance, 5))]
                    print(",".join(result))
                    results.append(result)

        print("len(results):", len(results))

def run_test(model, FLAGS, device):
    batch_size = FLAGS.batch_size
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = ImageNetDataset(FLAGS.img_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    correct = 0
    for i, batch in enumerate(dataloader):
        '''
        batch['image'] is a batch of images (X)
        batch['target'] is a batch of labels (Y)
        '''
        batch['image'] = batch['image'].to(device)
        batch['target'] = batch['target'].to(device)
        # if(i==10):
        #     break

        predictions = model.forward(batch['image']).argmax(dim=1, keepdim=True)
        correct += predictions.eq(batch['target'].view_as(predictions)).sum().item()

        print(i, "Result:", correct, "/", (i+1)*batch_size)
    print("Result:", correct, "/", (i+1)*batch_size)
    print("Accuracy:", correct/((i+1)*batch_size))


#Adapted from: https://towardsdatascience.com/know-your-enemy-7f7c5038bdf3
def projected_gradient_descent(model, x, y, loss_fn, num_steps, step_size, step_norm, eps, eps_norm, clamp=(0,1), y_target=None):
    """Performs the projected gradient descent attack on a batch of images."""
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    targeted = y_target is not None
    num_channels = x.shape[1]

    for i in range(num_steps):
        _x_adv = x_adv.clone().detach().requires_grad_(True)

        prediction = model(_x_adv)
        loss = loss_fn(prediction, y_target if targeted else y)
        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            if step_norm == 'inf':
                gradients = _x_adv.grad.sign() * step_size
            else:
                # Note .view() assumes batched image data as 4D tensor
                # print("_x_adv.grad:", _x_adv.grad)
                # print("step_size:", step_size)
                # print("_x_adv.grad * step_size:", _x_adv.grad * step_size)
                # print("_x_adv.grad.view(_x_adv.shape[0], -1):", _x_adv.grad.view(_x_adv.shape[0], -1))
                # print("_x_adv.grad.view(_x_adv.shape[0], -1).shape:", _x_adv.grad.view(_x_adv.shape[0], -1).shape)
                # print("_x_adv.shape:", _x_adv.shape)
                # print("_x_adv.grad.view(_x_adv.shape[0], -1).norm(step_norm, dim=-1):", _x_adv.grad.view(_x_adv.shape[0], -1).norm(step_norm, dim=-1))
                # print("_x_adv.grad.view(_x_adv.shape[0], -1).norm(step_norm):", _x_adv.grad.view(_x_adv.shape[0], -1).norm(step_norm))
                # print("step_norm:", step_norm)
                # print("_x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1).norm(step_norm, dim=-1):", _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1).norm(step_norm, dim=-1))
                #gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1).norm(step_norm, dim=-1)#.view(-1, num_channels, 1, 1)
                # print("_x_adv.grad.view(_x_adv.shape[0], -1).norm(step_norm, dim=-1):", _x_adv.grad.view(_x_adv.shape[0], -1).norm(step_norm, dim=-1))
                # print("_x_adv.grad.norm(step_norm, dim=-1):", _x_adv.grad.norm(step_norm, dim=-1))
                gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1).norm(step_norm, dim=-1).view(batch_size, -1, 1, 1)

            if targeted:
                # Targeted: Gradient descent with on the loss of the (incorrect) target label
                # w.r.t. the image data
                x_adv -= gradients
            else:
                # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                # the model parameters
                x_adv += gradients

        # Project back into l_norm ball and correct range
        if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        else:
            delta = x_adv - x

            # Assume x and x_adv are batched tensors where the first dimension is
            # a batch dimension
            mask = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1) <= eps

            scaling_factor = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1)
            scaling_factor[mask] = eps

            # .view() assumes batched images as a 4D Tensor
            delta *= eps / scaling_factor.view(-1, 1, 1, 1)

            x_adv = x + delta
        
        x_adv = x_adv.clamp(*clamp)
    
    return x_adv.detach()

if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('Project Phase 1 (PGD).')
    parser.add_argument('--model',
                        type=str, 
                        default='resnet',
                        help='(resnet or vgg) Select VGG or ResNet model.')
    parser.add_argument('--PGD',
                        type=str,
                        default='on',
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
                        default=8,
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