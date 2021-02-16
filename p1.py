# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 2021

@author: Tom

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
from PIL import Image
from advertorch.attacks import PGDAttack
from advertorch.utils import NormalizeByChannelMeanStd

batch_size=8

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
        
    #Create a normalization layer to append to pretrained model
    normalization_layer = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    #Append normalization layer to the front of the model, now we don't have to use the ImageNet normalization the tensor (Image) beforehand
    #And now the output gradients are for the loss with respect to the unnormalized tensor (Image) input
    model = torch.nn.Sequential(normalization_layer, model) 
    model.to(device)
    model.eval()

    ''' Test Network '''
    if(FLAGS.PGD=='off'):
        run_test(model, FLAGS, device)
    if(FLAGS.PGD=='on'):
        print("network, image_number, epsilon, L_norm, label, sample_predicted_label, adv_image_predicted_label, saved_adv_image_predicted_label, adv_image_L2_distance, saved_adv_image_L2_distance, adv_image_Linf_distance, saved_adv_image_Linf_distance, sample_loss, adv_image_loss, saved_adv_image_loss")
        
        results = []
        #Used to transform training images from ImageNet
        transform = transforms.Compose([
            transforms.Resize(256), #Imagenet resizes to 256
            transforms.CenterCrop(224), #And then center crops to 224
            transforms.ToTensor() #Converts to tensor, new range is [0,1]
        ])

        dataset = ImageNetDataset(FLAGS.img_dir, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=FLAGS.num_workers)
        
        for i, batch in enumerate(dataloader):
            # if(i!=2):
            #     continue
            # if(i>2):
            #     break
            sample = batch['image'].to(device)
            target = batch['target'].squeeze(1).to(device)

            sample.requires_grad = True
            prediction = model.forward(sample)
            loss = torch.nn.CrossEntropyLoss()
            output = loss(prediction.float(), target)
            
            model.zero_grad()
            output.backward()

            for e in range(2,11):                
                #Adversary object performs the attack
                adversary = create_adversary(model, e, FLAGS.norm)

                adv_sample = adversary.perturb(sample, target)
                adv_sample.to(device)
                adv_prediction = model.forward(adv_sample)

                #Save adversarial images to disk in PNG format so we get discrete pixel values
                saved_adv_sample = torch.empty(batch_size, 3, 224, 224).to(device)
                cloned_adv_sample = adv_sample.clone().to(device)
                for n in range(batch_size):
                    image_number = (i*batch_size)+n+1
                    image_path = "temp/saved_adv"+FLAGS.model+"_"+str(FLAGS.norm)+".PNG"

                    save_image(cloned_adv_sample[n], image_path)
                    saved_adv_sample[n] = transforms.ToTensor()(Image.open(image_path).convert('RGB'))
  
                    #Saves adv_image and perturbation for report
                    if(FLAGS.save_img == image_number):
                        save_img_path = "temp/AdvImage_eps"+str(e)+".PNG"
                        save_image(saved_adv_sample[n].clone().detach(), save_img_path)
                        
                        save_img_path = "temp/Perturbation_eps"+str(e)+".PNG"
                        save_image(((sample[n].clone().detach()-saved_adv_sample[n].clone().detach())*10)+0.5, save_img_path)
                        
                        #Print out the original training sample
                        if(e==2):
                            save_img_path = "temp/SampleImage.PNG"
                            save_image(sample[n].clone().detach(), save_img_path)

                
                #Predict saved adversarial images
                saved_adv_prediction = model.forward(saved_adv_sample)
                
                #Output Results
                for n in range(batch_size):
                    image_number = (i*batch_size)+n+1
                    sample_pred = prediction.argmax(dim=1, keepdim=True)[n].item() #Model's prediction on original image
                    
                    adv_pred = adv_prediction.argmax(dim=1, keepdim=True)[n].item() #Model's prediction on adversarial image
                    saved_adv_pred = saved_adv_prediction.argmax(dim=1, keepdim=True)[n].item() #Model's prediction on saved adversarial image
                    
                    adv_L2_distance = torch.dist(sample[n], adv_sample[n], 2).item() #L2 Distance between original and adversarial image
                    saved_adv_L2_distance = torch.dist(sample[n], saved_adv_sample[n], 2).item() #L2 Distance between original and saved adversarial image
                    
                    adv_Linf_distance = torch.norm(sample[n]-adv_sample[n], float("inf")).item() #Linf Distance between original and adversarial image
                    saved_adv_Linf_distance = torch.norm(sample[n]-saved_adv_sample[n], float("inf")).item() #Linf Distance between original and saved adversarial image
                    
                    single_sample_loss = round(loss(prediction[n].unsqueeze(0), target[n].unsqueeze(0)).item(), 6) #Loss for original training sample
                    single_adv_loss = round(loss(adv_prediction[n].unsqueeze(0), target[n].unsqueeze(0)).item(), 6) #Loss for adversarial image
                    single_saved_adv_loss = round(loss(saved_adv_prediction[n].unsqueeze(0), target[n].unsqueeze(0)).item(), 6) #Loss for saved adversarial image

                    result = [
                        FLAGS.model, 
                        str(image_number), 
                        str(e),
                        "L"+str(FLAGS.norm),
                        str(target[n].item()), 
                        str(sample_pred), 
                        str(adv_pred), 
                        str(saved_adv_pred), 
                        str(round(adv_L2_distance, 6)), 
                        str(round(saved_adv_L2_distance, 6)), 
                        str(round(adv_Linf_distance, 6)), 
                        str(round(saved_adv_Linf_distance, 6)),
                        str(single_sample_loss),
                        str(single_adv_loss),
                        str(single_saved_adv_loss)
                    ]
                    print(",".join(result))
                    results.append(result)

        print("len(results):", len(results))

def run_test(model, FLAGS, device):
    #
    batch_size = FLAGS.batch_size
    transform = transforms.Compose([
        transforms.Resize(256), #Imagenet resizes to 256
        transforms.CenterCrop(224), #And then center crops to 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = ImageNetDataset(FLAGS.img_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=FLAGS.num_workers)

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


def create_adversary(model, epsilon, norm):
    '''
    
    Parameters
    ----------
    model : TYPE
        network model.
    epsilon : number
        max perturbation.
    norm : "2" or "inf"
        Norm used for bounding L_p ball.

    Returns
    -------
    adversary : advertorch adversary
        Adversary object used for performing the attack

    '''
    # Ref: https://advertorch.readthedocs.io/en/latest/advertorch/attacks.html#advertorch.attacks.PGDAttack
    # ==============================================================================
    # class advertorch.attacks.PGDAttack(
    #                                   predict, 
    #                                   loss_fn=None, 
    #                                   eps=0.3, 
    #                                   nb_iter=40, 
    #                                   eps_iter=0.01, 
    #                                   rand_init=True, 
    #                                   clip_min=0.0, 
    #                                   clip_max=1.0, 
    #                                   ord=<Mock name='mock.inf' id='140083310782224'>, 
    #                                   l1_sparsity=None, 
    #                                   targeted=False)
    # The projected gradient descent attack (Madry et al, 2017). The attack performs
    # nb_iter steps of size eps_iter, while always staying within eps from the initial
    # point. Paper: https://arxiv.org/pdf/1706.06083.pdf

    # Parameters:	
    #     predict – forward pass function.
    #     loss_fn – loss function.
    #     eps – maximum distortion.
    #     nb_iter – number of iterations.
    #     eps_iter – attack step size.
    #     rand_init – (optional bool) random initialization.
    #     clip_min – mininum value per input dimension.
    #     clip_max – maximum value per input dimension.
    #     ord – (optional) the order of maximum distortion (inf or 2).
    #     targeted – if the attack is targeted.
    
    # ==============================================================================
    # advertorch.attacks.PGDAttack.perturb(self, 
    #                                      x, 
    #                                      y=None)
    # Given examples (x, y), returns their adversarial counterparts with an attack length of eps.

    # Parameters:	
    #     x – input tensor.
    #     y – label tensor. - if None and self.targeted=False, compute y as predicted labels.
    #                       - if self.targeted=True, then y must be the targeted labels.
    # Returns:	
    #     tensor containing perturbed inputs.
    
    iterations = epsilon*2 #From project description
    if(norm=="2" or norm==2):
        L_norm = 2
        step_size = (1.0*epsilon)/iterations
        epsilon_normalized = float(epsilon*224*224) #Multiply by number of pixels, L2 max perturbation also limited to epsilon
    else:
        L_norm = float("inf")
        step_size = 1.0/255.0
        epsilon_normalized = epsilon/255.0 #Normalize to same range as image tensor [0,1]

    adversary = PGDAttack(
                          model, 
                          eps=epsilon_normalized, 
                          eps_iter=step_size, 
                          nb_iter=iterations,
                          rand_init=False, #Maybe this should be true?
                          targeted=False,
                          ord=L_norm
                         )
    return adversary

if __name__ == '__main__':
    # Set parameters 
    parser = argparse.ArgumentParser('Project Phase 1 (PGD).')
    parser.add_argument('--model',
                        type=str, 
                        default='vgg',
                        help='(resnet or vgg) Select VGG or ResNet model.')
    parser.add_argument('--PGD',
                        type=str,
                        default='on',
                        help='(on or off) Whether to use adversarial images created with PGD or original imagenet images.')
    parser.add_argument('--norm',
                        type=str, 
                        default='2',
                        help='(2 or inf) Which norm to use to constrain the adversarial images')
    parser.add_argument('--img_dir',
                        type=str, 
                        default='../../Data/ImageNet2012/ILSVRC2012_img_val/',
                        help='Directory location of ImageNet validation images')
    parser.add_argument('--batch_size',
                        type=int, 
                        default=1,
                        help='Batch size for testing network')
    parser.add_argument('--save_img',
                        type=int, 
                        default=3,
                        help='Which training image to save in output or -1 to not save any.')
    parser.add_argument('--num_workers',
                        type=int, 
                        default=0,
                        help='Number of worker processes for the dataloader.')

    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    print("FLAGS", FLAGS)

    #Keep track of run time
    start = time.time()
    run_main(FLAGS)
    end = time.time()
    print("Elapsed Time")
    print(end - start)