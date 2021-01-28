# AdvCV_P1
Advanced Computer Vision Project Phase #1

## Validation Dataset
Requires 6.7G ImageNet2012 Validation dataset which can be found (https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5)[here]:


## Run Code:
``` 
$ conda activate
$ python p1.py --help
 usage: Project Phase 1 (PGD). [-h] [--model MODEL] [--PGD PGD] [--epsilon EPSILON]
                     [--img_dir IMG_DIR] [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         (resnet or vgg) Select VGG or ResNet model.
  --PGD PGD             (on or off) Whether to use adversarial images created
                        with PGD or original imagenet images.
  --epsilon EPSILON     (2 to 10) Max allowed perturbation
  --img_dir IMG_DIR     Directory location of ImageNet validation images
  --batch_size BATCH_SIZE
                        Batch size for testing network
```
