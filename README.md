# AdvCV_P1
Advanced Computer Vision Project Phase #1

## Validation Dataset
Requires 6.7G ImageNet2012 Validation dataset which can be found (https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5)[here]:

## Required Modules:
Requires torchnet, pandas, and advertorch.
```
pip install torchnet --user
pip install pandas --user
pip install advertorch --user
```

## Run Code on newton:
Change the slurm file to add the desired flags (--img_dir, --batch_size, and --num_workers). Slurm file automatically installs required modules using "--user" flag.
``` 
$ git clone https://github.com/LegendaryTom/AdvCV_P1.git
$ cd AdvCV_P1/
$ p1.slurm
```

## Run code on local machine:
```
$ conda activate
$ python p1.py --img_dir='../Data/ImageNet2012/ILSVRC2012_img_val' --batch_size=32 --model=resnet --num_workers=8 --PGD=on --norm=inf
$ python p1.py --img_dir='../Data/ImageNet2012/ILSVRC2012_img_val' --batch_size=32 --model=resnet --num_workers=8 --PGD=on --norm=2
$ python p1.py --img_dir='../Data/ImageNet2012/ILSVRC2012_img_val' --batch_size=32 --model=vgg --num_workers=8 --PGD=on --norm=inf
$ python p1.py --img_dir='../Data/ImageNet2012/ILSVRC2012_img_val' --batch_size=32 --model=vgg --num_workers=8 --PGD=on --norm=2
```
