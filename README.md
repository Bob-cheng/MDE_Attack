# Monocular Depth Estimation Attack
This is the reference PyTorch implementation for "Physical Attack on Monocular Depth Estimation in Autonomous Driving with Optimal Adversarial Patches"

## Setup
### Dependencies
We recommend use Anaconda to manage the packegs and dependencies. Run the following command to install the required depencencies in a new environment.
```
conda install --yes --file requirements.txt
```

### Dataset
We use KITTI 3D object detection dataset as our background scene dataset. It can be downloaded [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Then you need to organize the data in the following way. The image split files can be downloaded [here](https://github.com/charlesq34/frustum-pointnets/tree/master/kitti/image_sets).
```
KITTI/object/
    
    train.txt
    val.txt
    test.txt 
    
    training/
        calib/
        image_2/ #left image
        image_3/ #right image
        label_2/
        velodyne/ 

    testing/
        calib/
        image_2/
        image_3/
        velodyne/
```

### Data Preparation
- Style Image Folder: `./DeepPhotoStyle_pytorch/asset/src_img/style`

  Put style image `XXX.png`, and style mask `XXX_StyleMask.png` inside.

- Content Image Folder: `./DeepPhotoStyle_pytorch/asset/src_img/content`

  Put content image `XXX.png`, and content mask `XXX_ContentMask.png` inside.

- Object Image Folder: `./DeepPhotoStyle_pytorch/asset/src_img/car`

  Put object Image `XXX.png`, object mask `XXX_CarMask.png` and fixed patch regions `XXX_PaintMaskAA.png` (AA is the paint mask number).

- Download Monodepth2 model at [here](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip) and unzip it to `./model/` 

## Adversarial Optimization

1. specify directory locations in `./DeepPhotoStyle_pytorch/utils.py`:
```
kitti_object_path  = '/path/to/kitti/object/'
project_root       = '/path/to/project/root/'
log_dir            = '/path/to/logdir'
```

2. Run the following command to start generating adversatial patch. For explainations of each command line options, see `python my_main.py -h`.

```
cd ./DeepPhotoStyle_pytorch

python my_main.py -s Warnning.png -c Warnning.png -v BMW.png -pm -2 --steps 10000 -lr 0.3689 -cw 1000 -sw 1000000 -at disp -aw 1000000 -tw 0.0001 -bs 6 -mw 1000.0 -dm monodepth2 -rw 10 --random-scene -lp mono_car_Rob_disp --late-start -bl proposed -sl 2 
```

3. check the attack performance with tensorboard
```
tensorboard --logdir '/path/to/logdir' --samples_per_plugin images=200
```