# Contrastive Unpaired Translation (CUT)

### [website](http://taesung.me/ContrastiveUnpairedTranslation/) |   [paper](https://arxiv.org/pdf/2007.15651)

We provide our PyTorch implementation of unpaired image-to-image translation based on patchwise contrastive learning and adversarial learning.  No hand-crafted loss and inverse network is used. Compared to [CycleGAN](https://github.com/junyanz/CycleGAN), our model training is faster and less memory-intensive. In addition, our method can be extended to single image training, where each “domain” is only a *single* image.

[Contrastive Learning for Unpaired Image-to-Image Translation](http://taesung.me/ContrastiveUnpairedTranslation/)  
 [Taesung Park](https://taesung.me/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/), [Richard Zhang](https://richzhang.github.io/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)<br>
UC Berkeley and Adobe Research<br>
 In ECCV 2020

## Example Results

### Unpaired Image-to-Image Translation
<img src="imgs/results.gif" width="800px"/>

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Getting started

- Clone this repo:
```bash
git clone https://github.com/gretelai/contrastive-unpaired-translation CUT
cd CUT
```

- Install PyTorch 1.1 and other dependencies (e.g., torchvision, visdom, dominate, gputil).

  For pip users, please type the command `pip install -r requirements.txt`.

  For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.


### CUT and FastCUT Training and Test

- Download the `ebike_locations` dataset (Maps -> Maps with Scooter Locations)
```bash
bash ./datasets/download_ebike_data.sh
```
The dataset is downloaded and unzipped at `./datasets/ebike_locations/`.

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

- Train the FastCUT model:
- 
 ```bash
python train.py --dataroot ./datasets/ebike_locations --name locations_FastCUT --CUT_mode FastCUT
```
The checkpoints will be stored at `./checkpoints/locations_FastCUT/web`.

- Test the CUT model:
```bash
python test.py --dataroot ./datasets/ebike_locations --name locations_FastCUT --CUT_mode FastCUT --phase train
```

The test results will be saved to a html file here: `./results/locations_FastCUT/latest_train/index.html`.

### Training using our launcher scripts

Please see `experiments/location_launcher.py` that generates the above command line arguments. The launcher scripts are useful for configuring rather complicated command-line arguments of training and testing.

Using the launcher, the command below generates the training command of CUT and FastCUT.
```bash
python -m experiments location_data train 1  # FastCUT
```

To test using the launcher,
```bash
python -m experiments location_data test 1   # FastCUT
```

### Apply a pre-trained CUT model and evaluate FID

To run the pretrained models, run the following.

```bash

# Download and unzip the pretrained models. The weights should be located at
# checkpoints/horse2zebra_cut_pretrained/latest_net_G.pth, for example.
wget https://gretel-public-website.s3.amazonaws.com/datasets/fastcut_models/pretrained_models.tar.gz
tar -zxvf pretrained_models.tar.gz

# Generate outputs. The dataset paths might need to be adjusted.
# To do this, modify the lines of experiments/pretrained_launcher.py
# [id] corresponds to the respective commands defined in pretrained_launcher.py
# 1 - FastCUT on ebike_data
python -m experiments pretrained run_test [id]
```

#### Preprocessing of input images

The preprocessing of the input images, such as resizing or random cropping, is controlled by the option `--preprocess`, `--load_size`, and `--crop_size`. The usage follows the [CycleGAN/pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repo. 

For example, the default setting `--preprocess resize_and_crop --load_size 286 --crop_size 256` resizes the input image to `286x286`, and then makes a random crop of size `256x256` as a way to perform data augmentation. There are other preprocessing options that can be specified, and they are specified in [base_dataset.py](https://github.com/taesungp/contrastive-unpaired-translation/blob/master/data/base_dataset.py#L82). Below are some example options. 

 - `--preprocess none`: does not perform any preprocessing. Note that the image size is still scaled to be a closest multiple of 4, because the convolutional generator cannot maintain the same image size otherwise. 
 - `--preprocess scale_width --load_size 768`: scales the width of the image to be of size 768.
 - `--preprocess scale_shortside_and_crop`: scales the image preserving aspect ratio so that the short side is `load_size`, and then performs random cropping of window size `crop_size`.

More preprocessing options can be added by modifying [`get_transform()`](https://github.com/taesungp/contrastive-unpaired-translation/blob/master/data/base_dataset.py#L82) of `base_dataset.py`. 


### Citation
[paper](https://arxiv.org/pdf/2007.15651).
```
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

