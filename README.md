# Contrastive Unpaired Translation (CUT)

## Example Results

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Getting started

- Clone this repo:
```bash
git clone https://github.com/gretelai/contrastive-unpaired-translation
cd contrastive-unpaired-translation
```

- Install PyTorch 1.1 and other dependencies (e.g., torchvision, visdom, dominate, gputil).

  For pip users, please type the command `pip install -r requirements.txt`.

  For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.

## Create or download location datasets

### Download the ebike_locations dataset

- Download the `ebike_locations` dataset (Maps -> Maps with Scooter Locations)
```sh
sh datasets/download_ebike_data.sh
```
The dataset is downloaded and unzipped at `./datasets/ebike_locations/`.

### Create your own training and test sets

- To create the training dataset from our public e-bike dataset of scooter locations in the USA, run:

`python -m location_utils.create_training_data`

- To create a test dataset for a new location, run:

`python -m location_utils.create_test_dataset --lat 35.652832 --lon 139.839478 --name Tokyo`


### FastCUT Training and Test

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

- Train the FastCUT model:
 ```bash
python train.py --dataroot ./datasets/ebike_locations --name locations_FastCUT --CUT_mode FastCUT --n_epochs 50
```
The checkpoints will be stored at `./checkpoints/locations_FastCUT/web`.

- Test the FastCUT model:
```bash
python test.py --dataroot ./datasets/ebike_locations --name locations_FastCUT --CUT_mode FastCUT --num_test 500 --phase test --preprocess scale_width --load_size 256
```

The test results will be saved to a html file here: `./results/locations_FastCUT/latest_train/index.html`.

### Training using our launcher scripts

Please see `experiments/location_launcher.py` that generates the above command line arguments. The launcher scripts are useful for configuring rather complicated command-line arguments of training and testing.

Using the launcher, the command below generates the training command of CUT and FastCUT.
```bash
python -m experiments locations train 1  # FastCUT
```

To test using the launcher,
```bash
python -m experiments locations test 1   # FastCUT
```

### Or, use a pre-trained CUT model

To run a pretrained model, run the following.

```bash

# Download and unzip the pretrained models
wget https://gretel-public-website.s3.amazonaws.com/datasets/fastcut_models/pretrained_models.tar.gz
tar -zxvf pretrained_models.tar.gz

# Generate outputs. The dataset paths might need to be adjusted.
# To do this, modify the lines of experiments/pretrained_launcher.py
# [id] corresponds to the respective commands defined in pretrained_launcher.py
# 6 - FastCUT on ebike_data
python -m experiments pretrained run_test [id]
```

## Create the geo dataset
- To convert synthetic images to locations, run:

```bash
python -m location_utils.images_to_geo --image_path results/locations_FastCUT/test_latest/images/fake_B --name Tokyo.csv
```


### Citation

### [website](http://taesung.me/ContrastiveUnpairedTranslation/) |   [paper](https://arxiv.org/pdf/2007.15651)

[Contrastive Learning for Unpaired Image-to-Image Translation](http://taesung.me/ContrastiveUnpairedTranslation/)  
 [Taesung Park](https://taesung.me/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/), [Richard Zhang](https://richzhang.github.io/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)<br>
UC Berkeley and Adobe Research<br>
 In ECCV 2020
 
```
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

