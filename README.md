# Encoder based regularization

## Requirements

This code requires to pre-install certain libraries:

* python 3.6
* scipy 0.19.1
* scikit-image 0.13.0
* visdom

We tested our system in these combinations of architectures and operating systems:
- Ubuntu 14.04, 64-bits.

## Installation

1. Clone the repository by doing:
    
    > ```git clone https://github.com/ignaciorlando/encoder-based-regularization.git```

## Data preparation

Use the code in ```./data_preparation``` to download and organize all the data sets, extract patches from all of them and organize them to run the experiments.
Each of the relevant scripts is described in the sequel:

### ```organize_datasets``` 

Run this script to prepare all the data sets (DRIVE, STARE, CHASEDB1 and HRF) in their traditional splits into training, validation and test, and to generate FOV masks for CHASEDB1 and HRF data sets. STARE FOV masks were downloaded [from this website](http://www.uhu.es/retinopathy/muestras2.php).

The organized data will be automatically saved in ```./data```. 

If you want to prepare a single database, you can use the functions in ```data_preparation```.

> **Note** You need to download the ```DRIVE.zip``` file manually and paste it in ```./tmp```, as it requires to register [in this webpage](https://www.isi.uu.nl/Research/Databases/DRIVE/download.php). All the other databases will be downloaded automatically from their original sources.


### ```extract_patches_from_datasets```

Run this script to extract patches from all the data sets (DRIVE, STARE, CHASEDB1 and HRF). This patches are extracted from ```training``` and ```validation``` folders, so make sure to run ```organize_datasets``` first.

For each of the databases, 6 folders will be automatically created, one for each combination of image preprocessing (original RGB and RGB after [contrast equalization](https://arxiv.org/abs/1706.03008)) and sampling strategy (uniform sampling and guided by labels).

**Script parameters:**
- ```patch_size```: length (in pixels) of the squared patches (default: 64 pixels).
- ```num_patches```: number of patches to extract for each of the images (default: 1000 pixels).


### ```initialize_data_for_experiments```

Run this script to prepare data for the experiments in the paper.