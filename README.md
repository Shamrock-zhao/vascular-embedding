# Encoder based regularization

## Requirements

This code requires to pre-install certain libraries:

* python 3.6
* scipy 0.19.1
* scikit-image 0.13.0
* visdom
* cv2
* pydensecrf

We tested our system in these combinations of architectures and operating systems:
- Ubuntu 14.04, 64-bits.
- macOS X High Sierra, 64 bits.

> **OpenCV in macOS X** We found difficult to find a nice post describing how to install openCV in python 3.6 using Anaconda. Use the command ```conda install -y -c conda-forge opencv```.

> **pydensecrf** To install [pydensecrf](https://github.com/lucasb-eyer/pydensecrf) in Anaconda, clone this repository and run: ```python3 setup.py install```.

## Installation

1. Clone the repository by doing:
    
    > ```git clone https://github.com/ignaciorlando/encoder-based-regularization.git```

2. Move yourself in the terminal to ```./encoder-based-regularization```:

    > ```cd encoder-based-regularization```

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

For each of the databases, 6 folders will be automatically created, one for each combination of image preprocessing (original RGB, RGB after [contrast equalization](https://arxiv.org/abs/1706.03008) and RGB after CLAHE enhancement) and sampling strategy (uniform sampling and guided by labels).

*Script parameters:*
- ```--patch_size```: length (in pixels) of the squared patches (default: 64 pixels).
- ```--num_patches```: number of patches to extract for each of the images (default: 200.000 pixels).


### ```initialize_data_for_experiments```

Run this script to organize the patches for running the experiments. Be aware that the parameters are optional.

If you want to monitor the evolution of the training/validation losses and the Dice coefficient on the validation set, make sure to run on a separate terminal the following command:

```python3 -m visdom.server```

In your browser, go to [http://localhost:8097/](http://localhost:8097/) and in the visdom interface select the experiment that you are running.

*Script parameters:*
- ```--data_path```: path where all the data sets are saved (default: ./data).
- ```--experiment```: a number indicating the type of experiment to prepare data for. (0) All the experiments. (1) All patches together.



## Training and evaluation of a U-Net

If you want to train and/or evaluate a U-Net for blood vessel segmentation in fundus images, you need to run the following scripts:

### ```train```

This script trains a U-Net based on an input configuration file. After each epoch, a checkpoint file is saved on a ```checkpoint``` folder in the output path. The final model, namely ```model.pkl```, is saved in the same folder.

*Script parameters:*
- ```config_file```: full path to a configuration file with the experimental setup. Use [this example file](https://github.com/ignaciorlando/encoder-based-regularization/blob/master/experiments/example.ini) to guide yourself about how to configure a experiment.
- ```--load```: a boolean value indicating if the model has to be loaded from a previous checkpoint (default: ```False```).

### ```predict```

Use this script to evaluate a pre-trained model on a given data set.

*Script parameters:*
- ```image_path```: path to the images.
- ```fov_path```: path to the FOV masks.
- ```output_path```: path were the results will be saved.
- ```model_filename```: full path to a .pkl file with a trained model. Notice that checkpoints have only weights.
- ```--image_preprocessing```: type of image preprocessing. The current available options are ```rgb``` (default), ```eq``` (RGB equalized) and ```clahe``` (RGB after contrast enhancement using CLAHE).
