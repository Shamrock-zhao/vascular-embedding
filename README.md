# Simultaneous segmentation and characterization of the retinal vasculature using U-Nets

This code is inspired in the paper by Luca Giancardo, Kirk Roberts and Zhongming Zhao, "Representation Learning for Retinal Vasculature Embedding".
I have introduced several modifications to the original contribution of the authors, but the general idea remains there.

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

> **OpenCV in macOS X** I found difficult to find a nice post describing how to install openCV in python 3.6 using Anaconda. Use the command ```conda install -y -c conda-forge opencv```.

> **pydensecrf** To install [pydensecrf](https://github.com/lucasb-eyer/pydensecrf) in Anaconda, clone this repository and run: ```python3 setup.py install```.

## Installation

0. Create a project folder with any name that you want. E.g:

    > ```mkdir my_vascular_embedding_project```

1. Clone the repository by doing:
    
    > ```git clone https://github.com/ignaciorlando/encoder-based-regularization.git```

2. Move yourself in the terminal to ```./vascular-embedding```:

    > ```cd encoder-based-regularization```

