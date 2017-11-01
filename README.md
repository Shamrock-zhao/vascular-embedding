# Encoder based regularization

## Requirements

python 3.6


## Installation



## Data preparation

In this section you will find a brief description of the main scripts for organizing your data.

***

```setup_data.py``` 

Run this script to prepare all the data sets (DRIVE, STARE, CHASEDB1 and HRF) in their traditional splits into training, validation and test. 

The organized data will be automatically saved in ```./data```. 

If you want to prepare a single database, you can use the functions in ```data_preparation```.

> **Note** You need to download the ```DRIVE.zip``` file manually and paste it in ```./tmp```, as it requires to register [in this webpage](https://www.isi.uu.nl/Research/Databases/DRIVE/download.php). All the other databases will be downloaded automatically from their original sources.

***