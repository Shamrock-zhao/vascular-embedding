# Encoder based regularization

## Requirements

python 3.6
scipy 0.19.1
scikit-image 0.13.0



## Installation



## Data preparation

```organize_datasets``` 

Run this script to prepare all the data sets (DRIVE, STARE, CHASEDB1 and HRF) in their traditional splits into training, validation and test, and to generate FOV masks for CHASEDB1 and HRF data sets. STARE FOV masks were downloaded [from this website](http://www.uhu.es/retinopathy/muestras2.php).

The organized data will be automatically saved in ```./data```. 

If you want to prepare a single database, you can use the functions in ```data_preparation```.

> **Note** You need to download the ```DRIVE.zip``` file manually and paste it in ```./tmp```, as it requires to register [in this webpage](https://www.isi.uu.nl/Research/Databases/DRIVE/download.php). All the other databases will be downloaded automatically from their original sources.


```prepare_data_for_experiments```