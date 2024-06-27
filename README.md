## Dataset location
* Whatsapp compressed data sets are is present at :
[RSNA](https://zenodo.org/records/11632392), [NIH](https://zenodo.org/records/11569375), [Chexphoto](), [VinDr]()  
   
* Original data is present at : 
[RSNA](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data), 
[NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC), [Chexphoto](http://download.cs.stanford.edu/deep/CheXphoto-v1.0-split/CheXphoto-v1.0-split.txt), [Vindr](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data)


## Data Description
* RSNA : 
    - Train : 26684 images
    - Test  : 3000  images
* NIH : 
    - Train : 86524 images
    - Test  : 25596 images
* Datasets are created by passing the original images taken from NIH/RSNA and sending them via whatsapp using API. 

## Installation and testing models

Install with 
```
pip install cheXwhatsApp
```
## PI Score Calculation
pi_score : measures how different are predictions on original and whatsapp images.   
Create dataframes with ImageNames, Labels as index for predictions of whatsapp and original images
```
from cheXwhatsApp import pi_score
pi_score = pi_score(data_frame1, data_frame2)
```

## LI Score Calculation
This repository contains the li_score function, which calculates a metric (LI score) that evaluates the performance of bounding box predictions from high-resolution and low-resolution images against ground truth bounding boxes.



