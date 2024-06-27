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

Parameters:

df_hr (DataFrame): A DataFrame containing bounding box predictions for high-resolution images.

df_lr (DataFrame): A DataFrame containing bounding box predictions for low-resolution images.

df_gt_hr (DataFrame): A DataFrame containing ground truth bounding boxes for high-resolution images.

df_gt_lr (DataFrame): A DataFrame containing ground truth bounding boxes for low-resolution images.

iou_thresholds (list, optional): A list of IoU thresholds to use for evaluation. Default is [0.5].

Returns:

ans (dictionary): The computed LI score(s) based on the IoU thresholds and class labels.

```
import pandas as pd
from cheXwhatsApp import li_score

# Sample data for high-resolution predictions
data_hr = {
    'Name': ['cat', 'dog'],
    'label': [0, 1],
    'x_min': [10, 20],
    'y_min': [30, 40],
    'x_max': [50, 60],
    'y_max': [70, 80]
}
df_hr = pd.DataFrame(data_hr)

# Sample data for low-resolution predictions
data_lr = {
    'Name': ['cat', 'dog'],
    'label': [0, 1],
    'x_min': [12, 22],
    'y_min': [32, 42],
    'x_max': [52, 62],
    'y_max': [80, 100]
}
df_lr = pd.DataFrame(data_lr)

# Sample data for ground truth high-resolution
data_gt_hr = {
    'Name': ['cat', 'dog'],
    'class': [0, 1],
    'x_min': [11, 21],
    'y_min': [31, 41],
    'x_max': [51, 61],
    'y_max': [71, 81]
}
df_gt_hr = pd.DataFrame(data_gt_hr)

# Sample data for ground truth low-resolution
data_gt_lr = {
    'Name': ['cat', 'dog'],
    'class': [0, 1],
    'x_min': [13, 23],
    'y_min': [33, 43],
    'x_max': [53, 63],
    'y_max': [73, 83]
}
df_gt_lr = pd.DataFrame(data_gt_lr)

# Compute the LI score
iou_thresholds = [0.5, 0.75]
li_score_result = li_score(df_hr, df_lr, df_gt_hr, df_gt_lr, iou_thresholds)
print(li_score_result)

```

## OLS Score Calculation
This repository contains the ols_score function, which computes the Object-Level Similarity (OLS) score for images using high-resolution and low-resolution heatmaps. The function returns the results as a DataFrame.

Parameters:

image_path (str): Path to the directory containing the original images.

heatmap_hr_dir (str): Directory containing high-resolution heatmaps.

heatmap_lr_dir (str): Directory containing low-resolution heatmaps.

device (str): Device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.

image_size (int): Size to resize the images. Default is 512.

thresholds (list): List of thresholds for score calculation. Default is [0.5].

Returns:

pd.DataFrame: DataFrame containing the OLS scores.

```
import pandas as pd
from cheXwhatsApp import ols_score

# Define the paths and parameters
image_path = 'path/to/your/images'
heatmap_hr_dir = 'path/to/high-resolution/heatmaps'
heatmap_lr_dir = 'path/to/low-resolution/heatmaps'
device = 'cpu'  # or 'cuda' if you have a GPU
image_size = 512
thresholds = [0.5, 0.75]

# Compute the OLS score
ols_score_result = ols_score(image_path, heatmap_hr_dir, heatmap_lr_dir, device, image_size, thresholds)

# Print the results
print(ols_score_result)

```





