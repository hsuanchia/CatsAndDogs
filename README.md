# Cats And Dogs classification
# Dataset
* [Kaggle public dataset](https://www.kaggle.com/competitions/dogs-vs-cats/data)
* test1 dataset has no corresponding label, so I labeling the test1 dataset by myself, labeling result saved in test1_label.csv
# Run codes
# Codes
```text
CatsAndDogs/
├── dogs-vs-cats/           # Folder containing dataset
|
├── models/                 # Folder save trained models
|
├── data_preprocess.py      # Data preprocessing
|
├── image_labeler_gui.py    # Image labeling tools
|
├── model.py                # Model architecture definition
|
├── train.py                # Model training procedure
|
└── README.md
```
# Demo
* Result  
<img src="./figures/SimpleCNN_30ep_result.png" width="150">

* Confusion matrix 
<img src="./figures/SimpleCNN_30ep_cnf_mat.png" width="200">

* ROC curve 
<img src="./figures/SimpleCNN_30ep_roc_curve.png" width="400">

* Labeling tools 
<img src="./figures/Labeling_tools.png" width="400">

# Package version
```bash
python==3.8
torch==1.13.1+cu116
torchversion==0.14.1+cu116
torchinfo==1.8.0
tqdm==4.63.1
rich==13.5.2
```