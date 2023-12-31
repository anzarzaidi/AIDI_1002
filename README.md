# VBL-V001
Baseline methods for the paper [Lab-scale Vibration Analysis Dataset and Baseline Methods for Machinery Fault Diagnosis with Machine Learning](https://arxiv.org/abs/2212.14732).

# Dataset
Dataset is here: [https://zenodo.org/record/7006575#.Y3W9lzPP2og](https://zenodo.org/record/7006575#.Y3W9lzPP2og).  
Locate the dataset under 'data/existing`.  
Structure of dataset:  
```bash
bagus@m049:VBL-VA001$ tree -L 2 . --filelimit 100
.
├── bearing [1000 entries exceeds filelimit, not opening dir]
├── misalignment [1000 entries exceeds filelimit, not opening dir]
├── normal [1000 entries exceeds filelimit, not opening dir]
└── unbalance [1000 entries exceeds filelimit, not opening dir]

4 directories, 4000 files
```



# Running the program (All classifiers)
```bash
# Exracted features that were provided by original authors are present in 'data/existing' folder.
# There is also a much larger feature set that is present in 'data/extracted' folder. 
# Below command will execute all classification models, models added by orignal authors along with new models added by current authors.
$ python main.py
```
# Running the program (A specific classifier)
```bash
# To execute an particular existing classifier
$ python train_svm_10fold.py
```


# Note on BPFO/BPFI

The BPFO and BPFI values are obtained from the pump bearing type datasheet, namely type NTN Bearing 6201 which has BPFO coefficient of 2.62 and BPFI coefficient of 4.38.


# Citation (Bibtex)
  ```bibtex
  @ARTICLE{Atmaja2023,  
	author = {Atmaja, Bagus Tris and Ihsannur, Haris and Suyanto and Arifianto, Dhany},  
	title = {Lab-Scale Vibration Analysis Dataset and Baseline Methods for Machinery Fault Diagnosis with Machine Learning},  
	year = {2023},  
	journal = {Journal of Vibration Engineering and Technologies},  
	doi = {10.1007/s42417-023-00959-9},  
	type = {Article},  
	publication_stage = {Article in press},  
	source = {Scopus},  
}
```
