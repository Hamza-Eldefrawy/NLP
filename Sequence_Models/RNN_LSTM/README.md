# Project Overview
This project focuses on classifying text queries into nine distinct toxicity categories using PyTorch. It involves data inspection, data cleaning, and the training of baseline sequence-classification models (RNN and LSTM) to establish initial performance benchmarks.

# Dataset Analysis

## Dataset Overview
The original dataset contains 3,000 samples and 3 columns:
* `query` (2,009 unique values)
* `image descriptions` (12 unique values)
* `Toxic Category` (9 classes)

The target variable contains 9 classes: Safe, Violent Crimes, Elections, Sex-Related Crimes, unsafe, Non-Violent Crimes, Child Sexual Exploitation, Unknown S-Type, and Suicide & Self-Harm.

## Data Quality and Inspection
No missing or empty values were found in the dataset. However, an in-depth data inspection revealed significant duplication:
* There are 973 exact duplicate rows.
* There are 991 duplicated queries.

## Data Cleaning and Preparation
To prepare the dataset for accurate modeling, the whitespace in the `query` column was normalized. The duplicate rows were then dropped. During this preparation step, the `image descriptions` column was removed, leaving only the `query` and `Toxic Category` columns.

After cleaning, the dataset shape was reduced from 3,000 rows to 2,009 rows.

## Class Distribution (Before and After Cleaning)
The dataset is highly imbalanced. The majority of the data belongs to the `Safe` and `Violent Crimes` categories. Notably, the data inspection and cleaning phase revealed that removing duplicates drastically reduces the sample sizes of the minority classes, with several dropping to fewer than 5 samples. 

| Class | Original Samples | Cleaned Samples |
| :--- | :--- | :--- |
| Safe | 995 | 879 |
| Violent Crimes | 792 | 691 |
| Non-Violent Crimes | 301 | 205 |
| unsafe | 274 | 141 |
| Unknown S-Type | 196 | 84 |
| Sex-Related Crimes | 115 | 2 |
| Suicide & Self-Harm | 114 | 3 |
| Elections | 110 | 2 |
| Child Sexual Exploitation | 103 | 2 |

# Baseline Models
Two simple sequence-classification models were trained on the raw dataset: an RNN and an LSTM. Both models utilize an architecture consisting of an Embedding layer, the respective recurrent layer (RNN or LSTM), and a Linear head. The models were evaluated using overall accuracy and per-class precision, recall, and F1 scores.

## LSTM Results
* **Overall Accuracy:** 49.3%

| Class | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| Child Sexual Exploitation | 1.000 | 1.000 | 1.000 |
| Elections | 1.000 | 1.000 | 1.000 |
| Non-Violent Crimes | 0.000 | 0.000 | 0.000 |
| Safe | 0.395 | 0.986 | 0.564 |
| Sex-Related Crimes | 0.917 | 1.000 | 0.957 |
| Suicide & Self-Harm | 1.000 | 0.950 | 0.974 |
| Unknown S-Type | 1.000 | 0.516 | 0.681 |
| Violent Crimes | 0.000 | 0.000 | 0.000 |
| unsafe | 0.000 | 0.000 | 0.000 |

## RNN Results
* **Overall Accuracy:** 53.1%

| Class | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| Child Sexual Exploitation | 0.923 | 1.000 | 0.960 |
| Elections | 1.000 | 1.000 | 1.000 |
| Non-Violent Crimes | 0.000 | 0.000 | 0.000 |
| Safe | 0.413 | 1.000 | 0.585 |
| Sex-Related Crimes | 1.000 | 1.000 | 1.000 |
| Suicide & Self-Harm | 1.000 | 0.950 | 0.974 |
| Unknown S-Type | 1.000 | 0.516 | 0.681 |
| Violent Crimes | 0.000 | 0.000 | 0.000 |
| unsafe | 1.000 | 0.341 | 0.508 |

# Initial Analysis
Both baseline models achieved relatively low overall accuracy:
* **RNN:** 53.1%
* **LSTM:** 49.3%

The per-class results indicate that the models perform very poorly on some classes, particularly `Non-Violent Crimes`, `Violent Crimes`, and `unsafe`. 

The primary issue appears to be the strong class imbalance within the dataset. The majority classes (`Safe` and `Violent Crimes`) contain substantially more samples than the minority classes. Furthermore, the original dataset contains only 2,009 unique queries out of 3,000 samples, and the `image descriptions` column contains only 12 unique values. The data cleaning process highlighted that the artificially inflated minority classes collapse once exact duplicates are dropped, limiting the models' ability to accurately learn those categories.

# Next Steps
As a quick baseline experiment to address the extreme sparsity, the minority categories can be grouped into a single **`Other`** category. This approach reduces the number of classes and may provide a more balanced classification problem:

`Child Sexual Exploitation`, `Elections`, `Non-Violent Crimes`, `Safe`, `Sex-Related Crimes`, `Suicide & Self-Harm`, `Unknown S-Type`, `Violent Crimes`, `unsafe` → **Major classes + Other**

The resulting merged dataset can then be used to retrain the RNN and LSTM and compare the new performance against the original 9-class models.

> **Note:** Grouping classes into `Other` serves as a useful quick experiment but should not be considered a definitive solution to class imbalance. Future experiments should also consider techniques such as class-weighted loss, oversampling, or collecting additional unique samples for the minority classes.