# Credit Card Fraud Detection Project
![](forest.png)

[![Python 3.7](https://img.shields.io/badge/Python-3.7-gree.svg)](https://www.python.org/downloads/release/python-370/)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-gree.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-gree.svg)](https://www.python.org/downloads/release/python-390/)


This project is about use Random Forest approach to credit card fraud detection [3]. Based on [1], random forest approach have interesting results in this issue. Moreover, random forest approach offers opportunities for customization to achieve better results for issues with particularities. In this project, we have a customized method of Random Forest using a dynamic tree selection Monte Carlo based. The first implementation is found in [2] (using Common Lisp).

### Structure

- **EDA**: Using [Pandas Profiling](https://pandas-profiling.github.io/pandas-profiling/docs/master/index.html) for take a look into the data. [Notebook](./EDA.ipynb).

- **Feat. Analysis**: Transforming the data into representations for minor time and space complexity. [Notebook](./Feat_Eng.ipynb). 

- **Baseline Prediction Model**: Experiments using [Random Forest from Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). [Notebook](./Baseline_Model.ipynb).

- **Customized Prediction Model**: Random forest custimized with a dynamic tree selection Monte Carlo based. [Notebook](./Customized_Model.ipynb).

# References

[1] [Le Borgne, Y.A., & Bontempi, G. (2021). Machine Learning for Credit Card Fraud Detection - Practical Handbook. Université Libre de Bruxelles](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html).

[2] [Laboratory of Decision Tree and Random Forest (`github/ysraell/random-forest-lab`)](https://github.com/ysraell/random-forest-lab). GitHub repository.

[3] Credit Card Fraud Detection. Anonymized credit card transactions labeled as fraudulent or genuine. Kaggle. Access: <https://www.kaggle.com/mlg-ulb/creditcardfraud>.

### Notes

- Python requirements in `requirements.txt`. Better for Python `>=3.7`. Run the follow command inside this repository:

```bash
$ pip3 install -r requirements.txt --no-cache-dir
```

### Development Framework (optional)

- [My data science Docker image](https://github.com/ysraell/my-ds).

With this image you can run all notebooks and scripts Python inside this repository.

### TODO:

- Feature importance analysis comparing the customized and baseline models.
