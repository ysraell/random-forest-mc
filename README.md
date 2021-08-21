# Random Forest with Dynamic Tree Selection Monte Carlo Based (RF-TSMC)
![](forest.png)

[![Python 3.7](https://img.shields.io/badge/Python-3.7-gree.svg)](https://www.python.org/downloads/release/python-370/)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-gree.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-gree.svg)](https://www.python.org/downloads/release/python-390/)

This project is about use Random Forest approach for *multiclass classification* using a dynamic tree selection Monte Carlo based. The first implementation is found in [2] (using Common Lisp).

## Project status: `Pre-Alpha`.

## Fundamentals:

- Based on Random Forest method principles: ensemble of models (decision trees).

- In bootstrap process:

    - the data sampled ensure the balance between classes, for training and validation;

    - the list of features used are randomly sampled (with random number of features and order).

- For each tree:

    - fallowing the sequence of a given list of features, the data is splited half/half based on meadian value;

    - the splitting process ends when the samples have one only class;

    - validation process based on dynamic threshold can discard the tree.

- For use the forest:

    - all trees predictions are combined as a vote;

    - it is possible to use soft or hard-voting.

- Positive side-effects:

    - possible more generalization caused by the combination of overfitted trees, each tree is highly specialized in a smallest and different set of feature;

    - robustness for unbalanced and missing data, in case of missing data, the feature could be skipped without degrade the optimization process;

    - in prediction process, a missing value could be dealt with a tree replication considering the two possible paths;

    - the survived trees have a potential information about feature importance.

### References

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

### TODO v0.1:

- Implement pre-commit.
- Implement automated tests using the fallow datasets:
    - Titanic;
    - Credit Card Fraud;
    - Iris;
    - artificial with random data;
    - and artificial with perfect classification.

### TODO v0.2:

- Add parallel processing using or TQDM or csv2es style.
- Prediction with missing values: `useTree` must be functional and branching when missing value, combining classes at leaves with their probabilities.
- [Plus] Add a method to return the list of feaures and their degrees of importance.
