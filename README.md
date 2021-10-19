# Random Forest with Dynamic Tree Selection Monte Carlo Based (RF-TSMC)
![](forest.png)

<a href="https://pypi.org/project/random-forest-mc"><img src="https://img.shields.io/pypi/pyversions/random-forest-mc" alt="Python versions"></a>
<a href="https://pypi.org/project/random-forest-mc"><img src="https://img.shields.io/pypi/v/random-forest-mc?color=blue" alt="PyPI version"></a>
![](https://img.shields.io/badge/Coverage-100%25-green)
![](https://img.shields.io/badge/Status-Stable-green)
![](https://img.shields.io/badge/Dev--status-WIP-orange)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/ysraell/random-forest-mc.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ysraell/random-forest-mc/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ysraell/random-forest-mc.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ysraell/random-forest-mc/context:python)

This project is about use Random Forest approach for *multiclass classification* using a dynamic tree selection Monte Carlo based. The first implementation is found in [2] (using Common Lisp).

## Install:

Install using `pip`:

```bash
$ pip3 install random-forest-mc
```

Install from this repo:

```bash
$ git clone https://github.com/ysraell/random-forest-mc.git
$ cd random-forest-mc
$ pip3 install .
```

## Usage:

Example of a full cycle using `titanic.csv`:

```python
import numpy as np
import pandas as pd

from random_forest_mc.model import RandomForestMC
from random_forest_mc.utils import LoadDicts

dicts = LoadDicts("tests/")
dataset_dict = dicts.datasets_metadata
ds_name = "titanic"
params = dataset_dict[ds_name]
target_col = params["target_col"]
dataset = (
    pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
    .dropna()
    .reset_index(drop=True)
)
dataset["Age"] = dataset["Age"].astype(np.uint8)
dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
dataset["Pclass"] = dataset["Pclass"].astype(str)
dataset["Fare"] = dataset["Fare"].astype(np.uint32)
cls = RandomForestMC(
    n_trees=8, target_col=target_col, max_discard_trees=4
)
cls.process_dataset(dataset)
cls.fit() # or with cls.fitParallel(max_workers=8)
y_test = dataset[params["target_col"]].to_list()
cls.setWeightedTrees(True) # predictions weighted by survive scores
y_pred = cls.testForest(dataset)
accuracy_hard = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
cls.setSoftVoting(True) # for predicitons using soft voting strategy
y_pred = cls.testForest(dataset)
accuracy_soft = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)

# Saving model:
ModelDict = cls.model2dict()
dump_file_json(path_dict, ModelDict)
del ModelDict

# Loading model
ModelDict = load_file_json(path_dict)
cls = RandomForestMC()
cls.dict2model(ModelDict)
# Before run fit again, load dataset. Check if the features are the same!
cls.process_dataset(dataset)

row = dataset.loc[0]
# Feature counting (how much features in each tree):
cls.featCount() # or cls.sampleClassFeatCount(row, row[target_col])
(
    (3.5, 0.5, 3, 4),  # (mean, std, min, max)
    [3, 4, 3, 4, 3, 4] # List of counting of features in each tree.
)

# Feature importance:
cls.featImportance() # or cls.sampleClassFeatImportance(row, row[target_col])
{
    'feat 1': 0.900000,
    'feat 2': 0.804688,
    'feat 3': 0.398438,
    ...
}

# Permutation feature importance:
cls.featPairImportance() # or cls.sampleClassFeatPairImportance(row, row[target_col])
{
    ('feat 1', 'feat 2'): 0.12,
    ('feat 1', 'feat 3'): 0.13,
    ('feat 2', 'feat 3'): 0.23,
    ...
}

# Permutation feature importance in matrix (dataframe):
cls.featCorrDataFrame() # or cls.sampleClassFeatCorrDataFrame(row, row[target_col])
               feat 1     feat 2     feat 3
feat 1       0.900000   0.120000   0.130000
feat 2       0.120000   0.804688   0.230000
feat 3       0.130000   0.000000   0.398438
```

### Notes:

- Classes values must be converted to `str` before make predicts.
- `fit` always add new trees (keep the trees generated before).

### LoadDicts:

LoadDicts works loading all `JSON` files inside a given path, creating an object helper to use this files as dictionaries.

For example:
```python
>>> from random_forest_mc.utils import LoadDicts
>>> # JSONs: path/data.json, path/metdada.json
>>> dicts = LoadDicts("path/")
>>> # you have: dicts.data and dicts.metdada as dictionaries
>>> # And a list of dictionaries loaded in:
>>> dicts.List
["data", "metdada"]
```

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

    - Robust for mssing values in categorical features during prediction process.

### References

[2] [Laboratory of Decision Tree and Random Forest (`github/ysraell/random-forest-lab`)](https://github.com/ysraell/random-forest-lab). GitHub repository.

[3] Credit Card Fraud Detection. Anonymized credit card transactions labeled as fraudulent or genuine. Kaggle. Access: <https://www.kaggle.com/mlg-ulb/creditcardfraud>.

### Development Framework (optional)

- [My data science Docker image](https://github.com/ysraell/my-ds).

With this image you can run all notebooks and scripts Python inside this repository.

### BUGS:
- Fix automated version collect from `pyproject.toml`.

### TODO v1.0:

- Mssing data issue:
    - Data Imputation using the Forest (with and without true label).
    - Prediction with missing values, approaches:
        - *A*) only for numeric feature, `useTree` must be functional and branching when missing value, combining classes at leaves with their probabilities (duplicate the tree in each node with missing value)), for categorical features, use the default value branching already implemented and working well.
        - *B*) Use imputation data before prediction.
- Add parameter for limit depth, min sample to leaf, min feature completed cycles.
- Add parameter for active validation score (like loss) for each set of a given number of trees generated.
- Docstring.

### TODO v1.1:

- Automated tests using GitHub Actions.

### TODO V2.0:

- Extender for predict by regression.
- Refactor to use NumPy or built in Python features as core data operations.
- Tree management framework: to remove or add new trees, version management for set of trees.
