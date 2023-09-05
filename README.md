# Random Forest with Tree Selection Monte Carlo Based (RF-TSMC)
![](forest.png)

<a href="https://pypi.org/project/random-forest-mc"><img src="https://img.shields.io/pypi/pyversions/random-forest-mc" alt="Python versions"></a>
<a href="https://pypi.org/project/random-forest-mc"><img src="https://img.shields.io/pypi/v/random-forest-mc?color=blue" alt="PyPI version"></a>
![](https://img.shields.io/badge/Coverage-100%25-green)
![](https://img.shields.io/badge/Status-Stable-green)
![](https://img.shields.io/badge/Dev--status-Released-green)
[![](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![](https://img.shields.io/badge/CodeQL-0%20alerts-green)
This project is about use Random Forest approach for *multiclass classification* using a dynamic tree selection Monte Carlo based. The first implementation is found in [2] (using Common Lisp).

## Description

This version of Random Forest makes the selection of trees based on indirect optimization via Monte Carlo simulations. Highly robust against unbalance between classes and missing data (for training and prediction). Optimized for training in parallel processing. Merge native between separate trained models, with simple merge or optimized; thus new models can be mixed with old models generating mixed models (mixing of decision trees), reducing possible new data vieses.
This model is highly recommended for EDAs, because it offers a high degree of explainability for PoCs, as it has an excellent relationship between generalization and cost of training and maintenance.
Already validated in production, with real data, for churn prediction, with data in the form of time series. In addition, it is excellent for studying Python, because the model is 100% Python code, using some of the most pytonic features! Inspect the code and enjoy!

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

-------
### Example of a full cycle using `titanic.csv`:

```python
import numpy as np
import pandas as pd

from random_forest_mc.model import RandomForestMC
from random_forest_mc.utils import LoadDicts, load_file_json, dump_file_json

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

# Simply predictions:

# One row
row = dataset.loc[0]
cls.predict(row)
{'0': 0.75, '1': 0.25}

# Multiple rows (dataset)
cls.predict(dataset.sample(n=10))
['0', '1', ...]

# Get the probabilities:
cls.predict_proba(dataset.sample(n=10))
[
    {'0': 0.75, '1': 0.25},
    {'0': 1.0, '1': 0.0},
    ...
    {'0': 0.625, '1': 0.375}
]
```
-------
### Works with missing values:

```python

cols = list(dataset.columns)
cols.pop(cols.index('Class'))
ds = dataset[cols[:10]+['Class']]

row = ds.loc[0]
cls.predict(row)
{'0': 0.75, '1': 0.25}

cls.predict(ds.sample(n=10))
['0', '1', ...]
```
-------
### Save and load model:

```python
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

```
-------
### Feature counting and importance:

```python

# Feature counting (how much features in each tree):
cls.featCount() 
(
    (3.5, 0.5, 3, 4),  # (mean, std, min, max)
    [3, 4, 3, 4, 3, 4] # List of counting of features in each tree.
)

# Feature counting considering a given sample (using only the trees that predicted correctly):
row = dataset.loc[0]
cls.sampleClassFeatCount(row, row[target_col])
(
    (3.5, 0.5, 3, 4),  # (mean, std, min, max)
    [3, 4, 3, 4, 3, 4] # List of counting of features in each tree.
)
# The follow methods have the same for a given sample in the format `cls.sampleClass...(row, row[target_col])`.


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
feat 3       0.130000   0.230000   0.398438
```
-------
### For merge different models (forests):

```python
...
cls.fit()
cls2.fit()

# Simply add all trees from cls2 in cls.
cls.mergeForest(cls2)

# Merge all trees from both models and keep the trees with scores within the top N survived scores.
cls.mergeForest(cls2, N, 'score')

# Merge all trees from both models and keep N random trees.
cls.mergeForest(cls2, N, 'random')

```
-------
### Predicting missing values:

Data with missing data (`nan` values):  
`dataset_missing_values = `
|    |   Pclass | Sex   |   Age |   SibSp |   Fare | Embarked   |   Survived |
|---:|---------:|:------|------:|--------:|-------:|:-----------|-----------:|
|  0 |        1 | `nan`   |    45 |       0 |     28 | S          |          0 |
|  1 |        1 | male  |    34 |       0 |     26 | `nan`        |          1 |

Dictionary of features and possible values:

`dict_values = `
```json

{
    "Sex": ["male", "female"],
    "Embarked": ["S", "C", "Q"]
}

```

`cls.predictMissingValues(dataset_missing_values, dict_values)` returns:

|    |   Pclass | Sex    |   Age |   SibSp |   Fare | Embarked   |   Survived |        0 |        1 |   row_id |
|---:|---------:|:-------|------:|--------:|-------:|:-----------|-----------:|---------:|---------:|---------:|
|  0 |        1 | `nan`    |    45 |       0 |     28 | S          |          0 | `nan`      | `nan`      |        0 |
|  1 |        1 | male   |    45 |       0 |     28 | S          |          0 |   0.5625 |   0.4375 |        0 |
|  2 |        1 | female |    45 |       0 |     28 | S          |          0 |   0.0625 |   0.9375 |        0 |
|  3 |        1 | male   |    34 |       0 |     26 | `nan`        |          1 | `nan`      | `nan`      |        1 |
|  4 |        1 | male   |    34 |       0 |     26 | S          |          1 |   0.5625 |   0.4375 |        1 |
|  5 |        1 | male   |    34 |       0 |     26 | C          |          1 |   0.5625 |   0.4375 |        1 |
|  6 |        1 | male   |    34 |       0 |     26 | Q          |          1 |   0.5625 |   0.4375 |        1 |


You can see three new columns: in this case, you see one column (with the probabilities) for each target value (so if you have `n` different target values, you'll have `n` columns), and the lat one named `row_id` linking all predicted ones to the first row (the original row with the missing value).
Obs: let the target column is optionally.

-------
### Notes:

- Classes values must be converted to `str` before make predicts.
- `fit` always add new trees (keep the trees generated before).
-------
### LoadDicts:

LoadDicts works loading all `JSON` files inside a given path, creating an object helper to use this files as dictionaries.

For example:
```python
from random_forest_mc.utils import LoadDicts
# JSONs: path/data.json, path/metdada.json
dicts = LoadDicts("path/")
# you have: dicts.data and dicts.metdada as dictionaries
# And a list of dictionaries loaded in:
dicts.List
> ["data", "metdada"]
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

### TO-DO list.

For TO-DO list see `TODO.md`.
