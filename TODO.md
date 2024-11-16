# TODO and bugs.

### BUGS:
- Review warning `SettingWithCopyWarning`in model.py:494 `dataset[self.target_col] = dataset[self.target_col].astype(str)`
- Review warning `DeprecationWarning: This process (pid=000000) is multi-threaded, use of fork() may lead to deadlocks in the child. pid = os.fork()`, in Python 3.12.4-5.
- Fix automated version collect from `pyproject.toml`.

### TODO Utils:

- Add methods from [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html) for comparison.

### TODO v1.1.2:

~~- Add the different Python versions in the automated tests.~~
~~- Discover how automate the unit tests for test with each compatible midle version.~~

### TODO v1.2:

~~- Fix bug with function `drop_duplicated_trees` that is keeping the duplicated only.~~
- Add in the `dict2model` the possibility to add and not only replace the current Forest. E.g.: dict2model(Dict, add = False) -> Default: False.
- Remove `split_train_val_replace`.
- Add the possibility to duplicate rows only in train data (`ds_T`).
~~- Use the trees in parallel processing.~~
- Docstring.  
~~- Add parameter for active validation score (like loss) for each set of a given number of trees generated.~~
- Add a training process that you can use a validation dataset to compute the performance after the cretion of a set of trees (like epochs).
- Create new forests from a cross merging between other forests, for a given amount of trees for the output forest:~
    - by optimization, based on a GA and MC approaches, using a given test dataset;
    - Design as a subclass of the `RandomForestMC` for optimization approaches and a function for randomness and sorted merging.

### TODO v1.3:
- Add [`memray`](https://github.com/bloomberg/memray) in the automated test process.
- Create a notebook with [Memray](https://github.com/bloomberg/memray) applied to the model with different datasets.
- Add a set of functions for generate perfomance metrics: like trees generation/validation time.

### TODO V2.0:
- Fix multithread logic.
- Extender for predict by regression.
- Refactor to use NumPy or built in Python features as core data operations.
- Tree management framework: to remove or add new trees, version management for set of trees.

### TODO Extras:
- Automated tests using GitHub Actions.
- Develop a Scikit-Learn full compatible model. Ref.: <https://scikit-learn.org/stable/developers/develop.html>.
- Write and publish a article.
- Read the Scikit-learn governance and decision-making (https://scikit-learn.org/stable/governance.html#governance).
