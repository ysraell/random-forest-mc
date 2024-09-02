# TODO and bugs.

### BUGS:
- Review warning `SettingWithCopyWarning`in model.py:494 `dataset[self.target_col] = dataset[self.target_col].astype(str)`
- Review warning `DeprecationWarning: This process (pid=000000) is multi-threaded, use of fork() may lead to deadlocks in the child. pid = os.fork()`, in Python 3.12.4-5.
- Fix automated version collect from `pyproject.toml`.

### TODO Utils:

- Add methods from [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html) for comparison.

### TODO v1.1.2:

~~- Add the different Python versions in the automated tests.~~

### TODO v1.2:

- Add [`memray`](https://github.com/bloomberg/memray) in the automated test process. 
- Fix multithread logic.
- Create new forests from a cross merging between other forests, for a given amount of trees for the output forest:~
    - by optimization, based on a GA and MC approaches, using a given test dataset;
    - Design as a subclass of the `RandomForestMC` for optimization approaches and a function for randomness and sorted merging.
- Create a notebook with [Memray](https://github.com/bloomberg/memray) applied to the model with different datasets.
- Discover how automate the unit tests for test with each compatible minor version.
- Add parameter for active validation score (like loss) for each set of a given number of trees generated.
- Add a set of functions for generate perfomance metrics: like trees generation/validation time.
- Docstring.

### TODO V2.0:

- Extender for predict by regression.
- Refactor to use NumPy or built in Python features as core data operations.
- Tree management framework: to remove or add new trees, version management for set of trees.

### TODO Extras:
- Automated tests using GitHub Actions.
- Develop a Scikit-Learn full compatible model. Ref.: <https://scikit-learn.org/stable/developers/develop.html>.
- Write and publish a article.
- Read the Scikit-learn governance and decision-making (https://scikit-learn.org/stable/governance.html#governance).
