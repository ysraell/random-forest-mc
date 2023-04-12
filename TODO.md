# TODO and bugs.

### BUGS:
- Fix automated version collect from `pyproject.toml`.

### TODO Utils:

- Add methods from [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html) for comparison.

### TODO v1.1:

- Add [`memray`](https://github.com/bloomberg/memray) in the automated test process. 
- Fix multithread logic.
- Mssing data issue:
    - Data Imputation using the Forest (with and without true label).
    - Prediction with missing values, approaches:
        - *A*) Use imputation data before prediction. Different from *A*, choose the value with the higher probability.
        - *B*) (User) Set a default value for each feature a priori. When facing a missing value, use the given default value.
- Create new forests from a cross merging between other forests, for a given amount of trees for the output forest:~
    - by optimization, based on a GA and MC approaches, using a given test dataset;
    - Design as a subclass of the `RandomForestMC` for optimization approaches and a function for randomness and sorted merging.
- Create a notebook with [Memray](https://github.com/bloomberg/memray) applied to the model with different datasets.
- Discover how automate the unit tests for test with each compatible minor version.
- Add parameter for limit depth, min sample to leaf, min feature completed cycles.
- Add parameter for active validation score (like loss) for each set of a given number of trees generated.
- Add a set of functions for generate perfomance metrics: like trees generation/validation time.
- Review the use of the threshold (TH) for validation process. How we could set the dynamic portion? How spread the TH? The set of rows, for a set of rounds, not reaching the TH, drop and get next? Dynamicaly decreasing the TH for each N sets of rows without sucess?
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
