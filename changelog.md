### v1.2.0:

- Fix bug with function `drop_duplicated_trees` that is keeping the duplicated only.
- Add in the `dict2model` the possibility to add and not only replace the current Forest. E.g.: dict2model(Dict, add = False) -> Default: False.
- Remove `split_train_val_replace`.
- Use the trees in parallel processing.
- Docstring.  
- Add parameter for active validation score (like loss) for each set of a given number of trees generated.

# v1.1.2

    1) Add the different Python versions in the automated tests.

# v1.1.1

    1) Add `NaN`, `None`, `Null` cheking before use and generate the Tree.
    2) Remove `self.dataset = dataset.dropna()` (l. 509, `model.py`).

# v1.1.0

    1) Fix utils.

# v1.1.0

    1) fix: `survived_score` getting wrong value. When the Tree is not dropped, the value got is `survived_score` and not th_val. Is not crictical, because generally the trees require many trees, so the score is got correctly.
    2) Add quality checks in `utils.LoadDicts`.
    3) Add input parameter `ignore_errors` in `utils.LoadDicts`.
    4) Add iterable protocol in `utils.LoadDicts`.
    5) Refactor some type hints.
    6) Add missing values prediction feature.

# v1.0.3

    1) Add the parameter `max_depth` to limit how deep the branching will be.
    2) Add the `min_samples_split`, once get this minimun or less, the leaf is generated instead a new branching.
    3) Fix the `__let__` that was with `>` instead `<`.
    4) Remove the threading mode for fit in paralell processing.

# v1.0.2

    1) Replace sum with fsum in many parts., from math standard lib. See: https://docs.python.org/3/library/math.html. 
