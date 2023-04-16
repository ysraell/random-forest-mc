# v1.0.4

    1) fix: `survived_score` getting wrong value. When the Tree is not dropped, the value got is `survived_score` and not th_val. Is not crictical, because generally the trees require many trees, so the score is got correctly.
    2) Add qaulity checks in `utils.LoadDicts`.
    3) Add input parameter `ignore_errors` in `utils.LoadDicts`.
    4) Add iterable protocol in `utils.LoadDicts`.

# v1.0.3

    1) Add the parameter `max_depth` to limit how deep the branching will be.
    2) Add the `min_samples_split`, once get this minimun or less, the leaf is generated instead a new branching.
    3) Fix the `__let__` that was with `>` instead `<`.
    4) Remove the threading mode for fit in paralell processing.

# v1.0.2

    1) Replace sum with fsum in many parts., from math standard lib. See: https://docs.python.org/3/library/math.html. 
