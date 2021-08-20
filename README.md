# Random Forest with Dyanmic Tree Selection Monte Carlo Based (RF-TSMC)
![](forest.png)

[![Python 3.7](https://img.shields.io/badge/Python-3.7-gree.svg)](https://www.python.org/downloads/release/python-370/)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-gree.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-gree.svg)](https://www.python.org/downloads/release/python-390/)


This project is about use Random Forest approach using a dynamic tree selection Monte Carlo based. The first implementation is found in [2] (using Common Lisp).

# References

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

- Implement the code.
    - [Plus] Add a method to return the list of feaures and their degrees of importance.
- Set Poetry and publish to PyPI.
- Add parallel processing using or TQDM or csv2es style.
