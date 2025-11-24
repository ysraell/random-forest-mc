# 🚀 Supercharging Random Forest with NumPy! ⚡

I recently refactored the `random-forest-mc` library to replace Pandas with NumPy for the core tree-building logic. The results are incredible!

Here is a benchmark comparing the new version against v1.3.0:

| Dataset | Trees | 1.3.0 (s) | 1.4.0 (s) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| creditcard | 8 | 2.1448 | 1.1488 | **1.87x** |
| creditcard | 16 | 3.9049 | 2.1223 | **1.84x** |
| creditcard | 32 | 10.5083 | 4.2060 | **2.50x** |
| creditcard | 64 | 36.7896 | 8.3063 | **4.43x** |
| creditcard | 128 | 118.8016 | 15.8434 | **7.50x** |
| creditcard_trans_float | 8 | 1.8764 | 1.2724 | **1.47x** |
| creditcard_trans_float | 16 | 4.0079 | 2.0963 | **1.91x** |
| creditcard_trans_float | 32 | 10.6948 | 4.1734 | **2.56x** |
| creditcard_trans_float | 64 | 33.0080 | 8.4900 | **3.89x** |
| creditcard_trans_float | 128 | 130.8448 | 16.9229 | **7.73x** |
| creditcard_trans_int | 8 | 1.9897 | 1.1274 | **1.76x** |
| creditcard_trans_int | 16 | 3.5978 | 2.5863 | **1.39x** |
| creditcard_trans_int | 32 | 11.9480 | 3.9811 | **3.00x** |
| creditcard_trans_int | 64 | 31.0352 | 8.2566 | **3.76x** |
| creditcard_trans_int | 128 | 144.7600 | 16.3357 | **8.86x** |
| titanic | 8 | 1.5529 | 0.0635 | **24.46x** |
| titanic | 16 | 5.0757 | 0.1439 | **35.28x** |
| titanic | 32 | 22.1289 | 0.4631 | **47.79x** |

Check out the project on GitHub: https://github.com/ysraell/random-forest-mc

#Python #DataScience #MachineLearning #NumPy #Performance #OpenSource
