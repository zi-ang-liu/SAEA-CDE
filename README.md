# SAEA-CDE
Source files of experiment resutls for the manusctipt that submitted to ESWA.

After the acceptance, source codes will also be uploaded in here.

## Computational results
Files "case1" and "case2" contain the raw data of results for our manuscript.

## Parameter tuning
The parameter tuning is conducted by using "Tree-structured Parzen Estimator algorithm" in Optuna.
The results are saved to file "db.sqlite3".
You can check the parameter tuning resutls.  
See https://optuna.org/.   

% pip install optuna-dashboard   
% optuna-dashboard sqlite:///db.sqlite3   

![parameter tuning reuslts (CDE)](https://github.com/zi-ang-liu/SAEA-CDE/blob/main/figures/param_tune_CDE.png "parameter tuning reuslts (CDE)")

