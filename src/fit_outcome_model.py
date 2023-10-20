import os
import pickle
import yaml
import pandas as pd
import xgboost as xgb
import logging
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from skopt.space import Integer, Real
from skopt import BayesSearchCV
from functions import get_timepoint_list

def main():
    params_path = os.path.join(os.path.dirname(__file__),"..","params.yaml")
    analysis = yaml.load(open(params_path), Loader=yaml.FullLoader)["analysis"] 

    registry = analysis.get("registry","pancreas")
    setting = analysis[registry].get("setting", "palliative")
    data_dir = os.path.join(os.path.dirname(__file__),"..","data","{r}").format(r=registry)
    bootstrap = analysis.get("bootstrap", False)

    reg_data = pickle.load(open(os.path.join(data_dir,"regression_data.pkl"), "rb"))
    out_path = os.path.join(data_dir, "outcome_regression.pkl")
    log_path = os.path.join(os.path.dirname(__file__), "..", "log", f"log_{registry}.log")
    
    logging.basicConfig(filename=log_path, level=logging.INFO, filemode="a")

    logging.info("\n Fitting outcome regression model \n")

    reg_names = [
        "xgb",
        "rf",
        "linreg",
        "dummy"
    ]

    X = reg_data["predictors"]
    X_pmm = reg_data["predictors_pmm_unscaled"]
    y_pmm = reg_data["outcome_pmm"]
    
    y = reg_data["outcome"]
    z = reg_data["status"]
    
    X1 = X.loc[~y.isna()]
    y1 = y.loc[~y.isna()]

    Xtrain, Xvalidate, ytrain, yvalidate = train_test_split(X1, y1, train_size=.8, random_state=23)

    reg = get_stacking_regressor(reg_names, Xtrain, ytrain)
    
    outcome_regression, coefs = fit_outcome_model(X, X1, y1, reg, return_coefs=True)
    outcome_regression["regressed_outcome_pmm"] = pattern_mixture_estimation(X_pmm, y_pmm, registry=registry, setting=setting)

    logging.info(f"\n Coefficients for {reg_names} in stacking regressor: {outcome_regression['fitted_regressor'].final_estimator_.coef_}")

    ypred = outcome_regression["fitted_regressor"].predict(Xvalidate.drop("pat_no", errors="ignore", axis=1))
    logging.info(f"R2 score of stacked regressor: {r2_score(yvalidate, ypred)}")

    if bootstrap:
        bs_samples = pickle.load(open(os.path.join(data_dir, "bootstrap_samples.pkl"), "rb"))
        samples = bs_samples["bs_samples"]
        samples_pmm = bs_samples["bs_samples_pmm"]

        bs_outcome_reg_list = []
        
        k =  0

        for s, s_pmm in zip(samples, samples_pmm):
            print("fitting outcome regression to bootstrap sample", k+1)
            X_bs = X.loc[s]
            X_pmm_bs = X_pmm.loc[s_pmm]
            y_bs = y.loc[s]
            y_pmm_bs = y_pmm.loc[s_pmm]
            X1_bs = X_bs.loc[~y_bs.isna()]
            y1_bs = y_bs.loc[~y_bs.isna()]
            outcome_reg_bs = fit_outcome_model(X_bs, X1_bs, y1_bs, reg)
            outcome_reg_bs["regressed_outcome_pmm"] = pattern_mixture_estimation(X_pmm_bs, y_pmm_bs, registry=registry, setting=setting)
            bs_outcome_reg_list += [outcome_reg_bs]
            k+=1

        outcome_regression["bootstrap_regressions"] = bs_outcome_reg_list
   
    with open(out_path, "wb") as file:
        pickle.dump(outcome_regression, file)

    coefs.to_csv(os.path.join(data_dir,"outcome_regression_coefficients.csv"), sep=";")

def pattern_mixture_estimation(X, y, method="extrapolate", registry="pancreas", setting="palliative"):
    '''fit pattern mixture model to data and return regressed outcome.'''

    if method=="regression":
        X = X.drop("pat_no", axis=1, errors="ignore")
    X1 = X.loc[~y.isna()]
    y1 = y.loc[~y.isna()]
    M1 = X1.copy()
    M1["aval"] = y1
    M1 = M1.reset_index().set_index("obs_no")
    s = pd.Series(dtype=float)

    if method=="regression":            
        reg = LinearRegression(fit_intercept=True)
        for t in set(X1.tseqmax):
            M1_tseq = M1.query("tseqmax==@t")
            X_tseq = X.query("tseqmax==@t")[["tseq"]]
            X1_tseq = M1_tseq.drop("aval", axis=1)[["tseq"]]
            reg = reg.fit(X1_tseq, M1_tseq.aval)
            s_new = pd.Series(reg.predict(X_tseq))
            s_new.index = X_tseq.index
            s = pd.concat([s, s_new], ignore_index=False)
        s = s.rename("aval_hat_pmm")

    if method=="extrapolate":
        M = X.copy()
        M["aval"] = y
        last_aval = M.query("tseq==tseqmax")[["pat_no", "aval"]].drop_duplicates().set_index("pat_no")
        last_aval = last_aval.rename(columns={"aval":"last_aval"})
        M = M.join(last_aval, on="pat_no")
        M["aval_diff"] = M.groupby("pat_no")["aval"].transform(lambda x: x.diff(periods=1))
        mean_delta = get_mean_delta(M, registry)
        s = M.apply(lambda x: get_pmm_estimate(x.aval, x.last_aval, int(x.tseq), int(x.tseqmax), mean_delta), axis=1)
        s = s.rename("aval_hat_pmm")

    return s

def get_pmm_estimate(aval, last_aval, tseq, tseqmax, mean_delta_fun):
    '''Computes the PMM estimate'''
    if tseq<=tseqmax:
        val = aval
    else:
        val = last_aval
        for t in range(tseqmax+1, tseq+1):
            val += mean_delta_fun(t)
    return val

def get_mean_delta(M, registry, setting="palliative"):
    '''Returns a function giving for any questionnaire number k the mean delta between the QoL reported at k and the one reported at k-1 (averaged
    over all patients who answered the second-to-last questionnaire). The delta is estimated by averaging over all time points where the
    second-to-last questionnaire has the same temporal distance.'''

    # the if-else distinction is not strictly necessary and only made for performance reasons (case of mamma also applies to pancreas)
    if registry == "pancreas":
        mean_delta = M.query("tseq==tseqmax").aval_diff.mean() 
        fun = lambda k: mean_delta
        
    elif registry == "mamma":
        df = get_time_diffs(registry, setting)
        fun_series = df.groupby("t_diff")["n"].transform(lambda x: M.query("(tseq==tseqmax) and (tseq in @x.to_list())").aval_diff.mean())
        fun = lambda k : fun_series[k]
    
    return fun
        
def get_time_diffs(registry, setting):
    T, N = get_timepoint_list(registry, setting)
    df = pd.DataFrame({"t": T, "n": N})
    df["t_diff"] = df["t"].diff(1)
    df = df.drop(0)
    return df

def get_stacking_regressor(reg_names, X, y):
    reg_list = []

    for r in reg_names:
        if r in ["xgb", "rf", "knn", "mlp"]:
            reg = get_optimized_regressor(X, y, regressor=r)
        elif r == "dummy":
            reg = DummyRegressor(strategy="mean")
        elif r == "linreg":
            reg = LinearRegression(fit_intercept=True)
        elif r == "mlp" :
            reg = MLPRegressor()
        else: 
            continue
        reg_list += [reg]

    reg_tuples = list(zip(reg_names, reg_list))

    return StackingRegressor(reg_tuples, LinearRegression(positive=True, fit_intercept=False))


def get_optimized_regressor(X, y, regressor="xgb"):
    print(f"Optimizing hyperparameters of {regressor} regressor")
    if regressor == "xgb":
        opt = BayesSearchCV(
            xgb.XGBRegressor(),
            {
                "learning_rate": Real(0.001,1),
                "max_depth": Integer(2,10),
                "min_child_weight": [50]
            },
            scoring = None,
            n_iter = 50,
            n_jobs = 5,
            cv = 10,
            refit = True,
            return_train_score=True,
            verbose=0,
            random_state = 17
        )

    elif regressor == "rf":
        opt = BayesSearchCV(
            RandomForestRegressor(),
            {
                "max_depth": Integer(2,20),
                "max_features" : Real(0.1, 1, prior="uniform"),
                "min_samples_leaf" : [50]
            },
            scoring = None,
            n_iter = 100,
            n_jobs = 5,
            cv = 10,
            refit = True,
            return_train_score=True,
            verbose=0,
            random_state = 17
        )

    opt = opt.fit(X.drop("pat_no", axis=1), y, groups= X.pat_no)

    logging.info("="*50) 
    logging.info(f"Optimal parameters for {regressor}:")

    for key, value in opt.best_params_.items():
        logging.info(f"{key}: {value}")

    return opt.best_estimator_

def fit_outcome_model(X, X1, y1, reg, return_coefs=False):
    '''outcome_regression, s = fit_outcome_model(X, X1, y1, return_coefs=False) regresses the outcome vector y1 against the predictor matrix X1 and uses the obtained model to predict the
    outcome of all observations in the predictor matrix X. The return value outcome_regression is a dictionary containing the regressed outcome, the
    predictors and the fitted regressor. If return coefs=True, also the regression coefficients are returned.'''

    X1 = X1.drop("pat_no", axis=1, errors="ignore")
    reg = reg.fit(X1, y1)

    if "linreg" in reg.named_estimators_.keys():
        s = pd.Series(reg.named_estimators_["linreg"].coef_)
        s.index = X1.columns
        s = pd.DataFrame(s.sort_values().rename("coeff_Y"))  

    else:
        s = pd.DataFrame()

    y_hat = pd.Series(reg.predict(X.drop("pat_no", axis=1))).rename("aval_hat")
    y_hat.index = X.index

    outcome_regression = {
            "regressed_outcome" : y_hat,
            "predictors" : X,
            "fitted_regressor" : reg
        }

    if return_coefs:
        return outcome_regression, s
    else:
        return outcome_regression

if __name__=="__main__":
    main()