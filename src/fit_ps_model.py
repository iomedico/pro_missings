import logging
import os
from re import A
import numpy as np
import pickle
import yaml
import pandas as pd
import xgboost as xgb
import datetime as dt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# from scipy.stats import ttest_1samp
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import classification_report
from skopt.space import Categorical, Integer, Real
from skopt import BayesSearchCV
from scipy.optimize import minimize, LinearConstraint
from mystackingclassifier import MyStackingClassifier

def main():
    params_path = os.path.join(os.path.dirname(__file__),"..","params.yaml")
    analysis = yaml.load(open(params_path), Loader=yaml.FullLoader)["analysis"]
    registry = analysis.get("registry","pancreas")
    bootstrap = analysis.get("bootstrap", False)    
    data_dir = os.path.join(os.path.dirname(__file__),"..","data","{r}").format(r=registry)
    log_path = os.path.join(os.path.dirname(__file__), "..", "log", f"log_{registry}.log")
    reg_data = pickle.load(open(os.path.join(data_dir,"regression_data.pkl"), "rb"))
    bootstrap_samples = pickle.load(open(os.path.join(data_dir, "bootstrap_samples.pkl"), "rb"))["bs_samples"]

    logging.basicConfig(filename=log_path, level=logging.INFO, filemode="w")

    logging.info("Fitting model for propensity score estimation\n")
    logging.info(dt.datetime.now())

    clf_list = [
        "logreg",
        "xgb",
        "dummy"
    ]

    X = reg_data["predictors"]

    if "knn" in clf_list:
        X_scaled = pd.DataFrame(StandardScaler().fit_transform(X))
        X_scaled.columns = X.columns
        X_scaled.index = X.index
        X = X_scaled
    
    z = reg_data["status"]
    
    Xtrain, Xvalidate, ztrain, zvalidate = train_test_split(X, z, train_size=.8, random_state=23)

    if len(clf_list)>1:
        clf = get_stacking_classifier(clf_list, Xtrain, ztrain)
    else:
        clf = get_optimized_classifier(Xtrain, ztrain, "logreg")

    if "logreg" in clf_list:
        opt, coefs = fit_ps_model(X, z, clf, return_coefs=True, log_report=True)
        coefs.to_csv(os.path.join(data_dir,"ps_regression_coefficients.csv"), sep=";")
    else:
         opt = fit_ps_model(X, z, clf, return_coefs=False)       

    p =  os.path.join(data_dir,"ps_regression_results.pkl")
    with open(p, "wb") as file:
        pickle.dump(opt, file)   
 
    if bootstrap:
        bs_reg_list = []

        for k, s in enumerate(bootstrap_samples):
            print(f"Fitting classifier bootstrap sample", k+1)

            X_bs = X.loc[s]
            z_bs = z.loc[s]

            reg_bs = fit_ps_model(X_bs, z_bs, clf, log_report=False)
            bs_reg_list += [reg_bs]

    p_bs =  os.path.join(data_dir,"bootstrap_ps_regression_results.pkl")

    with open(p_bs, "wb") as file:
        pickle.dump(bs_reg_list, file)

def fit_ps_model(X, z, clf, return_coefs=False, log_report=False):

    X = X.drop("pat_no", errors="ignore", axis=1)
    clf = clf.fit(X, z)
    z_pred = clf.predict(X)

    if log_report:
        logging.info("Classification report:")
        logging.info(classification_report(z, z_pred))

    if return_coefs:
        if isinstance(clf, StackingClassifier) or isinstance(clf, MyStackingClassifier):
            for c in clf.estimators:
                if c[0] == "logreg":
                    logreg = c[1].fit(X, z)
                    break

            s = pd.Series(logreg.coef_.ravel()).rename("coeff_Z")
            s.index = logreg.feature_names_in_

        elif isinstance(clf, LogisticRegression):
            logreg = clf
       
        s = pd.Series(logreg.coef_.ravel()).rename("coeff_Z")
        s.index = logreg.feature_names_in_           

        return clf, s

    else:
        return clf

def get_stacking_classifier(clf_names, X, z, strategy="balance"):
    if strategy == "balance":
        clf_list = []

        for c in clf_names:
            if c in ["logreg","rf","xgb","knn"]:
                print(f"Optimizing hyperparameters of {c} classifier")
                clf = get_optimized_classifier(X, z, classifier=c)
            elif c == "tree":
                clf = DecisionTreeClassifier()
            elif c == "dummy":
                clf = DummyClassifier(strategy="prior")
            elif c == "logreg":
                clf = LogisticRegression(penalty="none")
            else:
                continue
            clf_list += [clf]

        clf_tuples = list(zip(clf_names, clf_list))

        def loss_function(weights):
            estimator = MyStackingClassifier(clf_tuples, weights).fit(X.drop("pat_no", axis=1), z)
            return -balance_score(estimator, X.drop("pat_no", axis=1), z)

        constr_list = [LinearConstraint(np.ones((1, len(clf_tuples))), 1, 1)] # all coefficients sum to one
        
        for k, _ in enumerate(clf_tuples):
            v = np.zeros((1,len(clf_tuples)))
            v[0,k] = 1
            constr_list += [LinearConstraint(v, 0, 1)] # all coefficients are positive (and less than 1)

        ini_guess = pd.DataFrame(normalize(np.random.rand(30,len(clf_tuples)), axis=1, norm="l1"))
        fun_vals = []
        opt_list = []

        for idx in ini_guess.index:
            opt = minimize(loss_function, ini_guess.loc[idx].values, options={"maxiter": 10000, "disp":True}, constraints=constr_list)
            fun_vals += [opt.fun]
            opt_list += [opt]

        fun_vals = pd.Series(fun_vals).astype(float)
        opt = opt_list[fun_vals.idxmin()]
        
        print(opt)

        logging.info(f"\n Coefficients for {clf_names} in MyStackingClassifier: {opt.x}")

        return MyStackingClassifier(clf_tuples, opt.x)

    elif strategy=="error":
        clf_list = []

        for c in clf_names:
            if c in ["rf","xgb","knn","logreg"]:
                print(f"\n Optimizing hyperparameters of {c} classifier")
                clf = get_optimized_classifier(X, z, classifier=c)
            elif c == "tree":
                clf = DecisionTreeClassifier()
            elif c == "dummy":
                clf = DummyClassifier(strategy="prior")
            else:
                continue
            clf_list += [clf]

        clf_tuples = list(zip(clf_names, clf_list))
    
        return StackingClassifier(clf_tuples)

    else:
        raise ValueError("strategy must be either 'balance' or 'error'")

def get_optimized_classifier(X, z, classifier="xgb"):
    '''Optimize hyperparameters of a classifier'''
    if classifier == "xgb":
        opt = BayesSearchCV(
            xgb.XGBClassifier(),
            {
                "learning_rate": Real(0.001,1, prior="uniform"),
                "max_depth": Integer(2,10),
                "min_child_weight": [50],
                "use_label_encoder" : [False],
                "eval_metric" : ["logloss"]
            },
            scoring = balance_score,
            n_iter = 50,
            n_jobs = 5,
            cv = 10,
            refit = True,
            return_train_score=True,
            verbose=0,
            random_state = 17
        )

    elif classifier == "rf":
        opt = BayesSearchCV(
            RandomForestClassifier(),
            {
                "max_depth": Integer(2,20),
                "max_features" : Real(0.01, 1, prior="uniform"),
                "min_samples_leaf" : [50]
            },
            scoring = balance_score,
            n_iter = 100,
            n_jobs = 5,
            cv = 10,
            refit = True,
            return_train_score=True,
            verbose=0,
            random_state = 17
        )

    elif classifier == "knn":
        opt = BayesSearchCV(
            KNeighborsClassifier(),
            {
                "n_neighbors": Integer(10,50),
                "weights" : Categorical(["uniform"]),
                "p" : Categorical([1,2])
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
    
    elif classifier == "logreg":
        opt = BayesSearchCV(
            LogisticRegression(),
            {
                "penalty": Categorical(["none","l2"]),
                "C" : Real(0.0001,50, prior="uniform"),
                "solver" : ["lbfgs"],
                "max_iter" : [5000]
            },
            scoring = balance_score,
            n_iter = 100,
            n_jobs = 5,
            cv = 10,
            refit = True,
            return_train_score=True,
            verbose=0,
            random_state = 17
        )

    opt = opt.fit(X.drop("pat_no", axis=1, errors="ignore"), z, groups= X.pat_no)

    logging.info(f"\n Optimal parameters for {classifier}:")

    for key, value in opt.best_params_.items():
        logging.info(f"{key}: {value}")

    return opt.best_estimator_

def balance_score(estimator, X, z):    
    predictors = X.columns
    z = z.rename("z")
    p = pd.Series(estimator.predict_proba(X)[:,1])
    p.index = X.index
    q = (1-p)
    inv_p = (1/p).rename("inv_p")
    inv_q = (1/q).rename("inv_q")
    M = X.join(z).join(inv_p).join(inv_q)
    M0 = M.query("z==0").copy()
    M1 = M.query("z==1").copy()
    Z0 = M0["inv_q"].sum()
    Z1 = M1["inv_p"].sum()
    means_weighted_1 = M1[predictors].apply(lambda x: x*M1["inv_p"], axis=0).sum()/Z1
    pop_mean = M[predictors].mean()
    std = M[predictors].std()+10**(-9)
    std_diffs = (1/std*np.abs(means_weighted_1-pop_mean))
    score = -np.abs(std_diffs).sum()

    return score

if __name__=="__main__":
    main()
