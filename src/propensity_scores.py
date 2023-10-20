import pickle 
import pandas as pd
import yaml 
import os
from matplotlib import pyplot as plt
import seaborn as sns
from functions import saveplot

def main():
    params_path = os.path.join(os.path.dirname(__file__),"..","params.yaml")
    params = yaml.load(open(params_path), Loader=yaml.FullLoader)
    analysis = params["analysis"]
    registry = analysis["registry"]
    bootstrap = analysis["bootstrap"]   
    w_max = analysis["w_max"]
    
    data_dir = os.path.join(os.path.dirname(__file__),"..","data","{r}").format(r=registry)   
    out_path = os.path.join(data_dir,"propensity_scores.pkl")
    fig_path = os.path.join(os.path.dirname(__file__),"..","figures","{r}","$fname").format(r=registry)

    reg = pickle.load(open(os.path.join(data_dir, "ps_regression_results.pkl"), "rb"))
    reg_data = pickle.load(open(os.path.join(data_dir, "regression_data.pkl"), "rb"))   
    X = reg_data["predictors"]
    
    if "pat_no" in X.columns:
        X.drop("pat_no", axis=1, inplace=True)

    p = pd.DataFrame(reg.predict_proba(X), index=X.index).rename(columns={0: "q", 1:"p"})
    p["invprob_p"] = p["p"].apply(lambda p: w_max if p<1/w_max else 1/p)
    p["invprob_q"] = p["q"].apply(lambda q: w_max if q<1/w_max else 1/q)   
 
    results_dict = {"propensity_scores" : p}

    plot_ps_dist(p, reg_data["status"], fig_path)

    if bootstrap:
        samples_path = os.path.join(data_dir, "bootstrap_samples.pkl")

        with open(samples_path, "rb") as file:
            bs_samples = pickle.load(file)["bs_samples"]

        bs_reg_path = os.path.join(data_dir,"bootstrap_ps_regression_results.pkl")

        with open(bs_reg_path, "rb") as file:
            bs_reg_list = pickle.load(file)

        bs_ps_list = []

        for k, s in enumerate(bs_samples):
            X_bs = X.loc[s]
            bs_reg = bs_reg_list[k]
            bs_p = pd.DataFrame(bs_reg.predict_proba(X_bs), index=X_bs.index).rename(columns={0: "q", 1:"p"})
            bs_p["invprob_p"] = bs_p["p"].apply(lambda p: w_max if p<1/w_max else 1/p)
            bs_p["invprob_q"] = bs_p["q"].apply(lambda q: w_max if q<1/w_max else 1/q)

            bs_ps_list += [bs_p]

        results_dict["bootstrap_propensity_scores"] = bs_ps_list

    with open(out_path, "wb") as file:        
        pickle.dump(results_dict, file)

def plot_ps_dist(p, status, fig_path):
    r = p.join(status)

    bins = [k*0.05 for k in range(21)]
    z0_color = "grey"
    z1_color = "k"
    bw = .75
    r.query("status==0")["p"].hist(color=z0_color, alpha=1, density=True, bins=bins, histtype="step", linewidth=1)
    sns.kdeplot(r.query("status==0")["p"], color=z0_color, label="not answered", linewidth=2, bw_adjust=bw, clip=[0,1])
    ax = plt.gca()
    r.query("status==1")["p"].hist(color=z1_color, alpha=1, linewidth=1, density=True, bins=bins, histtype="step")
    sns.kdeplot(r.query("status==1")["p"], color=z1_color, label="answered", linewidth=2, bw_adjust=bw, clip=[0,1])
    plt.legend(loc="upper left")
    plt.xlim(left=0)
    plt.xlim(right=1)
    plt.xlabel("Propensity score")
    saveplot(fig_path.replace("$fname","ps_distribution.{fmt}"))
if __name__=="__main__":
    main()