import os
import re
import yaml
import numpy as np
import pandas as pd
import pickle

def main():
    params_path = os.path.join(os.path.dirname(__file__),"..","params.yaml")
    params = yaml.load(open(params_path), Loader=yaml.FullLoader)
    analysis = params["analysis"]
    registry = analysis["registry"]
    bootstrap = analysis.get("bootstrap", False)
    plot_types = analysis["plots"].get("aggregations", [])
    data_dir = os.path.join(os.path.dirname(__file__),"..","data","{r}").format(r=registry)
    out_path = os.path.join(os.path.join(data_dir,"results_tables.pkl"))

    data = pickle.load(open(os.path.join(data_dir, "qol_data.pkl"), "rb"))
    bootstrap_samples = pickle.load(open(os.path.join(data_dir, "bootstrap_samples.pkl"), "rb"))
    ps_dict = pickle.load(open(os.path.join(data_dir, "propensity_scores.pkl"), "rb"))
    outcome_regression =  pickle.load(open(os.path.join(data_dir, "outcome_regression.pkl"), "rb"))

    answer_data = data.answer_data
    propensity_scores = ps_dict["propensity_scores"]
    regressed_outcome = outcome_regression["regressed_outcome"]
    regressed_outcome_pmm = outcome_regression["regressed_outcome_pmm"]

    results_table_dict = dict()

    if registry == "pancreas":
        answer_data["pat_no"] = answer_data.index.get_level_values("pat_no")
    elif registry == "mamma":
        answer_data["pat_no"] = answer_data.reset_index(level=[0,1]).apply(lambda x: str(x.cid)+"-"+str(x.patientoid), axis=1).values

    answer_data = answer_data.reset_index(drop=True).set_index(["pat_no", "tseq"], drop=False)

    pat_nos = bootstrap_samples["pat_nos"]
    answer_data_sample = answer_data.query("pat_no in @pat_nos").copy()

    for plt_type in plot_types:
        res_table = results_table(
            answer_data_sample, 
            propensity_scores,
            regressed_outcome,
            regressed_outcome_pmm,
            plt_type,
            change_to_baseline = analysis["change_to_bl"],
            dead_as_zero = analysis["dead_as_zero"],
            maximum_weight = analysis["w_max"],
            filter = analysis["filter"],
            theta = analysis["theta"],
            score_min = analysis[registry]["qol_score"]["min"],
            bin_width = analysis[registry].get("bin_width", 60),
            bin_max = analysis[registry].get("bin_max", 900),
            registry = registry
            )

        if bootstrap:
            bs_res_tables = []
            for k, pat_sample in enumerate(bootstrap_samples["bs_pat_samples"]):                
                propensity_scores = deduplicate(ps_dict["bootstrap_propensity_scores"][k])
                regressed_outcome = deduplicate(outcome_regression["bootstrap_regressions"][k]["regressed_outcome"])
                regressed_outcome_pmm = deduplicate(outcome_regression["bootstrap_regressions"][k]["regressed_outcome_pmm"])
                answer_data_sample = answer_data.reset_index(level=1, drop=True).loc[pat_sample].set_index(["pat_no", "tseq"], drop=False)
                
                bs_res_table = results_table(
                    answer_data_sample,
                    propensity_scores,
                    regressed_outcome,
                    regressed_outcome_pmm,
                    plt_type,
                    change_to_baseline = analysis["change_to_bl"],
                    dead_as_zero = analysis["dead_as_zero"],
                    maximum_weight = analysis["w_max"],
                    filter = analysis["filter"],
                    theta = analysis["theta"],
                    score_min = analysis[registry]["qol_score"]["min"],
                    bin_width = analysis[registry].get("bin_width", 60),
                    bin_max = analysis[registry].get("bin_max", 900),
                    registry = registry
                )
                bs_res_tables += [bs_res_table]

            joined_table = bs_res_tables[0]

            for k, bs_res_table in enumerate(bs_res_tables):
                if k == 0:
                    continue 
                else:
                    joined_table = joined_table.join(bs_res_table, rsuffix=f"_{k}", how="left")
            
            for stat in ["mean_aval_na", "mean_aval_na_pmm", "delta_ac_na"]: #mean_aval_locf, median_aval]
                joined_stat = joined_table.filter(regex=re.compile(stat+"_\d"))
                joined_stat[stat] = joined_stat.mean(axis=1)
                ci_lb_stat = joined_stat[stat]-joined_stat.drop(stat, axis=1).quantile(0.025, axis=1).rename("ci_lb_"+stat)
                ci_ub_stat = joined_stat.drop(stat, axis=1).quantile(0.975, axis=1).rename("ci_ub_"+stat)-joined_stat[stat]
                
                res_table = res_table.join(ci_lb_stat.rename("ci_lb_"+stat)).join(ci_ub_stat.rename("ci_ub_"+stat))

        results_table_dict[plt_type]  = res_table 

    with open(out_path, "wb") as file:
        pickle.dump(results_table_dict, file)

def results_table(answer_data, propensity_scores, regressed_outcome, regressed_outcome_pmm, plt_type, **kwargs): 
    """Prepare table of results"""   
    dead_as_zero = kwargs.get("dead_as_zero", False)
    cond = kwargs.get("filter",None)
    bin_width = kwargs.get("bin_width", 60)
    bin_max = kwargs.get("bin_max", 900)
    registry = kwargs.get("registry", "pancreas")

    round_pmm_vals = True if registry == "pancreas" else False

    r = answer_data.query(cond) if cond is not None else answer_data

    to_join = propensity_scores
    for s in [regressed_outcome, regressed_outcome_pmm]:
        to_join = to_join.join(s, how="outer")    

    r = r.join(to_join, on="obs_no", how="left")

    r = r[["obs_no","t0val","tseq","tseqmax","days_to_death","days_from_therapy_start","aval","status","p", "aval_hat","aval_hat_pmm"]]
    
    if "cid" in r.index.names and "patientoid" in r.index.names:
        pat_nos = pd.Series(r.reset_index(level=[0,1]).apply(lambda x: str(x.cid)+"-"+str(x.patientoid), axis=1)).rename("pat_no")
        pat_nos.index = r.index
        r["pat_no"] = pat_nos

    r["non_answer"] = r["status"]==0

    grouped = r.groupby("pat_no")
    is_complete = ~(grouped["non_answer"].any()).rename("is_complete")
    r = r.join(is_complete, on="pat_no")

    if dead_as_zero:
        r = r.query("status in [0,1,2]")
        r["aval"] = r.apply(lambda x: 0 if x["status"]==2 else x["aval"], axis=1).copy()
        r["aval_hat"] = r.apply(lambda x: 0 if x["status"]==2 else x["aval_hat"], axis=1).copy()
        r["aval_hat_pmm"] = r.apply(lambda x: 0 if x["status"]==2 else x["aval_hat_pmm"], axis=1).copy()
        r["p"] = r.apply(lambda x: 1 if x["status"] == 2 else x["p"], axis=1).copy()
        r["status"] = r.apply(lambda x: 1 if x.status==2 else x.status, axis=1).copy()
    else:
        r = r.query("status in [0,1]")
    
    r["aval_hat_pmm"] = r.apply(lambda x: x["aval_hat_pmm"] if not pd.isna(x["aval_hat_pmm"]) else x["aval_hat"], axis=1)
    r["aval_hat_pmm"] = r["aval_hat_pmm"].apply(lambda x: max(1, x))
    
    if round_pmm_vals:
        r["aval_hat_pmm"] = r["aval_hat_pmm"].apply(lambda x: round(x))

    r["aval_locf"] = r.groupby("pat_no")["aval"].transform(lambda s: s.fillna(method="ffill"))

    # set probability of responding to 1 for tseq=0
    r["p"] = r.apply(lambda x: x["p"] if x.tseq>0 else 1, axis=1).copy()

    r["aval_hat"] = r.apply(lambda x: x.aval if x.tseq==0 else x.aval_hat, axis=1)
    r["p"] = r.apply(lambda x: 1 if x.tseq==0 else x.p, axis=1)
    r = r.query("aval_hat==aval_hat")

    # set aval to 0 for patients who did not respond (for convenience only)
    r["aval_nanfill"] = r.apply(lambda x: 9999 if x.status==0 else x["aval"], axis=1).copy()

    r["summand"] = r.apply(lambda x: get_dr_summand(x["aval_hat"], x["p"], x["aval_nanfill"], x["status"]), axis=1)

    if plt_type=="auc":
        r = r.reset_index(drop=True).copy()
        grouped = r.copy()
    
    if plt_type=="by_dtd":
        r = r.query("days_to_death>0").reset_index(drop=True).copy()
        r["dtd_binned"] = pd.cut(r.days_to_death, bins=np.arange(0,bin_max,bin_width))
        grouped = r.groupby("dtd_binned")
    
    if plt_type=="by_dfs":
        r = r.reset_index(drop=True).copy()
        r["dfs_binned"] = pd.cut(r.days_from_therapy_start, bins=np.arange(0,1900,30.4375*6))
        grouped = r.groupby("dfs_binned")

    if plt_type=="by_tseq":
        r = r.reset_index(drop=True).copy()
        grouped = r.groupby("tseq")

    if plt_type=="by_tseq_and_tseqmax":
        r = r.reset_index(drop=True).copy()
        grouped = r.groupby(["tseqmax","tseq"])

    r["aval_locf"] = grouped["aval"].transform(lambda x: x.fillna(method="ffill"))

    pltdata = grouped["summand"].mean()

    if plt_type == "auc":
        pltdata = pd.DataFrame({"mean_aval_all":[pltdata]})
    else: 
        pltdata = pd.DataFrame(pltdata.rename("mean_aval_all"))

    pltdata["n_alive"] = grouped["status"].count()
    pltdata["n_answer"] = grouped["status"].agg(lambda x: x[x==1].shape[0])
    pltdata["n_no_answer"] = grouped["status"].agg(lambda x: x[x==0].shape[0])
    pltdata["mean_aval_all_pmm"] = grouped["aval_hat_pmm"].mean()
    pltdata["mean_aval_ac"] = grouped["aval"].mean()
    pltdata["ci_lb_mean_aval_ac"] = 1.96*grouped["aval"].sem()
    pltdata["ci_ub_mean_aval_ac"] = pltdata["ci_lb_mean_aval_ac"]
    pltdata["mean_aval_na"] = pltdata.apply(get_aval_no_answer, axis=1)
    pltdata["mean_aval_na_pmm"] = pltdata.apply(get_aval_no_answer, estimate="pmm", axis=1)
    pltdata["delta_ac_na"] = pltdata["mean_aval_ac"]-pltdata["mean_aval_na"]

    return pltdata


def get_aval_no_answer(x, estimate="main"):    
    n_all = x["n_alive"]
    n_ac = x["n_answer"]
    n_na = x["n_no_answer"]
    y_ac = x["mean_aval_ac"]
    if estimate=="main":
        y_all = x["mean_aval_all"]
    elif estimate=="pmm":
        y_all = x["mean_aval_all_pmm"]

    return 1/(n_na)*(y_all*n_all-y_ac*n_ac)

def get_dr_summand(mu, P, Y, Z):
    return Z/P*(Y-mu)+mu

def get_ipw_summand(P, Y, Z):
    return Z/P*Y

def get_ipw_summand_sensitivity(P, Y, Z, theta):
    return Z*(Y+(Y-theta)*(1/P-1))
 
def get_regression_summand(mu):
    return mu

def get_cc_summand(Y, is_complete):
    if is_complete:
        return Y
    else:
        return pd.NA

def deduplicate(s):
    s = s.rename_axis(index="obs_no").reset_index(drop=False)
    s = s.drop_duplicates(["obs_no"], keep="first").set_index("obs_no", drop=True)
    return s

if __name__=="__main__":
    main()

