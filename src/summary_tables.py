import pandas as pd 
import pickle 
import os 
import yaml 
import numpy as np
import copy

def substitute(x, predictors):
    for p in predictors:
        x = x.replace(p["name"], p["label"])
    return x

def round_appropriately(x):
    if np.abs(x)>1:
        x = np.around(x,1)
    elif x == 0:
        return x
    else:
        log_x = np.log(np.abs(x))/np.log(10)
        first_sd = int(-np.floor(log_x))
        x = np.around(x, first_sd+2)
    return x

def main():
    params_path = os.path.join(os.path.dirname(__file__),"..","params.yaml")
    params = yaml.load(open(params_path), Loader=yaml.FullLoader)
    analysis = params["analysis"]
    registry = analysis["registry"]
    predictors = analysis[registry]["predictors"]
    pred_dict = {p["name"] : p for p in predictors}

    data_dir = os.path.join(os.path.dirname(__file__),"..","data","{r}").format(r=registry) 

    with open(os.path.join(data_dir, "regression_data.pkl"), "rb") as file: 
        reg_data = pickle.load(file)

    X = reg_data["predictors_unscaled_for_baltab"]
    X_bl = reg_data["predictors_baseline"]
    if "tnmt = TX" in X.columns:
        X["tnmt = missing\ or\ unknown"]= X["tnmt = missing\ or\ unknown"]+X["tnmt = TX"]
        X.drop("tnmt = TX", axis=1, inplace=True)
        X_bl["tnmt = missing\ or\ unknown"] = X_bl["tnmt = missing\ or\ unknown"]+X_bl["tnmt = TX"]
        X_bl.drop("tnmt = TX", axis=1, inplace=True)

    z = reg_data["status"]

    with open(os.path.join(data_dir, "propensity_scores.pkl"), "rb") as file:
        ps_dict = pickle.load(file)

    p = ps_dict["propensity_scores"]

    X = replace_with_NA(X)
    X_bl = replace_with_NA(X_bl)

    M = X.join(z).join(p)

    cols = X.drop("pat_no", axis=1).columns

    # create table 1 (summary of baseline variables) and export

    tab1 = make_table1(X_bl, predictors, pred_dict)
    tab1.to_csv(os.path.join(data_dir,"table1.csv"), sep=";", index=False)

    # create balance table and export (number formatting is done in R)
    baltab = make_baltab(M, cols, predictors, pred_dict)
    baltab.to_csv(os.path.join(data_dir,"baltab.csv"), sep=";", index=False)

def make_table1(X, predictors, pred_dict):
    X = X.drop("pat_no", axis=1, errors="ignore")
    pdict = copy.deepcopy(pred_dict)
    pdict["ecog_baseline"]["type"] = "categorical"
    pdict["charlsonscore"]["type"] = "categorical"

     # population mean, variance and standard deviation
    mean = pd.DataFrame(X.mean().rename("total"))
    mean["metric"] = "mean"
    count = pd.DataFrame(X.sum().rename("N"))
    mean = mean.join(count)
    median = pd.DataFrame(X.median().rename("total"))
    median["metric"] = "median"   
    std = pd.DataFrame(X.std().rename("total"))
    std["metric"] = "std"
    mini = pd.DataFrame(X.min().rename("total"))
    mini["metric"] = "min"
    maxi = pd.DataFrame(X.max().rename("total"))
    maxi["metric"] = "max"

    total_N = get_total_N(X, include_weighted=False)
    total_N.drop(["Z=0", "Z=1", "weighted (Z=1)"], axis=1, inplace=True)
    total_N["rowlabel"] = ["patients (N)"]

    all_metrics = pd.concat([
        mean,
        median,
        std,
        mini,
        maxi
    ], ignore_index=False)

    all_metrics = add_label_columns(all_metrics, pdict)
    all_metrics = add_order_column(all_metrics, predictors)
    all_metrics["is_baseline"] = all_metrics.apply(lambda x: pdict[x.varname].get("is_baseline", False), axis=1)    
    all_metrics["suborder"] = all_metrics["metric"].apply(suborder)

    all_metrics = all_metrics.query(
        "(varname=='age' and metric=='median' or\
         varname!='age' and metric=='mean' or \
         varname not in ['age', 'ecog_baseline', 'charlsonscore'] and type in ['continuous', 'ordinal'] and metric=='std' or\
         varname == 'age' and metric in ['min', 'max']) and\
         is_baseline == True"      
    )

    all_metrics = pd.concat([total_N, all_metrics], ignore_index=False)
    all_metrics["varlabel"] = all_metrics["varlabel"].apply(lambda x: x.replace(" at baseline", ""))

    tab1 = all_metrics.sort_values(["order", "suborder"])
    tab1 = tab1.drop(["is_baseline", "suborder", "metric"], axis=1)
    tab1["rowlabel"] = tab1.apply(rename_level, pred_dict=pdict, axis= 1)
    
    tab1.N = tab1.apply(lambda x: int(x.N) if x.type in ['categorical', "binary"] else pd.NA, axis=1) 
    
    return tab1

def suborder(x):
    if x == "mean":
        s = 0
    elif x == "median":
        s = 1
    elif x == "std":
        s = 2
    elif x == "min":
        s = 3
    elif x == "max":
        s = 4
    return s

def get_total_N(M):
    if "status" in M.columns:
        grouped = M.groupby("status")

    total_N = pd.DataFrame(
        data={
       "total" : [int(M.shape[0])],
       "Z=1" : [int(M.query("status==1").shape[0])] if "status" in M.columns else [pd.NA],
       "Z=0" : [int(M.query("status==0").shape[0])] if "status" in M.columns else [pd.NA],
        "weighted (Z=1)" : [int(round(grouped["invprob_p1"].sum().loc[1]))] if "invprob_p1" in M.columns else [pd.NA],
       "varlabel" : [""],
       "rowlabel" : ["observations (N)"],
       "order" : [0],
       "varname" : [""],
       "type" : ["int"]
    })  

    return total_N

def make_baltab(M, variables, predictors, pred_dict):
    M = M.drop("pat_no", axis=1, errors="ignore")

    normfactor_dict = get_normfactor_dict(M)

    # means and standard deviations stratified by answering status
    grouped = M.groupby("status")
    unweighted_mean = grouped.agg(weighted_avg).transpose().rename(columns={0:"Z=0", 1:"Z=1"})
    unweighted_std = grouped.agg(np.nanstd).transpose().rename(columns={0:"Z=0", 1:"Z=1"})

    total_N = get_total_N(M)

    # population mean, variance and standard deviation
    population_mean = pd.DataFrame(M.mean().rename("total"))
    population_var = pd.DataFrame(M.var().rename("total"))
    population_std = pd.DataFrame(M.std().rename("total"))

    # mean, variance and standard deviation in weighted population of answering patients 
    weighted_mean = pd.DataFrame(M.apply(weighted_avg, weights=M["invprob_p1"], axis=0).rename("weighted (Z=1)"))
    weighted_var = pd.DataFrame((M.apply(lambda x: M["invprob_p1"]*(x-weighted_avg(x, weights=M["invprob_p1"]))**2*normfactor_dict.get(x.name, normfactor_dict["all"]) if x.nunique()!=2 else -1, axis=0).sum()).rename("weighted (Z=1)"))
    weighted_std = pd.DataFrame(weighted_var["weighted (Z=1)"].apply(lambda v: np.sqrt(v)).rename("weighted (Z=1)"))

    # variance ratio between population and weighted population of answering patients
    weighted_var["weighted (Z=1)"] = weighted_var["weighted (Z=1)"].apply(lambda x: x if x>=0 else pd.NA)
    var_ratio = (population_var["total"]/(weighted_var["weighted (Z=1)"] +10**(-12))).rename("variance ratio")

    # standardized differences between population and weighted population of answering patients
    standard_diffs_weighted = pd.DataFrame(((weighted_mean["weighted (Z=1)"]-population_mean["total"])/population_std["total"]).rename("standardized difference (weighted)"))
    standard_diffs_unweighted = pd.DataFrame(((unweighted_mean["Z=1"]-population_mean["total"])/population_std["total"]).rename("standardized difference"))

    standard_diffs = pd.DataFrame(standard_diffs_unweighted).join(standard_diffs_weighted)

    # join standard_diffs and variance_ratio into deviation_stats
    deviation_stats = pd.DataFrame(standard_diffs).join(var_ratio)
    deviation_stats["metric"] = "mean"
    deviation_stats = deviation_stats.loc[variables]

    means = population_mean.join(weighted_mean).join(unweighted_mean)
    means = means.loc[variables]
    means["metric"] = "mean"
    
    stds = population_std.join(weighted_std).join(unweighted_std)
    stds = stds.loc[variables]
    stds["metric"] = "std"

    summary_stats = pd.concat([means, stds], ignore_index=False)

    summary_stats = add_label_columns(summary_stats, pred_dict)
    deviation_stats = add_label_columns(deviation_stats, pred_dict)

    summary_stats = summary_stats.query("rowlabel==rowlabel")
    summary_stats.drop("metric", axis=1, inplace=True)

    # add order column (needed by iomedicoR::tableOutFun)
    summary_stats = add_order_column(summary_stats, predictors)
    deviation_stats = add_order_column(deviation_stats, predictors)

    # add information on total number of patients
    summary_stats = pd.concat([total_N, summary_stats])
    summary_stats = summary_stats.sort_values(by=["order", "type"])

    # replace level names of some variables
    summary_stats["rowlabel"] = summary_stats.apply(rename_level, pred_dict=pred_dict, axis=1)
    deviation_stats["rowlabel"] = deviation_stats.apply(rename_level, pred_dict=pred_dict, axis=1)

    # merge summary statistics with deviation stats
    baltab = summary_stats.merge(deviation_stats[["varlabel", "rowlabel", "standardized difference", "standardized difference (weighted)", "variance ratio"]], on=["varlabel", "rowlabel"], how="left")
    
    return baltab

def get_normfactor_dict(M):
    '''Normalization factors used for calculuating weighted means and variances. For columns where means and variances are not calculated over entire
    population, adapted normalization factors are needed'''
    normfactor_dict = dict()
    M["invprob_p1"] = M["status"]*M["invprob_p"]
    N_weighted = M["invprob_p1"].sum()
    sq_weighted = (M["invprob_p1"]**2).sum() 
    N_weighted_died = M.query("`pat_stays_alive = no` == 1")["invprob_p1"].sum()
    sq_weighted_died = (M.query("`pat_stays_alive = no` == 1")["invprob_p1"]**2).sum()
    N_weighted_prog = M.query("`progression_free_patient = no` == 1")["invprob_p1"].sum()
    sq_weighted_prog = (M.query("`progression_free_patient = no` == 1 ")["invprob_p1"]**2).sum()
    normfactor_dict["all"] = N_weighted/(N_weighted**2-sq_weighted)
    normfactor_dict["days_to_death"] = N_weighted_died/(N_weighted_died**2-sq_weighted_died)
    normfactor_dict["days_from_progress"] = N_weighted_prog/(N_weighted_prog**2-sq_weighted_prog)

    return normfactor_dict

def weighted_avg(x, weights=None):
    masked = np.ma.masked_array(x, pd.isna(x))
    avg = np.ma.average(masked, weights=weights)
    return avg

def na_std(x):
    std = np.nanstd(x)
    return std

def replace_with_NA(X):
    '''replace values of days_to_death and days_from_progress with pd.NA for observations without recorded death or no progression'''
    alive_idx = X.query("`pat_stays_alive = yes` == 1").index
    progfree_idx = X.query("`progression_free_patient = yes` == 1").index 

    X.loc[alive_idx, "days_to_death"] = np.nan 
    X.loc[progfree_idx, "days_from_progress"] = np.nan

    return X

def add_order_column(df, predictors):
    '''add column order to dataframe (needed by iomedicoR::tableOutFun)'''
    df["order"] = pd.NA
    c = 1
    for p in predictors:
        df.loc[df.varlabel==p["label"], "order"] = c
        c+=1
    return df

def add_label_columns(df, pred_dict):
    '''add columns needed by iomedicoR::tableOutFun'''
    df["varname_cat"] = df.index
    df["varname"] = df["varname_cat"].apply(lambda x: x.split(" = ")[0])
    df["rowlabel"] = df.apply(get_rowlabel, axis=1, pred_dict=pred_dict)
    df["varlabel"] = df.apply(get_varlabel, axis=1, pred_dict=pred_dict)
    df["type"] = df["varname"].apply(lambda x: pred_dict[x]["type"])
    return df

def rename_level(x, pred_dict):
    vname = x.varname
    rowlabel = x.rowlabel
    p_dict = pred_dict.get(vname, dict())
    level_dict = p_dict.get("levels", dict())
    new_rowlabel = level_dict.get(rowlabel, rowlabel)
    return new_rowlabel

def get_varlabel(x, pred_dict):
    return pred_dict[x.varname]["label"]

def get_rowlabel(x, pred_dict):
    p = pred_dict[x.varname]
    type = p["type"]
    if type in ["ordinal", "continuous"]:
        if x.metric == "std":
            return "sd"
        elif x.metric in ["mean", "median", "min", "max"]:
            return x.metric
    else:
        if x["metric"] == "mean":
            if "rowlabel" in p.keys():
                return p["rowlabel"]
            else:
                return x.varname_cat.split(" = ")[-1].replace("\\", "")
        else :
            return pd.NA

if __name__=="__main__":
    main()