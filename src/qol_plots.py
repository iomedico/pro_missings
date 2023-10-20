import pandas as pd
import numpy as np
import os
import pickle
import yaml
from matplotlib import pyplot as plt, rc
from functions import get_timepoint_list

def main():
    params_path = os.path.join(os.path.dirname(__file__),"..","params.yaml")
    params = yaml.load(open(params_path), Loader=yaml.FullLoader)
    analysis = params["analysis"]
    registry = analysis["registry"]
    fs = analysis["plots"].get("font_size", 18)

    global score_label, score_name
    score_label=analysis[registry]["qol_score"]["label"]
    score_name=analysis[registry]["qol_score"]["name"]

    rc('font', size=fs)

    plot_types = analysis["plots"].get("aggregations",[])
    data_dir = os.path.join(os.path.dirname(__file__),"..","data","{r}").format(r=registry) 
    fig_dir = os.path.join(os.path.dirname(__file__),"..","figures","{r}", "result_plots").format(r=registry)    
    
    results_table_dict = pickle.load(open(os.path.join(data_dir,"results_tables.pkl"), "rb"))

    for plt_type in plot_types:
        if plt_type=="auc":
            continue
        pltdata = results_table_dict[plt_type]
        plot_data(pltdata, plt_type, fig_dir, analysis)

def plot_data(pltdata, plt_type, fig_dir, analysis):
    registry = analysis["registry"]
    setting = analysis[registry].get("setting", None)
    ymin = analysis[registry]["qol_score"]["min"]
    ymax = analysis[registry]["qol_score"]["max"]
    dead_as_zero = analysis.get("dead_as_zero", False)
    change_to_bl = analysis.get("change_to_bl", False)

    c_unadjusted = "tab:green"
    c_adjusted = "tab:orange"
    c_pmm = "tab:purple"
 
    pltdata = pltdata.reset_index()
    options = {
        "capsize": 5,
        "linestyle": "none",
        "linewidth": 1.25,
        "markersize" : 10 
        }

    ylabel = f"Rel. change in {score_label}" if change_to_bl else score_label
    
    delta = 0.15 

    if plt_type == "by_tseq_and_tseqmax":
        pltdata_full = pltdata.copy()
        plt.figure(figsize=(10,10))
        ax = plt.gca()
        for t in range(13):

            pltdata = pltdata_full.query("tseqmax==@t").set_index("tseq")

            xvals = np.array(pltdata.index.to_list())

            ax.errorbar(xvals-delta, pltdata.mean_aval.values, yerr=pltdata[["ci_lb_mean_aval_ac","ci_ub_mean_aval_ac"]].values.transpose(), label="answer", c=c_unadjusted, **options, marker="s")

            ax.errorbar(xvals, pltdata.mean_aval_na, yerr=pltdata[["ci_lb_mean_aval_na","ci_ub_mean_aval_na"]].values.transpose(), label="no answer (AIPW)")

            ax.errorbar(xvals+delta, pltdata.mean_aval_na_pmm, yerr=pltdata[["ci_lb_mean_aval_na_pmm", "ci_ub_mean_aval_na_pmm"]].values.transpose(), c=c_pmm, label = "no answer (PMM)", **options, marker="D")

        xticks = xvals
        xticklabels = xvals
        xlabel = "Questionnaire number"

    else:

        plt.figure(figsize=(10,10))

        xvals = np.array(pltdata.index.to_list())

        line1 = plt.errorbar(xvals-delta, pltdata.mean_aval_ac.values, yerr=pltdata[["ci_lb_mean_aval_ac","ci_ub_mean_aval_ac"]].values.transpose(), label="answer", c=c_unadjusted, **options, marker="s")
        ax = plt.gca()
        
        line2 = ax.errorbar(xvals+delta, pltdata.mean_aval_na, yerr=pltdata[["ci_lb_mean_aval_na","ci_ub_mean_aval_na"]].values.transpose(), label="no answer (AIPW)", c=c_adjusted,  **options, marker="o")

        line3 = ax.errorbar(xvals+2*delta, pltdata.mean_aval_na_pmm, yerr=pltdata[["ci_lb_mean_aval_na_pmm", "ci_ub_mean_aval_na_pmm"]].values.transpose(), c=c_pmm, label = "no answer (PMM)", **options, marker="D")

        if plt_type=="by_dtd":
            ax.invert_xaxis()
            xlabel = "Days before death"
            xticklabels = pltdata.dtd_binned 
            xticks = pltdata.index
            rot = 30
        elif plt_type=="by_tseq":
            xlabel = "Months post-baseline"
            xticklabels, xticks = get_timepoint_list(registry, setting)
            rot = 0
        elif plt_type=="by_dfs":
            xlabel= "Days since start of therapy"
            xticks = pltdata.index
            xticklabels = pltdata.dfs_binned

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=rot, horizontalalignment="right")
    ax.grid("y")
    ax.set_facecolor([1,1,1])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc="upper center", ncol=3)
    if plt_type=="by_tseq_and_tseqmax":
        ax.legend().remove()
    plt.axis(ymin=ymin, ymax=ymax)

    saveplot(f"mean_vs_weighted_mean_{plt_type}{'_dead_as_zero' if dead_as_zero else ''}_{score_name}", fig_dir)
    plt.show()
    plt.close()
   
    return 

def saveplot(fname, fig_dir):
    for f in ["png", "pdf", "svg"]:
        p = os.path.join(fig_dir,f)
        if not os.path.exists(p):
            os.makedirs(p)
        plt.savefig(os.path.join(p,fname+"."+f),bbox_inches='tight')
    return

if __name__=="__main__":    
    main()
     