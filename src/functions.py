from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd
from datetime import timedelta, date
from os.path import dirname, exists
from os import makedirs

def saveplot(path, formats=["pdf","png", "svg"]):
    for f in formats:
        path_fmt = path.format(fmt=f)
        if not exists(dirname(path_fmt)):
            makedirs(dirname(path_fmt))
        plt.savefig(path_fmt, bbox_inches="tight")

def format_number(x, n_digits=2):
    x = np.around(x, n_digits)
    x = str(x)
    splitted = x.split(".")
    
    if len(splitted)==2:    
        n = len(x.split(".")[1])    
    else:
        n = 0

    if n<n_digits:
        if n>0:
            x = x+(n_digits-n)*"0"
        else:
            x = x+"."+n_digits*"0"

    elif n>n_digits:
        if n_digits==0:
            x = x[:-(n-n_digits+1)]
        else:
            x = x[:-(n-n_digits)]
    
    return x

def numeric_sas_to_date(n):
    if pd.isna(n):
        return pd.NaT
    return date(1960, 1, 1)+timedelta(days=n)

def map_qnumber_to_month(k, registry, setting="palliative"):
    if registry == "pancreas":
        return 2*k
    elif registry == "mamma":
        if setting == "adjuvant":
            raise NotImplementedError("Not implemented for this registry/setting")
        elif setting == "palliative":
            if k<=4:
                return 3*k
            else:
                return 12+(k-4)*6

def get_timepoint_list(registry, setting=None):
    if registry == "pancreas":        
        return [2*k for k in range(13)], [k for k in range(13)]
    elif registry == "mamma":
        if setting.startswith("palliativ"):
            return [0, 3, 6, 9, 12]+[12+(k-4)*6 for k in range(5, 10)], [k for k in range(10)]
        if setting == "adjuvant":
            return [0,6,12,24,36,48,60], [k for k in range(7)]