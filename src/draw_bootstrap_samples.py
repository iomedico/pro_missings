import numpy as np
import os
import pickle
import yaml

def main():
    params_path = os.path.join(os.path.dirname(__file__),"..","params.yaml")
    analysis = yaml.load(open(params_path), Loader=yaml.FullLoader)["analysis"] 
    bootstrap = analysis.get("bootstrap", False)
    registry = analysis.get("registry","pancreas")    
    data_dir = os.path.join(os.path.dirname(__file__),"..","data","{r}").format(r=registry)
    n_bs_samples = analysis.get("n_bootstrap_samples", 200)
    
    reg_data = pickle.load(open(os.path.join(data_dir,"regression_data.pkl"), "rb"))
    out_path = os.path.join(data_dir, "bootstrap_samples.pkl")

    X = reg_data["predictors"].reset_index().set_index("pat_no")
    X_pmm = reg_data["predictors_pmm"].reset_index().set_index("pat_no")

    pat_nos = X.index.unique()

    samples = []
    bs_pat_samples = []
    samples_pmm = []

    rs = np.random.RandomState(23)

    if bootstrap:
        s = 0
        while s < n_bs_samples:
            pat_sample = rs.choice(pat_nos, replace=True, size=len(pat_nos))
            bs_pat_samples += [pat_sample]
            samples += [X.loc[pat_sample].obs_no.to_list()]
            samples_pmm += [X_pmm.loc[pat_sample].obs_no.to_list()]
            s+=1

    sample_dict = {
        "pat_nos": pat_nos,
        "bs_samples" : samples, 
        "bs_pat_samples" : bs_pat_samples,
        "bs_samples_pmm" : samples_pmm}

    with open(out_path, "wb") as file:
        pickle.dump(sample_dict, file)

if __name__=="__main__":
    main()
