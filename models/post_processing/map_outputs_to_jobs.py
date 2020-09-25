import pickle as pkl
import yaml
import numpy as np
import os
import fastText
from tqdm import tqdm
import torch

def main():
    global CFG
    with open("../../config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    rep_type = "ft"
    model_type = "disc_spe"
    bag_type = "cie"
    description = "disc_spe_cie_ft_avg_bagTransformer_bs64_TEST"
    tgt_file = os.path.join(CFG["datadir"], "OUTPUTS_" + description + ".pkl")
    with open(tgt_file, 'rb') as f:
        model_outputs = pkl.load(f)
    wc_file = os.path.join(CFG["datadir"], "OUTPUTS_well_classified_" + description)
    with open(wc_file + ".pkl", 'rb') as f:
        wc_test = pkl.load(f)
    lookup_file = os.path.join(CFG["datadir"], "profiles_jobs_skills_TEST.pkl")
    with open(lookup_file, 'rb') as f:
        all_people = pkl.load(f)
    ft_model = fastText.load_model(os.path.join(CFG["modeldir"], "ft_fs.bin"))
    lookup = {}
    for person in tqdm(all_people):
        lookup[person[0]] = {"skills": person[1],
                             "industry": person[2],
                             "jobs": person[3]}
    indices_to_pred_and_labels = {}
    for num, index in enumerate(model_outputs["indices"]):
        if index in wc_test["indices"]:
            indices_to_pred_and_labels[index] = {"pred": model_outputs["preds"][num].detach().numpy(),
                                                 'label': model_outputs["labels"][num].item()}
    selected_people = {}
    for index in wc_test["indices"]:
        selected_people[index] = {**lookup[index], **indices_to_pred_and_labels[index]}
    jobs_and_emb = {}
    for person in tqdm(selected_people.keys()):
        jobs_and_emb[person] = selected_people[person]
        jobs_and_emb[person]["job_emb"] = []
        for job in selected_people[person]["jobs"]:
            job_emb = np.zeros(300)
            for word in job["job"]:
                job_emb = np.concatenate((job_emb, ft_model.get_word_vector(word)), axis=-1)
            jobs_and_emb[person]["job_emb"].append(np.mean(np.reshape(job_emb, (-1, 300))[1:]))
    del ft_model
    file = os.path.join(CFG["datadir"], "jobs_and_emb_wc_" + bag_type + "_" + rep_type + "_TEST.pkl")
    with open(file, "wb") as f:
        torch.save(jobs_and_emb, f)

