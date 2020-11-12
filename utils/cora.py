import torch
import ipdb
from utils import DotDict
from data.datasets import DiscriminativeCoraDataset
from utils.models import get_model_params


def load_datasets(hparams, CFG, splits, high_level):
    if not high_level:
        bag_file = CFG["rep"]["cora"]["tracks"]
    else:
        bag_file = CFG["rep"]["cora"]["highlevelclasses"]
    datasets = []
    common_hparams = {
        "datadir": CFG["gpudatadir"],
        "bag_file": bag_file,
        "paper_file": CFG["rep"]["cora"]["papers"]["emb"],
        "ft_type": hparams.ft_type,
        "subsample": 0,
        "high_level": high_level,
        "load": hparams.load_dataset == "True"
    }
    for split in splits:
        datasets.append(DiscriminativeCoraDataset(**common_hparams, split=split))

    return datasets


def collate_for_disc_spe_model_cora(batch):
    ids = [i[0] for i in batch]
    avg_profiles = [torch.from_numpy(i[1]) for i in batch]
    sent_emb = [torch.from_numpy(i[2]) for i in batch]
    labels = [i[3] for i in batch]
    return ids, torch.stack(avg_profiles), sent_emb, torch.LongTensor(labels), batch[0][-1]


def init_model(hparams, dataset, datadir, xp_title, model_class):
    in_size, out_size = get_model_params(hparams, 300, len(dataset.track_rep))
    arguments = {'in_size': in_size,
                 'out_size': out_size,
                 'hparams': hparams,
                 'datadir': datadir,
                 'desc': xp_title,
                 "num_tracks": len(dataset.track_rep),
                 "input_type": hparams.input_type,
                 "ft_type": hparams.ft_type,
                 "optim": hparams.optim,
                 'init_type': hparams.init_type}

    print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = model_class(**arguments)
    print("Model Loaded.")
    return model


def xp_title_from_params(hparams):
    string = hparams.model_type
    if hparams.high_level_classes == "True":
        string += "_HL"
    string += '_' + hparams.init_type + "_" + hparams.optim
    if type(hparams) == DotDict:
        if "init" in hparams.keys():
            if hparams.init == "True":
                string += "_init"
        if 'frozen' in hparams.keys():
            if hparams.frozen == "True":
                string += "_frozen"
    else:
        if hasattr(hparams, 'init'):
            if hparams.init == "True":
                string += "_init"
        if hasattr(hparams, 'frozen'):
            if hparams.frozen == "True":
                string += "_frozen"
    string += "_" + hparams.ft_type + "_" + hparams.input_type + "_bs" + str(hparams.b_size)
    string += "_" + str(hparams.lr) + '_' + str(hparams.wd)
    return string
