import torch

from data.datasets import DiscriminativeCoraDataset


def load_datasets(hparams, CFG, splits, high_level):
    if high_level:
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