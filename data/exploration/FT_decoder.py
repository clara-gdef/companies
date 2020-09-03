import os
import pickle as pkl
import yaml
import fastText
import ipdb


def main():
    with ipdb.launch_ipdb_on_exception():
        print("Loading data...")
        with open(os.path.join(CFG["gpudatadir"], "lookup_ppl.pkl"), 'rb') as f_name:
            ppl_lookup = pkl.load(f_name)
        ft_model = fastText.load_model(os.path.join(CFG["modeldir"], "esann2020/ft_from_scratch.bin"))
        ipdb.set_trace()


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    main()
