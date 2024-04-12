import os
import torch
import torch.nn.functional as F
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import logging
from speechbrain.core import AMPConfig
from hyperpyyaml import load_hyperpyyaml
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb

# Import necessary classes and functions
import sys 
sys.path.append("SU-PA2-Q2")
from training.models import Separation, dataio_prep
from data.prepare_csv import prepare_wsjmix



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_FOLDER = "data/processed/Libri2Mix"

 # Load hyperparameters file with command-line overrides
hparams_file = "data/yamls/sepformer-customdataset.yaml"
overrides = f"data_folder: {DATA_FOLDER}\noutput_folder: reports"

with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, overrides)

# Logger info
logger = logging.getLogger(__name__)

# Create experiment directory
sb.create_experiment_directory(
    experiment_directory=hparams["output_folder"],
    hyperparams_to_save=hparams_file,
    overrides=overrides,
)

# Update precision to bf16 if the device is CPU and precision is fp16
if torch.device == "cpu" and hparams.get("precision") == "fp16":
    hparams["precision"] = "bf16"

# Check if wsj0_tr is set with dynamic mixing
if hparams["dynamic_mixing"] and not os.path.exists(hparams["base_folder_dm"]):
    raise ValueError(
        "Please, specify a valid base_folder_dm folder when using dynamic mixing"
    )

run_on_main(
    prepare_wsjmix,
    kwargs={
        "datapath": hparams["data_folder"],
        "savepath": hparams["save_folder"],
        "n_spks": hparams["num_spks"],
        "skip_prep": hparams["skip_prep"],
        "fs": hparams["sample_rate"],
    },
)

# Load pretrained model if pretrained_separator is present in the yaml
if "pretrained_separator" in hparams:
    run_on_main(hparams["pretrained_separator"].collect_files)
    hparams["pretrained_separator"].load_collected()

if torch.cuda.is_available():
    run_opts = {'device': 'cuda'}
else:
    run_opts = {'device': 'cpu'}

# Brain class initialization
separator = Separation(
    modules=hparams["modules"],
    opt_class=hparams["optimizer"],
    run_opts=run_opts,
    hparams=hparams,
    checkpointer=hparams["checkpointer"],
)

# re-initialize the parameters if we don't use a pretrained model
if "pretrained_separator" not in hparams:
    for module in separator.modules.values():
        separator.reset_layer_recursively(module)

df = pd.read_csv("reports/save/custom_data.csv")
train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)
train_data.to_csv('reports/save/custom_train.csv', index=False)
test_data.to_csv('reports/save/custom_test.csv', index=False)

train_data, test_data = dataio_prep(hparams)


# Initialize wandb
wandb.init(project="Sepformer", name="Fine Tuning")

# Training
separator.fit(
    separator.hparams.epoch_counter,
    train_data,
    train_loader_kwargs=hparams["dataloader_opts"],
    valid_loader_kwargs=hparams["dataloader_opts"],
)


# Eval
avg_test_loss = separator.evaluate(test_data, min_key="si-snr")

# Log final results to WandB
wandb.log({
    'Average SNR': avg_test_loss,
})

separator.save_results(test_data)

wandb.finish()