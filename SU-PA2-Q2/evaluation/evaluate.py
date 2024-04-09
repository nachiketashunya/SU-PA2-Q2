import torch 
from torch.utils.data import DataLoader, random_split
from ..data.make_dataset import LibriMixDataset
from speechbrain.inference.separation import SepformerSeparation as separator
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, SignalDistortionRatio
import wandb 
from tqdm import tqdm

# Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "/data/processed/Libri2mix/wav8k/max/test/mix_both"
csv_file = "/data/processed/Libri2Mix/wav8k/max/metadata/mixture_test_mix_both.csv"

dataset = LibriMixDataset(data_dir, csv_file)

# Split in 7:3 
total_size = len(dataset)
train_size = int(0.7 * total_size)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Load the model
model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr', run_opts={"device": torch.device})

# SNR and SDR instance
si_snr = ScaleInvariantSignalNoiseRatio()
sdr = SignalDistortionRatio()

# Initialize wandb
wandb.init(project="Sepformer", name="Eval on test set")

def print_intermediate(index, s1_snr, s2_snr, s1_sdr, s2_sdr):
    print(f"\nBatch {index}:")
    print(f"\tS1 SNR: {s1_snr:.4f}\tS2 SNR: {s2_snr:.4f}")
    print(f"\tS1 SDR: {s1_sdr:.4f}\tS2 SDR: {s2_sdr:.4f}")

total_snr, total_sdr = 0.0, 0.0
model = model.to(device)

with torch.no_grad():
    for index, (mix, s1, s2) in enumerate(tqdm(test_loader, desc="Processing batches")):
        mix = mix.to(device)

        est_sources = model.separate_batch(mix)

        est_sources = est_sources.detach().cpu()

        s1_snr = si_snr(est_sources[:, :, 0], s1).item()
        s2_snr = si_snr(est_sources[:, :, 1], s2).item()

        s1_sdr = sdr(est_sources[:, :, 0], s1).item()
        s2_sdr = sdr(est_sources[:, :, 1], s2).item()

        # Log intermediate results to WandB
        wandb.log({
            f'S1 SNR': s1_snr,
            f'S2 SNR': s2_snr,
            f'S1 SDR': s1_sdr,
            f'S2 SDR': s2_sdr
        })

        total_snr += (s1_snr + s2_snr)
        total_sdr += (s1_sdr + s2_sdr)

        # Print intermediate results
        print_intermediate(index, s1_snr, s2_snr, s1_sdr, s2_sdr)

average_snr = total_snr / len(test_loader)
average_sdr = total_sdr / len(test_loader)

# Log final results to WandB
wandb.log({
    'Average SNR': average_snr,
    'Average SDR': average_sdr
})

print("\nFinal Results:")
print(f"Average SNR: {average_snr:.4f}")
print(f"Average SDR: {average_sdr:.4f}")

wandb.finish()