import torch 
from torch.utils.data import DataLoader, random_split
from ..data.make_dataset import LibriMixDataset
from speechbrain.inference.separation import SepformerSeparation as separator
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, SignalDistortionRatio

# Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "path"
csv_file = "path"

dataset = LibriMixDataset(data_dir, csv_file)

# Split in 7:3 
total_size = len(dataset)
train_size = int(0.7 * total_size)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr', run_opts={"device":"cuda"})

si_snr = ScaleInvariantSignalNoiseRatio()
sdr = SignalDistortionRatio()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

total_snr, total_sdr = 0.0, 0.0

with torch.no_grad():
  for mix, s1, s2 in test_loader:
    mix = mix.to(device)

    est_sources = model.separate_batch(mix)

    est_sources = est_sources.detach().cpu()
    
    s1_snr = si_snr(est_sources[:, :, 0], s1)
    s2_snr = si_snr(est_sources[:, :, 1], s2)

    s1_sdr = sdr(est_sources[:, :, 0], s1)
    s2_sdr = sdr(est_sources[:, :, 1], s2)

    total_snr += (s1_snr + s2_snr) 
    total_sdr += (s1_sdr + s2_sdr)

average_snr = total_snr / len(test_loader)
average_sdr = total_sdr / len(test_loader)

print(f"Average SNR: {average_snr:.4f}")
print(f"Average SDR: {average_sdr:.4f}")