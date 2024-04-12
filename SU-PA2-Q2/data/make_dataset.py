import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset

class LibriMixDataset(Dataset):
  def __init__(self, data_dir, csv_file, max_frames=32000):
    self.data_dir = data_dir
    self.df = pd.read_csv(csv_file)
    self.max_frames = max_frames
  
  def __len__(self):
    return len(self.data_dir)
  
  def __getitem__(self, n):
    row_info = self.df.iloc[n]

    mixture_audio, _ = torchaudio.load(row_info['mixture_path'])
    source1_audio, _ = torchaudio.load(row_info['source_1_path'])
    source2_audio, _ = torchaudio.load(row_info['source_2_path'])

    ret_mixture = self.fix_frames(mixture_audio, mixture_audio.shape[1])
    ret_source1 = self.fix_frames(source1_audio, source1_audio.shape[1])
    ret_source2 = self.fix_frames(source2_audio, source2_audio.shape[1])

    return ret_mixture, ret_source1, ret_source2
   
  def fix_frames(self, wav, num_frames):
    if num_frames < self.max_frames:
      padding_size = self.max_frames - num_frames
      wav = torch.nn.functional.pad(wav, (0, padding_size), value=0)
    elif num_frames > self.max_frames:
      wav = wav[:, :self.max_frames]

    wav = wav.squeeze(0)

    return wav
  