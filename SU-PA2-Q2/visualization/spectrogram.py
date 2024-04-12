import pandas as pd
import matplotlib.pyplot as plt
import librosa
import numpy as np

df = pd.read_csv("data/processed/Libri2Mix/wav8k/max/metadata/mixture_test_mix_both.csv")

mixture = df.iloc[5]['mixture_path']
source1 = df.iloc[5]['source_1_path']
source2 = df.iloc[5]['source_2_path']

def create_spectrogram(path, title):
  y, sr = librosa.load(path)

  # Generate spectrogram
  spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

  # Convert to decibels (log scale)
  spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

  # Plot spectrogram
  plt.figure(figsize=(7, 3))
  librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
  plt.colorbar(format='%+2.0f dB')
  plt.title(f'Mel Spectrogram - {title}')
  plt.xlabel('Time')
  plt.ylabel('Frequency (Hz)')
  plt.tight_layout()
  plt.show()


create_spectrogram(mixture, title="Mixture")
create_spectrogram(source1, title="Source 1")
create_spectrogram(source2, title="Source 2")
