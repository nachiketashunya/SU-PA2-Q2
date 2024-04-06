import os
from .utils import create_librimix

# eps secures log and division
EPS = 1e-10
# Rate of the sources in LibriSpeech
RATE = 16000

librispeech_dir = "/content/drive/MyDrive/Assignment2/LibriSpeech"
wham_dir = "/content/drive/MyDrive/Assignment2/wham_noise"
metadata_dir = "/content/drive/MyDrive/Assignment2"
librimix_outdir = "/content/drive/MyDrive/Assignment2/"

n_src = 2
freqs = ['8k', '16k']
modes = ['min', 'max']
types = ['mix_clean', 'mix_both', 'mix_single']

librimix_outdir = os.path.join(librimix_outdir, f'Libri{n_src}Mix')
# Get the desired frequencies
freqs = [freq.lower() for freq in freqs]
modes = [mode.lower() for mode in modes]
types = [t.lower() for t in types]

# Call librimix creation function
create_librimix(librispeech_dir, wham_dir, librimix_outdir, metadata_dir, freqs, n_src, modes, types)
