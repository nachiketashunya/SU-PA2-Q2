import os
from utils import create_librimix

# eps secures log and division
EPS = 1e-10
# Rate of the sources in LibriSpeech
RATE = 16000

DATASET_DIR = "data/processed/"
librispeech_dir = os.path.join(DATASET_DIR, "LibriSpeech")
wham_dir = os.path.join(DATASET_DIR, "wham_noise")
metadata_dir = os.path.join("data/metadata")

n_src = 2
freqs = ['8k']
modes = ['max']
types = ['mix_both']

librimix_outdir = os.path.join(DATASET_DIR, f'Libri{n_src}Mix')

# Get the desired frequencies
freqs = [freq.lower() for freq in freqs]
modes = [mode.lower() for mode in modes]
types = [t.lower() for t in types]

# Call librimix creation function
create_librimix(librispeech_dir, wham_dir, librimix_outdir, metadata_dir, freqs, n_src, modes, types)
