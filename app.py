import gradio as gr
import torch
from hyperpyyaml import load_hyperpyyaml 
import yaml
from speechbrain.inference.separation import SepformerSeparation 

import sys
sys.path.append("SU-PA2-Q2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def separate_audio(mixture):
    # Convert mixture to tensor
    print(mixture)
    encoder_checkpoint = "models/encoder.pth"
    decoder_checkpoint = "models/decoder.pth"
    masknet_checkpoint = "models/masknet.pth"

    encoder = torch.load(encoder_checkpoint, map_location=device)
    decoder = torch.load(decoder_checkpoint, map_location=device)
    masknet = torch.load(masknet_checkpoint, map_location=device)

    # Load model
    # Step 2: Load Hyperparameters

    data_folder = "data/processed/librimix/libri2mix/max"
    # hparams_file = "data/yamls/sepformer-customdataset.yaml"
    overrides = f"data_folder: {data_folder}\noutput_folder: reports/"
    hyperparams_file = "data/yamls/sepformer-customdataset.yaml"
    with open(hyperparams_file, "r") as f:
        hparams = load_hyperpyyaml(f, overrides)

    hparams['Encoder'].load_state_dict(encoder)
    hparams['Decoder'].load_state_dict(decoder)
    hparams['MaskNet'].load_state_dict(masknet)  #

    separator = SepformerSeparation(
        modules=hparams["modules"],
        hparams=hparams
    )

    _, mixture = torch.tensor(mixture)
    est_sources = separator.separate_batch(mixture)

    s1 = est_sources[:, :, 0].cpu()
    s2 = est_sources[:, :, 1].cpu()
    # Return separated sources
    return [(16000,s1), (16000,s2)]

# Define the audio input component
input_audio = gr.Audio(sources=["upload"], waveform_options=dict(waveform_color="#01C6FF"))

# Define the audio output components (one for each processed stream)
output_audio1 = gr.Audio(autoplay=False)
output_audio2 = gr.Audio(autoplay=False)

# Create the Gradio interface
interface = gr.Interface(
    fn=separate_audio,
    inputs=input_audio,
    outputs=[output_audio1, output_audio2],
    title="Source Separation"
)

interface.launch()