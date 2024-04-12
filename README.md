# SU-PA2-Q2

## Speech Assignment Question 2 Repository

Welcome to the repository for Question 2 of the Speech Assignment!

### Instructions for Performing Tasks

Follow these steps to perform various tasks:

1. **Setup LibriSpeech Test Clean Dataset and Wham Noise:**
   - Run the `./setup.sh` file to acquire the LibriSpeech Test Clean Dataset and Wham Noise.
   - The data will be stored under the folder `data/processed`.

2. **Create Librimix Dataset:**
   - Execute the `create_librimix.py` script.
   - This process will generate the Librimix dataset and save it into `data/processed`.
   - Metadata will be stored in `data/metadata`.
   - Note: The dataset will be generated for 2 Speakers with an 8k sampling rate.
   - To display spectrograms - Execute `SU-PA2-Q2/visualization/spectrogram.py`

3. **Evaluation of Sepformer on Test Dataset:**
   - Run `SU-PA2-Q2/evaluation/evaluate.py`.

    *Evaluation Metrics:*
    - Scale-Invariant Signal-to-Noise Ratio (SI-SNR) and Signal Distortion Ratio (SDR) are computed for each separated source in the mixture.
    - These metrics quantify the quality of the separated sources compared to the ground truth.

    *Logging and Reporting:*
    - WandB (Weights & Biases) is used for experiment tracking.
    - Intermediate results (SNR and SDR for each source) are logged during evaluation.
    - Final average SNR and SDR values are calculated and reported.

4. **Fine Tuning of Sepformer:**
    - Model definition is provided in `SU-PA2-Q2/training/models.py`.
    - Execute `SU-PA2-Q2/training/train.py`.
    - After training, the model is evaluated on the test data.
    - Evaluation metrics, such as SI-SNR (Scale-Invariant Signal-to-Noise Ratio), are computed.
    - Results are saved for further analysis.

==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── README.md                               <- The top-level README for developers using this project.
    ├── data
    │   ├── processed                           <- The final, canonical data sets for modeling.
    │   └── raw                                 <- The original, immutable data dump.
    │
    ├── models                                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                               <- Jupyter notebooks.
    │
    ├── reports                                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt                        <- The requirements file for reproducing the analysis environment, e.g.
    │                                              generated with `pip freeze > requirements.txt`
    │
    ├── SU-PA2-Q2           <- Source code for use in this project.
    │   │
    │   ├── data                                <- Scripts to download or generate data
    │   │
    │   ├── utils                               <- Scripts utilities used during data generation or training
    │   │
    │   ├── training                            <- Scripts to train models
    │   │
    │   ├── validate                            <- Scripts to validate models
    |__
