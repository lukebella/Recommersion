# Recommersion
**Recommersion** is a Recommender System software act to suggest songs (taken from [DEAM](https://cvml.unige.ch/databases/DEAM/) and [PMEmo](https://dl.acm.org/doi/10.1145/3206025.3206037) datasets) based on emotional dimensional values (valence and arousal), captured by your speech or adjusted via sliders. SER models used are [Model for Dimensional Speech Emotion Recognition based on Wav2vec 2.0 by Audeering](https://zenodo.org/records/6221127) and a custom one. The last one was realized by fine-tuning Wav2vec2-base on a combination of [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_info.htm) and [MuSe-CAR](https://zenodo.org/records/4134758) datasets.

## GUI

## Overview

The Multimodal SER Model leverages both textual and acoustic features to classify emotions more accurately. The architecture consists of:

- **Initial Classification**: Each model (MLP 1, respectively MLP 2) individually classifies the emotion based on their respective features (extracted with Wav2Vec2, respectively DebertaV3).
- **Fusion and Final Classification**: The extracted features and initial predictions are combined using a Multi-Layer Perceptron (MLP 3) to provide the final emotion classification.

A research report detailing the development and evaluation of this architecture can be found at [research-raport.pdf](./research-raport.pdf).

<img src="./architecture.png" alt="Model Architecture" width="100%">

## Setup

1. Clone this repository.
2. Create a virtual python `3.10` environment.
3. Set the python packet manager to version `23.3.1`, using:
   ```bash
   $ pip upgrade --install pip==23.3.1
   ```
4. Install the imported libraries using:
   ```bash
   $ pip install requirements.txt
   ```

## Datasets

The datasets should be downloaded upon academic requested from:

- [MuSe-CAR](https://zenodo.org/records/4134758) 
- [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.