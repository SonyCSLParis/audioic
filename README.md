# AudioIC
This repository is the official implementation of the ICASSP 2025 paper ["Estimating Musical Surprisal in Audio"](https://arxiv.org/abs/2501.07474), retrained on open data. Below is the abstract of the paper:

> **Abstract**  
> In modeling musical surprisal expectancy with computational methods, it has been proposed to use the information content (IC) of one-step predictions from an autoregressive model as a proxy for surprisal in symbolic music. With an appropriately chosen model, the IC of musical events has been shown to correlate with human perception of surprise and complexity aspects, including tonal and rhythmic complexity. This work investigates whether an analogous methodology can be applied to music audio. We train an autoregressive Transformer model to predict compressed latent audio representations of a pretrained autoencoder network. We verify learning effects by estimating the decrease in IC with repetitions. We investigate the mean IC of musical segment types (e.g., A or B) and find that segment types appearing later in a piece have a higher IC than earlier ones on average. We investigate the IC's relation to audio and musical features and find it correlated with timbral variations and loudness and, to a lesser extent, dissonance, rhythmic complexity, and onset density related to audio and musical features. Finally, we investigate if the IC can predict EEG responses to songs and thus model humans' surprisal in music.

AudioIC provides tools for calculating the *information content* (IC) as a proxy for human experienced surprise when listening to music. It includes a command line tool and python classes for calculating IC.


## Installation
You can install the package using pip with or without the extra dependencies required for the demo.

### Install the package for general use:
```bash
pip install audioic
```

### Install the package with demo dependencies:
```bash
git clone https://github.com/sonycslparis/audioic.git
cd audioic
pip install .[demo]
```

## Usage

### Running the `audio_ic` Command-Line Tool

The [`audio_ic`](./audio_ic.py) command-line tool allows you to compute the *information content* (IC) of audio files. To use it, specify the audio files you want to process and provide an output directory where the results will be saved as CSV files:

```bash
python -m audio_ic --audio_files "['<audio-file1>', '<audio-file2>', ...]" --output_dir <output-dir> --device "cpu"
```

Replace `<audio-file1>`, `<audio-file2>`, etc., with the paths to your audio files, and `<output-dir>` with the directory where you want the output files to be stored.


To run the tool on a GPU (default), specify the `--device` argument as `"cuda"`:

```bash
CUDA_VISIBLE_DEVICES=<device-id> python -m audio_ic --audio_files "['<audio-file1>', '<audio-file2>', ...]" --output_dir <output-dir> --device "cuda"
```
Replace `<device-id>` by a cuda device id.


### Using the AudioIC programmatically
The [`demo.ipynb`](./demo.ipynb) notebook demonstrates how to use the library programmatically to calculate and visualize the IC of audio files.

## Citation
If you use this project in your research, please cite the following paper:

```bibtex
@INPROCEEDINGS{10890619,
    author={Bjare, Mathias Rose and Cantisani, Giorgia and Lattner, Stefan and Widmer, Gerhard},
    booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
    title={Estimating Musical Surprisal in Audio}, 
    year={2025},
    volume={},
    number={},
    pages={1-5},
    keywords={Computational modeling;Music;Predictive models;Signal processing;Brain modeling;Transformers;Electroencephalography;Complexity theory;Integrated circuit modeling;Speech processing;Music information retrieval;Musical surprisal;Perceptual models;Neural networks},
    doi={10.1109/ICASSP49660.2025.10890619}}

```

## License
This project is licensed under the CC BY-NC 4.0 License.
