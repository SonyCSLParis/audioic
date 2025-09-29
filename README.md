# AudioIC
AudioIC provides tools for calculating the *information content* (IC) as a proxy for human experienced surprise when listening to music.  This repo is the official implementation of:
- "Estimating Musical Surprisal from Audio in Autoregressive Diffusion Model Noise Spaces". **Code and documentation is coming soon**. 
- ["Estimating Musical Surprisal in Audio"](https://arxiv.org/abs/2501.07474), retrained on open data. 

AudioIC includes a command line tool and python classes for calculating IC using a GIVT model.


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
pip install ".[demo]"
```

## Usage

### Running the `audioic` Command-Line Tool

The [`audioic`](./audioic/audioic.py) command-line tool allows you to compute the *information content* (IC) of audio files. To use it, specify the audio files you want to process and provide an output directory where the results will be saved as CSV files:

```bash
python -m audioic.audioic --audio_files "['<audio-file1>', '<audio-file2>', ...]" --output_dir <output-dir> --device "cpu"
```

Replace `<audio-file1>`, `<audio-file2>`, etc., with the paths to your audio files, and `<output-dir>` with the directory where you want the output files to be stored.


To run the tool on a GPU (default), specify the `--device` argument as `"cuda"`:

```bash
CUDA_VISIBLE_DEVICES=<device-id> python -m audioic --audio_files "['<audio-file1>', '<audio-file2>', ...]" --output_dir <output-dir> --device "cuda"
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

To obtain a commercial license, please contact [music@csl.sony.fr](mailto:music@csl.sony.fr).

