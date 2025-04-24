import os
from pathlib import Path
from typing import List, Optional, Tuple
import einops
import librosa
from music2latent import EncoderDecoder
import numpy as np
import soundfile as sf
import math
import torch
import torchaudio
from x_transformers import ContinuousTransformerWrapper, Decoder
from jsonargparse import CLI
import csv
DIM_DATA = 64
PERIODE_MUSIC_2_LATENT = 4096 / 44100
class GMMIID:
    def __init__(self, n_modes = None, dim_out=None, dim_data=None, reduction = 'mean'):
        assert (n_modes is not None) ^  (dim_out is not None and dim_data is not None), 'Either n_modes or dim_out must be specified'
        if n_modes is None:
            n_modes = self.get_n_modes(dim_out, dim_data)
        self.n_modes = n_modes
        self.reduction = reduction


    def __call__(self, pred: torch.Tensor, target : torch.Tensor):
        return self.nlogp(pred, target, self.n_modes, self.reduction)

    @classmethod
    def unpack_out(cls, pred : torch.Tensor, n_modes : torch.Tensor = None, dim_data : torch.Tensor  = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim_out = pred.shape[-1]
        assert (n_modes is not None) ^  (dim_out is not None and dim_data is not None), 'Either n_modes or dim_out must be specified'
        if n_modes is None:
            n_modes = cls.get_n_modes(dim_out, dim_data)
        pred = einops.rearrange(pred, '... (n_modes d) -> ... n_modes d', n_modes=n_modes)
        D = (pred.shape[-1] - 1 )// 2
        means, std_sqrt, mixing_vars = pred[...,:D], pred[...,D:2*D], pred[...,-1].unsqueeze(-1)
        stds = torch.nn.functional.softplus(std_sqrt) ** .5
        mixing_logprobs = mixing_vars - torch.logsumexp(mixing_vars, dim=-2, keepdim=True)
        return means, stds, mixing_logprobs

    @classmethod
    def nlogp(cls, params: torch.Tensor, target : torch.Tensor, n_modes : int = 2, reduction = 'mean'):
        # extract means and covariance logits
        means, stds, mixing_logprobs = cls.unpack_out(params, n_modes)
        target = einops.repeat(target, '... d -> ... n_modes d', n_modes=n_modes)
        logprob = torch.distributions.Normal(means, stds).log_prob(target).sum(-1, keepdims=True)
        summed = mixing_logprobs + logprob
        nll = torch.logsumexp(summed, dim=-2).neg()[..., 0]
        if reduction == 'mean':
            nll = nll.mean()
        elif reduction != 'none':
            raise ValueError(f"Reduction {reduction} not understood")
        return nll

    @classmethod
    def logits_dim(cls, n_modes, out_dim):
        out_dim = 2 * out_dim * n_modes +  n_modes
        return out_dim
    
    @classmethod
    def get_n_modes(cls, dim_out: int, dim_data: int) -> int:
        n_modes = dim_out // (1+2*dim_data)
        return n_modes

class ContinuousTransformerModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out, max_seq_len, depth, heads, emb_dim = 256, attn_flash=True):
        super().__init__()
        self.model = ContinuousTransformerWrapper(
            emb_dropout=0.1,
            dim_in=dim_in,
            dim_out=dim_out,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(
                dim=emb_dim,
                depth=depth,
                heads=heads,
                rotary_pos_emb=True,
                attn_flash=attn_flash,
                attn_dropout=0.1,
                ff_dropout=0.1,
            )
        )

    def forward(self, x : torch.Tensor, mask : torch.Tensor, **kwargs):
        return self.model(x, mask=mask) # , x, 0

    def loss(self, mag):
        data_mask = torch.ones(mag.size(0), mag.size(1)).bool().to(mag.device)
        params = self(mag, mask=data_mask)
        loss_unreduced = GMMIID(dim_out=params.shape[-1], dim_data=mag.shape[-1], reduction='none')(params[:,:-1,:], mag[:,1:,:])
        return loss_unreduced


def from_ckpt(ckpt):
    """
    Load a pre-trained ContinuousTransformerModel from a checkpoint file.

    Args:
        ckpt (str): Path to the checkpoint file.

    Returns:
        ContinuousTransformerModel: An instance of the ContinuousTransformerModel
        loaded with the state dictionary from the checkpoint and set to evaluation mode.
    """
    ckpt_dict = torch.load(ckpt,map_location='cpu', weights_only=False)
    model_state_dict = ckpt_dict['model_state_dict']
    dim_out = model_state_dict['model.project_out.weight'].shape[0]
    max_seq_length = ckpt_dict['max_seq_length']
    model = ContinuousTransformerModel(
            dim_in=DIM_DATA,
            dim_out=dim_out,
            max_seq_len=max_seq_length,
            depth=12,
            heads=8,
            emb_dim=256
        )

    model.load_state_dict(model_state_dict)
    model = model.eval()
    return model

def compute_non_silence(audio, silence_extractor: str, fixed_sr: int):
    """
    Compute the non-silence regions of an audio signal.

    Args:
        audio (np.ndarray): The audio signal as a NumPy array.
        silence_extractor (str): Method to compute non-silence regions ('music' or 'voice').
        fixed_sr (int): The fixed sample rate of the audio signal.

    Returns:
        np.ndarray: An array of non-silence regions, where each row represents 
                    [start_index, end_index] of a non-silent segment.
    """
    if silence_extractor == "music":
        # Use librosa's split function to detect non-silent regions in music
        non_silence = librosa.effects.split(audio)
    elif silence_extractor == "voice":
        # Use torchaudio's Voice Activity Detection (VAD) for voice signals
        vad = torchaudio.transforms.Vad(sample_rate=fixed_sr)
        in_aud = torch.as_tensor(audio)
        # Calculate the silence at the beginning and end of the audio
        heading_silence = (audio.shape[-1] - vad(in_aud).shape[-1])
        trailing_silence = (vad(in_aud.flip([-1])).shape[-1])
        # Represent the non-silence region as a single range
        non_silence = np.array([[heading_silence, trailing_silence]])
    else:
        # Raise an error if the silence extractor type is not implemented
        raise NotImplementedError
    return non_silence

class ICCalcHelper:
    def __init__(
        self,
        device,
        ckpt: Optional[str] = None,
    ):
        """
        Helper class for calculating Information Content (IC).

        Args:
            device (str): The device to use for computation (e.g., 'cuda' or 'cpu').
            ckpt (Optional[str]): Path to the model checkpoint. If None, defaults to 'gmm.pt' in the library root.
        """
        if ckpt is None:
            filepath = os.path.abspath(__file__)
            lib_root = os.path.dirname(filepath)
            ckpt = lib_root + '/gmm.pt'
        # Load the model from the checkpoint and move it to the specified device
        self.model = from_ckpt(ckpt).to(device)
        self.device = device
        # Initialize the EncoderDecoder for encoding audio to latent representations
        self.encdec = EncoderDecoder(device=device)

    @torch.no_grad
    def ic(
        self,
        heading_nan_pad: int,
        trailing_nan_pad: int,
        len_wo_pad: int,
        latents_padded: torch.Tensor,
    ):
        """
        Calculate the negative log-likelihood (NLL) for the given latent representations.

        Args:
            heading_nan_pad (int): Number of NaN padding values at the start of the sequence.
            trailing_nan_pad (int): Number of NaN padding values at the end of the sequence.
            len_wo_pad (int): Length of the sequence without padding.
            latents_padded (torch.Tensor): Padded latent representations.

        Returns:
            np.ndarray: NLL values with NaN padding applied.
        """
        nll = self.model.loss(latents_padded)
        nll = nll[..., :len_wo_pad].to('cpu')
        nll = np.pad(
            nll,
            ((0, 0), (heading_nan_pad, trailing_nan_pad)),
            mode="constant",
            constant_values=np.nan,
        )
        return nll


    def encode_m2l(self, audio_file: str | np.ndarray, silence_extractor):
        """
        Encodes an audio file or array into latent representations, while handling silence and padding.

        Args:
            audio_file (str | np.ndarray): Path to the audio file or a NumPy array of audio samples.
            silence_extractor (str): Method to compute non-silence regions ('music' or 'voice').

        Returns:
            Tuple[int, int, int, torch.Tensor]: 
                - heading_nan_pad: Number of NaN padding values at the start.
                - trailing_nan_padding: Number of NaN padding values at the end.
                - len_wo_pad: Length of the sequence without padding.
                - latents_padded: Padded latent representations.
        """
        # If the input is a file path, read the audio file
        if isinstance(audio_file, str):
            audio, rate = sf.read(audio_file)
            # Resample the audio to 44100 Hz if it has a different sample rate
            if rate != 44100:
                audio_tensor = torch.from_numpy(audio)
                resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=44100)
                audio = resampler(audio_tensor)
        else:
            # If the input is already an array, assume a sample rate of 44100 Hz
            audio = audio_file
            rate = 44100

        # If the audio has multiple channels, convert it to mono by averaging
        if len(audio.shape) == 2:
            audio = audio.mean(axis=1)

        # Get the total number of audio samples
        audio_samples = len(audio)

        # Compute the non-silence regions based on the specified silence extractor
        non_silence = compute_non_silence(audio, silence_extractor, rate)
        heading_silence = non_silence[0][0]  # Silence at the beginning
        trailing_silence = non_silence[-1][1]  # Silence at the end

        # Calculate the number of NaN padding values for the start and end
        heading_nan_pad = math.floor(heading_silence / (rate * PERIODE_MUSIC_2_LATENT))
        trailing_nan_padding = math.ceil(
            (audio_samples - trailing_silence) / (rate * PERIODE_MUSIC_2_LATENT)
        )

        # Remove the silence from the audio
        audio = audio[heading_silence:trailing_silence]

        # Encode the audio into latent representations
        latents = self.encdec.encode(audio)
        latents = latents.float().permute(0, 2, 1)  # Adjust dimensions for processing

        # Get the length of the sequence without padding
        len_wo_pad = latents.shape[1]

        # Pad the latent representations to a fixed length
        latents_padded = torch.nn.functional.pad(
            latents, (0, 0, 0, 4800 - latents.shape[1])
        )

        # Return the padding information and the padded latent representations
        return heading_nan_pad, trailing_nan_padding, len_wo_pad, latents_padded

def calc_ic(
    audio_files: List[str] = ['Acoustic Grand Piano.wav'],
    audio_type: str = 'music',
    output_dir: str = './',
    device: str = 'cuda'
):
    """Calculate the Information Content (IC) for each audio file.
    
    Args:
        audio_files (List[str], optional): A list of audio file paths to process. 
            Defaults to ['Acoustic Grand Piano.wav'].
        audio_type (str, optional): The type of audio being processed (e.g., 'music'). 
            Defaults to 'music'.
        output_dir (str, optional): The directory where output files will be saved. 
            Defaults to './'.
        device (str, optional): The device to use for computation (e.g., 'cuda' or 'cpu'). 
            Defaults to 'cuda'.
    """
    ic_calc_helper = ICCalcHelper(device=device)
    for audio_file in audio_files:
        heading_nan_pad, trailing_nan_padding, len_wo_pad, latents_padded = ic_calc_helper.encode_m2l(audio_file, silence_extractor=audio_type)
        nll = ic_calc_helper.ic(heading_nan_pad, trailing_nan_padding, len_wo_pad, latents_padded)
        nll = nll[0]
        time = (np.arange(len(nll)) + 1) * PERIODE_MUSIC_2_LATENT
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_filename = output_dir.joinpath(f"{Path(audio_file).stem}.csv")
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time', 'IC'])
            writer.writerows(zip(time, nll))           
if __name__ == '__main__':
    CLI(calc_ic, as_positional=True)
    