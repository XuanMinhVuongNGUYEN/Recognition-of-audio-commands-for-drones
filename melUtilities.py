import random
import torch
import torchaudio
from torchaudio import transforms
from audiomentations import Compose, Trim, AddGaussianNoise, Gain


def open(file_path: str) -> tuple[torch.Tensor, int]:
    """Load an audio file.

    :param file_path: the audio file path.
    :type file_path: str
    :return: Return the signal as a tensor and the sample rate
    :rtype: tuple[torch.Tensor, int]
    """
    sig, sr = torchaudio.load(file_path)
    return (sig, sr)


def rechannel(aud: tuple[torch.Tensor, int], new_channel: int) -> tuple[torch.Tensor, int]:
    """Convert the given audio to the desired number of channels

    :param aud: the audio and the sampling rate.
    :type aud: tuple[torch.Tensor, int]
    :param new_channel: The number of channels to use => {1 or 2}
    :type new_channel: int
    :return: The audio with the desired number of channels and the sampling rate.
    :rtype: tuple[torch.Tensor, int]
    """
    sig, sr = aud

    if (sig.shape[0] == new_channel):
        # Nothing to do
        return aud

    if (new_channel == 1):
        # Convert from stereo to mono by selecting only the first channel
        resig = sig[:1, :]
    else:
        # Convert from mono to stereo by duplicating the first channel
        resig = torch.cat([sig[:1, :], sig[:1, :]])

    return (resig, sr)


def trim(aud: tuple[torch.Tensor, int]) -> tuple[torch.Tensor, int]:
    """Delete silence from beginning and end of the audio

    :param aud: the audio and the sampling rate.
    :type aud: tuple[torch.Tensor, int]
    :return: the trimmed audio.
    :rtype: tuple[torch.Tensor, int]
    """
    sig, sr = aud
    sig = Trim().apply(sig, sr)
    return sig, sr


def add_noise(aud: tuple[torch.Tensor, int]) -> tuple[torch.Tensor, int]:
    """Add gaussian noise to the audio file

    :param aud: the audio and the sampling rate.
    :type aud: tuple[torch.Tensor, int]
    :return: the audio with noise.
    :rtype: tuple[torch.Tensor, int]
    """
    sig, sr = aud
    noise_adder = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude= 0.02)])
    sig = noise_adder(sig, sr)
    return sig, sr


def change_volume(aud: tuple[torch.Tensor, int]) -> tuple[torch.Tensor, int]:
    """Increase or decrease volume

    :param aud: the audio and the sampling rate.
    :type aud: tuple[torch.Tensor, int]
    :return: the audio after volume change.
    :rtype: tuple[torch.Tensor, int]
    """
    sig, sr = aud
    volume_adapter = Compose([Gain()])
    sig = volume_adapter(sig, sr)
    return sig, sr


def resample(aud: tuple[torch.Tensor, int], newsr: int) -> tuple[torch.Tensor, int]:
    """Since Resample applies to a single channel, we resample one channel at a time

    :param aud: the audio and the sampling rate.
    :type aud: tuple[torch.Tensor, int]
    :param newsr: the new sampling rate.
    :type newsr: int
    :return: the audio loaded with the new sampling rate.
    :rtype: tuple[torch.Tensor, int]
    """
    sig, sr = aud

    if (sr == newsr):
        # Nothing to do
        return aud

    num_channels = sig.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    if (num_channels > 1):
        # Resample the second channel and merge both channels
        retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
        resig = torch.cat([resig, retwo])

    return ((resig, newsr))


def pad_trunc(aud: tuple[torch.Tensor, int], max_ms: int) -> tuple[torch.Tensor, int]:
    """Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds

    :param aud: the audio and the sampling rate.
    :type aud: tuple[torch.Tensor, int]
    :param max_ms: fixed length in milliseconds.
    :type max_ms: int
    :return: Truncated audio.
    :rtype: tuple[torch.Tensor, int]
    """
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
        # Truncate the signal to the given length
        sig = sig[:,:max_len]

    elif (sig_len < max_len):
        # Length of padding to add at the beginning and end of the signal
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len

        # Pad with 0s
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))

        sig = torch.cat((pad_begin, sig, pad_end), 1)

    return (sig, sr)


def time_shift(aud: tuple[torch.Tensor, int], shift_limit:float) -> tuple[torch.Tensor, int]:
    """Shifts the signal to the left or right by some percent. Values at the end are 'wrapped around' to the start of the transformed signal.

    :param aud: the audio and the sampling rate.
    :type aud: tuple[torch.Tensor, int]
    :param shift_limit: to which extent the audio can be shifted, takes values within [0,1]
    :type shift_limit: float
    :return: audio after time shifting.
    :rtype: tuple[torch.Tensor, int]
    """
    sig, sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)


def spectro_gram(aud: tuple[torch.Tensor, int], n_mels: int=64, n_fft: int=1024, hop_len=None) -> torch.Tensor:
    """Generate a Spectrogram

    :param aud: the audio and the sampling rate.
    :type aud: tuple[torch.Tensor, int]
    :param n_mels: number of mels, defaults to 64
    :type n_mels: int, optional
    :param n_fft: size of FFT, creates n_fft // 2 + 1 bins, defaults to 1024
    :type n_fft: int, optional
    :param hop_len: _, defaults to None
    :type hop_len: _type_, optional
    :return: The Mel Spectrogram.
    :rtype: torch.Tensor
    """
    sig,sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)

def spectro_augment(spec: torch.Tensor, max_mask_pct: float = 0.1, n_freq_masks: int = 1, n_time_masks: int = 1) -> torch.Tensor:
    """Augment the Spectrogram by masking out some sections of it in both the frequency
    dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    overfitting and to help the model generalise better. The masked sections are
    replaced with the mean value.

    :param spec: The Mel Spectrogram.
    :type spec: torch.Tensor
    :param max_mask_pct: The maximum percent susceptible to being masked, defaults to 0.1
    :type max_mask_pct: float, optional
    :param n_freq_masks: Number of frequency masks, defaults to 1
    :type n_freq_masks: int, optional
    :param n_time_masks: Number of time masks, defaults to 1
    :type n_time_masks: int, optional
    :return: Masked Mel Spectrogram.
    :rtype: torch.Tensor
    """
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
        aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec