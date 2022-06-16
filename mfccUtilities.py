
import numpy as np
import librosa
import random
from scipy.io import wavfile
from os import listdir
from os.path import join
import soundfile as sf


random.seed(0)
np.random.seed(0)



def load_audio(file_path: str, scale: bool = True, target_sr : int = 22050) -> tuple[np.ndarray, int]:
    """load the audio file.

    :param file_path: the audio file path.
    :type file_path: str
    :return: returns the audio and the sampling rate used to load it.
    :rtype: tuple[np.ndarray, int]
    """
    if scale:
        # by default librosa normalizes the data.
        audio, sr = librosa.load(file_path)
    else:
        # wavfile doesn't normalize the data.
        try:
            sr, audio = wavfile.read(file_path)
        except ValueError as e:
            y, s = librosa.load(file_path)
            sf.write(file_path, y, s, "PCM_24")
        
        sr, audio = wavfile.read(file_path)
        audio = audio.astype(float)

        if audio.ndim == 2:
            audio = np.mean(audio, axis = 1)

        audio, sr = librosa.resample(audio, sr, target_sr), target_sr

    return audio, sr


def compute_mfcc(aud: tuple[np.ndarray, int], n_mfcc: int = 12) -> np.ndarray:
    """Calculate the mfcc matrix.

    :param aud: the audio and the sampling rate.
    :type aud: tuple[np.ndarray, int]
    :param n_mfcc: the number of mfcc windows to calculate, defaults to 12.
    :type n_mfcc: int, optional
    :return: returns the mfcc matrix.
    :rtype: np.ndarray
    """
    audio, sr = aud
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, hop_length=1024, htk=True, n_mfcc = n_mfcc)
    return mfcc


def compute_mfcc_mean(mfcc: np.ndarray) -> np.ndarray:
    """Compute the mean of the mfcc relative to each window.

    :param mfcc: the mfcc matrix.
    :type mfcc: np.ndarray
    :return: returns a vector of means.
    :rtype: np.ndarray
    """
    mfcc_mean = np.mean(mfcc, axis = 1)
    return mfcc_mean


def compute_mfcc_std(mfcc: np.ndarray) -> np.ndarray:
    """Compute the standard deviation relative to each mfcc window.

    :param mfcc: the mfcc matrix.
    :type mfcc: np.ndarray
    :return: returns a list of standard deviations.
    :rtype: np.ndarray
    """
    mfcc_std = np.std(mfcc, axis = 1)
    return mfcc_std
