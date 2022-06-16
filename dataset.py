import umap
from pyexpat import model
from torch.utils.data import Dataset
from melUtilities import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# Sound Dataset
# ----------------------------
class DroneDS(Dataset):
    def __init__(self, df, train, nb_channels = 2, mode_reduc = ['PCA','TSNE','UMAP']):
        self.df = df
        self.duration = 4000
        self.sr = 44100
        self.train = train
        self.channel = nb_channels
        self.mode_reduc = mode_reduc
        self.shift_pct = 0.4

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)    

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        fpath = self.df.filepath.iloc[idx]
        # Get the Class ID
        labelId = self.df.labelId.iloc[idx]

        aud = open(fpath)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same 
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.

        if self.train:
            reaud = resample(aud, self.sr)
            t_aud = trim(reaud)
            gain_aud = change_volume(t_aud)
            noise_aud = add_noise(gain_aud)
            rechan = rechannel(noise_aud, self.channel)
            dur_aud = pad_trunc(rechan, self.duration)
            shift_aud = time_shift(dur_aud, self.shift_pct)
            sgram = spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
            sgram = spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        else:
            reaud = resample(aud, self.sr)
            t_aud = trim(reaud)
            rechan = rechannel(t_aud, self.channel)
            dur_aud = pad_trunc(rechan, self.duration)
            sgram = spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)

        if self.mode_reduc = "PCA"
            pca = PCA(n_components= self.pca_comp)
            if self.channel == 2:
                sgram = pca.fit_transform(sgram[0]), pca.fit_transform(sgram[1])
            else:
                sgram = pca.fit_transform(sgram)

            sgram = torch.Tensor(sgram)

        elif self.mode_reduc = "TSNE"

        return sgram, labelId
