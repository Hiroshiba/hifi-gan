import math
import os
import random
from typing import List, Optional, Tuple
import torch
import torch.utils.data
import numpy as np
from librosa.core import load
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav(full_path, sampling_rate=None):
    if os.path.splitext(full_path)[1] != '.npy':
        data, sampling_rate = load(full_path, sr=sampling_rate)
    else:
        a = np.load(full_path, allow_pickle=True).item()
        assert sampling_rate == a['rate']
        data = a['array']
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def resample(array: torch.Tensor, src_rate: int, tgt_rate: int, index = 0, length: int = None):
    if length is None:
        length = int(array.size(1) / src_rate * tgt_rate)
    indexes = (torch.rand(1) + index + torch.arange(length, dtype=torch.float32)) * (src_rate / tgt_rate)
    return array[:, indexes.long()]


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    def _get_files(x: str):
        filepaths = x.split('|')
        if len(filepaths) == 1:
            wav_path, f0_path = filepaths[0], None
        else:
            wav_path, f0_path = filepaths[0], filepaths[1]
        
        base_name = os.path.splitext(wav_path)[0]
        
        if not os.path.exists(wav_path):
            wav_path = os.path.join(a.input_wavs_dir, wav_path + ('.npy' if a.input_wavs_npy else '.wav'))
        if f0_path is not None and not os.path.exists(f0_path):
            f0_path = os.path.join(a.input_f0_dir, f0_path + '.npy')

        if a.fine_tuning:
            mel_path = os.path.join(a.input_mels_dir, base_name + '.npy')
        else:
            mel_path = None

        assert os.path.exists(wav_path), wav_path
        assert f0_path is None or os.path.exists(f0_path), f0_path

        return wav_path, f0_path, mel_path

    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [_get_files(x) for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [_get_files(x) for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files: List[Tuple[str, Optional[str], Optional[str]]], segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning

    def __getitem__(self, index):
        filename, f0_filename, mel_filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename, self.sampling_rate)
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if f0_filename is not None:
            obj = np.load(f0_filename, allow_pickle=True).item()
            f0, f0_rate = obj["array"], obj["rate"]
            f0[f0 > 0] = np.exp(f0[f0 > 0])
            f0 = np.pad(f0, (0, f0_rate//50), 'edge')  # ちょっと伸ばす
            f0 = torch.FloatTensor(f0)
            f0 = f0.unsqueeze(0)
        else:
            f0 = None

        frames_per_seg = math.ceil(self.segment_size / self.hop_size)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start) if max_audio_start > 0 else 0
                    audio = audio[:, audio_start:audio_start+self.segment_size]

                    if f0 is not None:
                        f0 = resample(f0, f0_rate, self.sampling_rate / self.hop_size, audio_start // self.hop_size, frames_per_seg)
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
                    f0 = resample(f0, f0_rate, self.sampling_rate / self.hop_size)
                    f0 = torch.nn.functional.pad(f0, (0, frames_per_seg - f0.size(1)), 'constant')
            else:
                if f0 is not None:
                    f0 = resample(f0, f0_rate, self.sampling_rate / self.hop_size)

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        else:
            mel = np.load(mel_filename)
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                if f0 is not None:
                    f0 = resample(f0, f0_rate, self.sampling_rate / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    max_mel_start = mel.size(2) - frames_per_seg - 1
                    mel_start = random.randint(0, max_mel_start) if max_mel_start > 0 else 0
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                    if f0 is not None:
                        f0 = resample(f0, f0_rate, self.sampling_rate / self.hop_size, mel_start, frames_per_seg)
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
                    if f0 is not None:
                        f0 = torch.nn.functional.pad(f0, (0, frames_per_seg - f0.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)
        if f0 is not None:
            f0 = f0[:, :mel.size(2)]

        return (
            mel.squeeze(),
            audio.squeeze(0),
            f0.squeeze(0) if f0 is not None else float('nan'),
            filename,
            mel_loss.squeeze(),
        )

    def __len__(self):
        return len(self.audio_files)
