import os
import time
import ast
import subprocess as sp
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import librosa
from scipy.fftpack import fft


class FeatureGenerator:
    '''
    feature generator for FMA (small),
    GTZAN (10-fold CV & fault-filtered)

    **  gen_feat() can be written in general form,
        but you should consider the file saving process
    '''
    def __init__(self):
        # feature configuration
        self.dur = 30
        self.sr = 22050
        self.n_fft = 1024
        self.hop_size = 512
        self.n_mels = 96
        
    def load_librosa(self, source_path, sr, dur):
        # using ffmpeg
        signal, _ = librosa.load(source_path, sr=sr, res_type='kaiser_fast')

        # zero padding or trim to specific length
        samples_num = sr * dur
        signal_padded = np.zeros(samples_num)
        if len(signal) <= samples_num:
            signal_padded[:len(signal)] = signal
        else:
            signal_padded = signal[:samples_num]
        return signal_padded
    
    def load_ffmpeg(self, source_path, sr, dur):
        # using ffmpeg
        command = ['ffmpeg',
            '-i', source_path,
            '-f', 'f32le',
            '-acodec', 'pcm_f32le',
            '-ac', '1',
            '-ar', str(sr)]  # channels: 2 for stereo, 1 for mono
        command.append('-')
        proc = sp.run(command, stdout=sp.PIPE, bufsize=10**8, stderr=sp.DEVNULL, check=True)
        signal = np.fromstring(proc.stdout, dtype="float32")

        # zero padding or trim to specific length
        samples_num = sr * dur
        signal_padded = np.zeros(samples_num)
        if len(signal) <= samples_num:
            signal_padded[:len(signal)] = signal
        else:
            signal_padded = signal[:samples_num]
        return signal_padded

    def frequency_modulation(self, x, norm=False):
        '''
        we assume that the input data shape is (num_data, fbin, frame_size)
        '''
        x_fm = np.zeros((x.shape[0], x.shape[1], x.shape[2]//2), dtype=np.float32)

        for s, song in enumerate(x):
            temp = np.abs(fft(song, axis=-1))
            temp = temp[:,:temp.shape[-1]//2]
            if norm is True:
                temp = (temp - np.mean(temp)) / np.std(temp)
            x_fm[s] = temp

        print('after modulating, max, min, mean, std')
        print(np.max(x_fm), np.min(x_fm), np.mean(x_fm), np.std(x_fm))

        return x_fm

    def gen_acoustic_features(self, wav_path):
        # generate 5 (stft, mel-scaled, harmonic, percussive, and


                # load music waveform
        signal = self.load_librosa(
            wav_path, 
            sr=self.sr, 
            dur=self.dur)
        
        # gen spectrogram by stft
        stft = librosa.core.stft(signal, 
                                 n_fft=self.n_fft, 
                                 hop_length=self.hop_size)
        harmonic_stft, percussive_stft = librosa.decompose.hpss(stft)

        stft = np.abs(stft)**2
        stft_db = librosa.power_to_db(stft)

        # spectrogram in mel scaled
        mel_stft = librosa.feature.melspectrogram(S=stft, 
                                                  n_mels=self.n_mels)
        mel_stft = librosa.power_to_db(mel_stft)
        
        # harmonic & percussive pattern from mel-scaled spectrogram
        harmonic_stft = np.abs(harmonic_stft)**2
        mel_harmonic_stft = librosa.feature.melspectrogram(S=harmonic_stft, 
                                                           n_mels=self.n_mels)
        mel_harmonic_stft = librosa.power_to_db(mel_harmonic_stft)

        percussive_stft = np.abs(percussive_stft)**2
        mel_percussive_stft = librosa.feature.melspectrogram(S=percussive_stft, 
                                                             n_mels=self.n_mels)
        mel_percussive_stft = librosa.power_to_db(mel_percussive_stft)
        
        # mfcc feature
        mfcc = librosa.feature.mfcc(S=mel_stft, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)

        return [stft_db, mel_stft, mel_harmonic_stft, mel_percussive_stft, mfcc]