import warnings
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
import csv
import pandas as pd

class Data_Handling:
    header = ''
    data = ''
    def build_data_headers():
        for i in range(1, 21):
        header += f' mfcc_{i}'
        for i in range(1, 21):
            header += f' mfcc_delta_{i}'
        for i in range(1, 21):
            header += f' mfcc_delta2_{i}'
        header += ' spectral_flux'
        header += ' label'
        header = header.split()
    def build_data_records():
        file = open('data.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
        dialects = 'belfast bradford cambridge dublin leeds london newcastle cardiff liverpool'.split()
        for g in dialects:
            for filename in os.listdir(f'./dataset/{g}'):
                audio = f'./dataset/{g}/{filename}'
                y, sr = librosa.load(audio)
                to_append = f''
                mfccs = librosa.feature.mfcc(y=y, sr=sr)
                for e in mfccs:
                    to_append += f' {np.mean(e)}'
                mfccs_delta = librosa.feature.delta(mfccs)
                for e in mfccs_delta:
                    to_append += f' {np.mean(e)}'
                mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
                for e in mfccs_delta2:
                    to_append += f' {np.mean(e)}'
                spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
                to_append += f' {np.mean(spectral_flux)}'
                to_append += f' {g}'
                file = open('data.csv', 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())

if __name__ == '__main__':
    data_handling = Data_Handling()
    