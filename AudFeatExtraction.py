from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy as sp

frame_size = 0.050*2
frame_stepsize = 0.025*2

labels = {}
datadir = "C:/Users/varin/Desktop/New Audio Files/"
data = np.zeros((100,34*5))
for i, dirname in enumerate(os.listdir(datadir)):
        if "lie" in dirname:
                labels[dirname] = 1
        else:
                labels[dirname] = 0

        [Fs, x] = audioBasicIO.readAudioFile(datadir+dirname)
        st_features = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size*Fs, frame_stepsize*Fs)
        num_features, num_windows = st_features.shape

        for j in range(num_features):
                avg = np.mean(st_features[j])
                min_value = np.min(st_features[j])
                max_value = np.max(st_features[j])
                std = np.std(st_features[j])
                skew = sp.stats.skew(st_features[j])
            
                segment = np.hstack((avg,min,max,std,skew))
                start = j*5
                end = j*5 + 5
                data[i,start:end] = segment
