from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy as sp
import pandas as pd

frame_size = 0.050*2
frame_stepsize = 0.025*2

labels = []
datadir = "C:/Users/varin/Desktop/New Audio Files/"
data = np.zeros((100,34*5))
for i, dirname in enumerate(os.listdir(datadir)):
        if "lie" in dirname:
                labels.append(1)
        else:
                labels.append(0)

        [Fs, x] = audioBasicIO.readAudioFile(datadir+dirname)
        st_features = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size*Fs, frame_stepsize*Fs)
        num_features, num_windows = st_features.shape

        for j in range(num_features):
                avg = np.mean(st_features[j])
                min_value = min(st_features[j])
                max_value = max(st_features[j])
                std = np.std(st_features[j])
                skew = sp.stats.skew(st_features[j])
            
                segment = np.hstack((avg,min_value,max_value,std,skew))
                segment = np.reshape(segment,(1,5))
                      
                data[i, 5*j:5*j+5] = segment
                

final_data = pd.DataFrame(data)
labels = np.array(labels)
final_data["Labels"] = labels
final_data.to_csv("audio_data.csv")
