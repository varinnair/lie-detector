from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
dirname = "C:/Users/varin/Desktop/ML audio to vector/New Recording 1-1 1.wav"
[Fs, x] = audioBasicIO.readAudioFile(dirname)
F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)

print (F.shape)
