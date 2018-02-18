from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import os

frame_size = 0.050*2
frame_stepsize = 0.025*2

labels = {}
datadir = "C:/Users/varin/Desktop/New Audio Files/"
for i, dirname in enumerate(os.listdir(datadir)):
	speaker = dirname 
	
	if "lie" in filename:
		#labels.append((filename, 1))
		labels[filename] = 1
	else:
		labels[filename] = 0

	[Fs, x] = audioBasicIO.readAudioFile(datadir+dirname)
	#we might want to play with the timeframe here - as it is this is giving us up to ~1.5k frames for our sequences
	#speaker_feat = speaker_features[dirname]
	st_features = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size*Fs, frame_stepsize*Fs)
	num_features, num_windows = st_features.shape #34, 388




	new_features = np.zeros((num_features, num_windows))
	for i in range(num_features):
		new_features[i] = (st_features[i] - speaker_feat[i])/speaker_feat[i]
	st_features = np.concatenate((st_features, new_features))
	features[filename] = st_features.tolist()
	total += 1
	print(i)


"""

dirname = "C:/Users/varin/Desktop/ML audio to vector/New Recording 1-1 1.wav"



[Fs, x] = audioBasicIO.readAudioFile(dirname)
F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)




print (F.shape) #34, 388
"""