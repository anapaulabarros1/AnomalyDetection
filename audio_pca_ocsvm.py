import numpy as np
import matplotlib.pyplot as plt

filename_train = 'data motor lab ensaio 1 - 1800rpm 5khz.txt'

file1 = open(filename_train, 'r')
Lines = file1.readlines()

samplingFrequency_audio = 44000
samplingFrequency_accel = 1000

list_time = []
list_audio = []
list_audio_fft = []
list_temp = []
list_ax = []
list_ay = []
list_az = []
list_gx = []
list_gy = []
list_gz = []
list_ax_fft = []
list_ay_fft = []
list_az_fft = []
list_gx_fft = []
list_gy_fft = []
list_gz_fft = []

list_ax_ = []
list_ay_ = []
list_az_ = []
list_gx_ = []
list_gy_ = []
list_gz_ = []

micros_diffs = []

check_order = 0
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    print(count)

    line_list = line.replace('\n','').split('\t')
    if '' in line_list:
        line_list.remove('')

    line_float = np.array(line_list,dtype=float)

    if len(line_list) == 501:
        if len(list_ax_) == 1000:
            list_ax.append(list_ax_)
            list_ay.append(list_ay_)
            list_az.append(list_az_)
            list_gx.append(list_gx_)
            list_gy.append(list_gy_)
            list_gz.append(list_gz_)

            # Frequency domain representation
            amplitude = list_ax[-1]
            fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
            fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
            fourierTransform[0] = 0
            fourierTransform = abs(fourierTransform)
            fourierTransform /= np.amax(fourierTransform)
            list_ax_fft.append(fourierTransform)

            amplitude = list_ay[-1]
            fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
            fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
            fourierTransform[0] = 0
            fourierTransform = abs(fourierTransform)
            fourierTransform /= np.amax(fourierTransform)
            list_ay_fft.append(fourierTransform)

            amplitude = list_az[-1]
            fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
            fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
            fourierTransform[0] = 0
            fourierTransform = abs(fourierTransform)
            fourierTransform /= np.amax(fourierTransform)
            list_az_fft.append(fourierTransform)

            amplitude = list_gx[-1]
            fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
            fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
            fourierTransform[0] = 0
            fourierTransform = abs(fourierTransform)
            fourierTransform /= np.amax(fourierTransform)
            list_gx_fft.append(fourierTransform)

            amplitude = list_gy[-1]
            fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
            fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
            fourierTransform[0] = 0
            fourierTransform = abs(fourierTransform)
            fourierTransform /= np.amax(fourierTransform)
            list_gy_fft.append(fourierTransform)

            amplitude = list_gz[-1]
            fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
            fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
            fourierTransform[0] = 0
            fourierTransform = abs(fourierTransform)
            fourierTransform /= np.amax(fourierTransform)
            list_gz_fft.append(fourierTransform)

            # apaga lista temporária
            list_ax_ = []
            list_ay_ = []
            list_az_ = []
            list_gx_ = []
            list_gy_ = []
            list_gz_ = []
        else:
            if len(list_ax_) != 0:
                raise Exception('Unexpected accelerometer packet size ' + str(count) + ' ' + str(len(list_ax_)))

        assert check_order == 0
        check_order = 1

        list_time.append(line_float[0])
        list_audio.append(line_float[1:])

        # Frequency domain representation
        amplitude = list_audio[-1]
        fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
        fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
        fourierTransform[0] = 0
        fourierTransform = abs(fourierTransform)
        # fourierTransform /= np.amax(fourierTransform)
        list_audio_fft.append(fourierTransform)


    elif len(line_list) == 1:
        assert check_order == 1
        check_order = 2

        # apaga lista temporária
        list_ax_ = []
        list_ay_ = []
        list_az_ = []
        list_gx_ = []
        list_gy_ = []
        list_gz_ = []

        list_temp.append(line_float[0])

    elif len(line_list) == 6:
        check_order = 0

        list_ax_.append(line_float[0])
        list_ay_.append(line_float[1])
        list_az_.append(line_float[2])

        list_gx_.append(line_float[3])
        list_gy_.append(line_float[4])
        list_gz_.append(line_float[5])

    else:
        raise Exception("Unexpected line length " + str(count) + ' ' + str(len(line_list)) + ' ' + line)

tpCount = len(list_audio[-1])
values = np.arange(int(tpCount / 2))
timePeriod = tpCount / samplingFrequency_audio
frequencies_audio = values / timePeriod

tpCount = len(list_ax[-1])
values = np.arange(int(tpCount / 2))
timePeriod = tpCount / samplingFrequency_accel
frequencies_accel = values / timePeriod

# reduz o tamanho do FFT
bins_fft_audio = 20
fft_audio_bin_length = int(len(frequencies_audio) / bins_fft_audio)
frequencies_audio_reduced = np.linspace(start=np.amin(frequencies_audio), stop=np.amax(frequencies_audio), num=bins_fft_audio)

list_audio_fft_reduced = []

for audio_fft in list_audio_fft:
    bin_id = 0
    audio_fft_reduced = []

    for i in range(bins_fft_audio):
        avg_bin = np.average(audio_fft[i*fft_audio_bin_length:(i+1)*fft_audio_bin_length])
        audio_fft_reduced.append(avg_bin)

    list_audio_fft_reduced.append(audio_fft_reduced)

# mostra a FFT reduzida
# for i in range(len(list_audio_fft)):
#     plt.clf()
#
#     ax = plt.subplot(211)
#     plt.plot(frequencies_audio, list_audio_fft[i])
#     ax.set_xlabel('Frequency')
#     ax.set_ylabel('Amplitude')
#
#     ax = plt.subplot(212)
#     plt.plot(frequencies_audio_reduced, list_audio_fft_reduced[i])
#     ax.set_xlabel('Frequency')
#     ax.set_ylabel('Amplitude')
#
#     plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

list_audio_fft_reduced = np.array(list_audio_fft_reduced)
list_audio_fft_reduced = StandardScaler().fit_transform(list_audio_fft_reduced)

n_components = 2

pca = PCA(n_components=n_components)

principalComponents = pca.fit_transform(list_audio_fft_reduced)

print(pca.explained_variance_ratio_)

list_audio_fft_reduced_transformed = pca.transform(list_audio_fft_reduced)

# plota os dados reduzidor de PCA, supondo dois componentes
# plt.clf()
#
# plt.scatter(x=list_audio_fft_reduced_transformed[:,0], y=list_audio_fft_reduced_transformed[:,1])
#
# plt.show()


from sklearn.svm import OneClassSVM

# Define "classifiers" to be used

nu = 0.1
gamma = 0.1

clf = OneClassSVM(nu=nu, gamma=gamma)

clf.fit(list_audio_fft_reduced_transformed)

# plota o resultado de clusterização
xx, yy = np.meshgrid(np.linspace(np.amin(list_audio_fft_reduced_transformed[:,0]), np.amax(list_audio_fft_reduced_transformed[:,0]), 500),
                     np.linspace(np.amin(list_audio_fft_reduced_transformed[:,1]), np.amax(list_audio_fft_reduced_transformed[:,1]), 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.clf()

plt.scatter(x=list_audio_fft_reduced_transformed[:,0], y=list_audio_fft_reduced_transformed[:,1])

plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")

plt.show()