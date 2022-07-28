import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

files_normal = ['data CE 1 - normal 1.txt',
                'data CE 1 - normal 2.txt',
                'data CE 1 - normal 3.txt',
                'data CE 1 - normal 4.txt']

files_anomaly = ['data CE 1 - peça solta 3.txt',
                 'data CE 1 - peça solta 2.txt',
                 'data CE 1 - peça solta 1.txt',
                 'data CE 1 - peça solta - plastico.txt',
                 'data CE 1 - desbalanceada - 1g.txt',
                 'data CE 1 - desbalanceada 1.txt',
                 'data CE 1 - desbalanceada 2.txt',
                 'data CE 1 - parada.txt']

random.shuffle(files_normal)
random.shuffle(files_anomaly)

filename_train = files_normal[0]
print(filename_train)

filename_val_normal = files_normal[1]
filename_val_anomaly = files_anomaly[0]
print(filename_val_normal)
print(filename_val_anomaly)

input()

filename_test_normal = files_normal[2]
filename_test_anomaly = files_anomaly[1]

samplingFrequency_audio = 44000
samplingFrequency_accel = 1000

# --------------------------------
# TODO: mais parâmetros!!!!
# hyperparameters:
bins_fft_audio = 5

n_components = 2

# OCSVM
nu = 0.1
gamma = 0.1


# --------------------------------

def load_file(filename):
    file1 = open(filename, 'r')
    Lines = file1.readlines()

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

        line_list = line.replace('\n', '').split('\t')
        if '' in line_list:
            line_list.remove('')

        line_float = np.array(line_list, dtype=float)

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
                # fourierTransform /= np.amax(fourierTransform)
                list_ax_fft.append(fourierTransform)

                amplitude = list_ay[-1]
                fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
                fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
                fourierTransform[0] = 0
                fourierTransform = abs(fourierTransform)
                # fourierTransform /= np.amax(fourierTransform)
                list_ay_fft.append(fourierTransform)

                amplitude = list_az[-1]
                fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
                fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
                fourierTransform[0] = 0
                fourierTransform = abs(fourierTransform)
                # fourierTransform /= np.amax(fourierTransform)
                list_az_fft.append(fourierTransform)

                amplitude = list_gx[-1]
                fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
                fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
                fourierTransform[0] = 0
                fourierTransform = abs(fourierTransform)
                # fourierTransform /= np.amax(fourierTransform)
                list_gx_fft.append(fourierTransform)

                amplitude = list_gy[-1]
                fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
                fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
                fourierTransform[0] = 0
                fourierTransform = abs(fourierTransform)
                # fourierTransform /= np.amax(fourierTransform)
                list_gy_fft.append(fourierTransform)

                amplitude = list_gz[-1]
                fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
                fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
                fourierTransform[0] = 0
                fourierTransform = abs(fourierTransform)
                # fourierTransform /= np.amax(fourierTransform)
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

    return frequencies_audio, frequencies_accel, list_audio_fft, list_ax_fft, list_ay_fft , list_az_fft, list_gx_fft, list_gy_fft, list_gz_fft


def reduce_fft(n_bins, frequencies, list_fft):
    # reduz o tamanho do FFT
    fft_audio_bin_length = int(len(frequencies) / n_bins)

    list_audio_fft_reduced = []

    for audio_fft in list_fft:
        bin_id = 0
        audio_fft_reduced = []

        for i in range(n_bins):
            avg_bin = np.average(audio_fft[i * fft_audio_bin_length:(i + 1) * fft_audio_bin_length])
            audio_fft_reduced.append(avg_bin)

        list_audio_fft_reduced.append(audio_fft_reduced)

    # #mostra a FFT reduzida
    # frequencies_audio_reduced = np.linspace(start=np.amin(frequencies), stop=np.amax(frequencies),
    #                                         num=bins_fft_audio)
    # for i in range(len(list_fft)):
    #     plt.clf()
    #
    #     ax = plt.subplot(211)
    #     plt.plot(frequencies, list_fft[i])
    #     ax.set_xlabel('Frequency')
    #     ax.set_ylabel('Amplitude')
    #
    #     ax = plt.subplot(212)
    #     plt.plot(frequencies_audio_reduced, list_audio_fft_reduced[i])
    #     ax.set_xlabel('Frequency')
    #     ax.set_ylabel('Amplitude')
    #
    #     plt.show()

    return np.array(list_audio_fft_reduced).copy()


frequencies_audio_train, frequencies_accel_train, list_audio_fft_train, _, _, _, _, _, _ = load_file(filename_train)

list_audio_fft_reduced = reduce_fft(n_bins=bins_fft_audio,
                                    frequencies=frequencies_audio_train,
                                    list_fft=list_audio_fft_train)

scaler = StandardScaler().fit(list_audio_fft_reduced)
list_audio_fft_reduced = scaler.transform(list_audio_fft_reduced)

pca = PCA(n_components=n_components)

principalComponents = pca.fit(list_audio_fft_reduced)

# print(pca.explained_variance_ratio_)

list_audio_fft_reduced_transformed_train = pca.transform(list_audio_fft_reduced)

clf = OneClassSVM(nu=nu, gamma=gamma)

clf.fit(list_audio_fft_reduced_transformed_train)

# Validation normal
frequencies_audio_val_normal, frequencies_accel_val_normal, list_audio_fft_val_normal, _, _, _, _, _, _ = load_file(filename_val_normal)
list_audio_fft_reduced = reduce_fft(n_bins=bins_fft_audio,
                                    frequencies=frequencies_audio_val_normal,
                                    list_fft=list_audio_fft_val_normal)
list_audio_fft_reduced = scaler.transform(list_audio_fft_reduced)
list_audio_fft_reduced_transformed_val_normal = pca.transform(list_audio_fft_reduced)

# Validation anomaly
frequencies_audio_val_anomaly, frequencies_accel_val_anomaly, list_audio_fft_val_anomaly, _, _, _, _, _, _ = load_file(
    filename_val_anomaly)
list_audio_fft_reduced = reduce_fft(n_bins=bins_fft_audio,
                                    frequencies=frequencies_audio_val_anomaly,
                                    list_fft=list_audio_fft_val_anomaly)
list_audio_fft_reduced = scaler.transform(list_audio_fft_reduced)
list_audio_fft_reduced_transformed_val_anomaly = pca.transform(list_audio_fft_reduced)

# plota o resultado de clusterização
xx, yy = np.meshgrid(np.linspace(np.amin(list_audio_fft_reduced_transformed_train[:, 0]),
                                 np.amax(list_audio_fft_reduced_transformed_train[:, 0]), 500),
                     np.linspace(np.amin(list_audio_fft_reduced_transformed_train[:, 1]),
                                 np.amax(list_audio_fft_reduced_transformed_train[:, 1]), 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

for x in list_audio_fft_reduced_transformed_val_anomaly:
    print(clf.predict([x]), clf.score_samples([x]))

plt.clf()

plt.scatter(x=list_audio_fft_reduced_transformed_train[:, 0], y=list_audio_fft_reduced_transformed_train[:, 1],
            c='blue', label='train')
plt.scatter(x=list_audio_fft_reduced_transformed_val_normal[:, 0],
            y=list_audio_fft_reduced_transformed_val_normal[:, 1], c='green', label='validation normal')
plt.scatter(x=list_audio_fft_reduced_transformed_val_anomaly[:, 0],
            y=list_audio_fft_reduced_transformed_val_anomaly[:, 1], c='yellow', label='validation anomaly')

plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")

plt.legend()
plt.show()
