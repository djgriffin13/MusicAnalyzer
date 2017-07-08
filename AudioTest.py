import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import AudioFeatureExtraction as fe

def plot(x, y, xlabel, ylabel, title):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

rate, wav_data = wavfile.read("Data/Samples/841.wav")

# Plot original audio
plot(fe.wav_indices_to_sec(wav_data, rate), wav_data, "Time (Seconds)", "Amplitude", "\"Nicotine\" by Kurt Vile - Original Audio")

# Plot FFT data
FFT_data = fe.get_FFT_data(wav_data, rate)
#plot(fe.FFT_indices_to_hz(FFT_data, rate), FFT_data, "Frequency (Hz)", "Intensity", "FFT of Audio")

# Smooth data
#resample_wav_data = fe.run_avg(wav_data, 0.1 * rate)
#resample_rate = rate / (0.1 * rate)
#plot(fe.wav_indices_to_sec(resample_wav_data, resample_rate), resample_wav_data, "Time (Seconds)", "Amplitude", "Resampled Audio at 10 Hz")

#smooth_FFT_data_1 = fe.conv_smooth(FFT_data, 300)
#plot(fe.FFT_indices_to_hz(smooth_FFT_data_1, rate), smooth_FFT_data_1, "Frequency (Hz)", "Intensity", "FFT Smoothed by Convolution (width 300 bins)")
#smooth_FFT_data_2 = fe.run_avg(FFT_data, 300)
#plot(fe.FFT_indices_to_hz(smooth_FFT_data_2, rate), smooth_FFT_data_2, "Frequency (Hz)", "Intensity", "FFT Smoothed by Running Average (width 300 bins)")

# Print peak finding methods
smooth_FFT_data_3 = fe.conv_smooth(FFT_data, 100)
plot(fe.FFT_indices_to_hz(smooth_FFT_data_3, rate), smooth_FFT_data_3, "Frequency (Hz)", "Intensity", "FFT Smoothed by Convolution (Box width: 100)")
#print "Top 10 Values: ", fe.top_n_vals(smooth_FFT_data_3, fe.FFT_indices_to_hz(smooth_FFT_data_3, rate), 10)
#print "Peak Finder (Threshold 1.3*10^7): ", fe.find_peaks(smooth_FFT_data_3, fe.FFT_indices_to_hz(smooth_FFT_data_3, rate), 1.3*(10**7))

# Note bin methods
#print 'NOTE\tAVG\tMAX'
#for i in range(0,97):
#    avg = fe.avg_note_intensity(fe.FFT_indices_to_hz(FFT_data, rate), FFT_data, i)
#    max = fe.max_note_intensity(fe.FFT_indices_to_hz(FFT_data, rate), FFT_data, i)
#    print fe.NOTES[i] + '\t' + str(avg) + '\t' + str(max)

# Evaluate note C_5
#print fe.avg_note_intensity(fe.FFT_indices_to_hz(FFT_data, rate), FFT_data, 0+12*5)
#print fe.max_note_intensity(fe.FFT_indices_to_hz(FFT_data, rate), FFT_data, 0+12*5)
#print fe.max_note_intensity(fe.FFT_indices_to_hz(smooth_FFT_data_2, rate), smooth_FFT_data_2, 0+12*5)

#print fe.max_note_intensity(fe.FFT_indices_to_hz(FFT_data, rate), FFT_data, 0)
#print fe.max_note_intensity(fe.FFT_indices_to_hz(smooth_FFT_data_2, rate), smooth_FFT_data_2, 0)

x_vals = fe.FFT_indices_to_hz(smooth_FFT_data_3, rate)
note_intensities = [fe.max_note_intensity(x_vals, FFT_data, i) for i in range(0,97)]
avg = fe.total_avg_note_intensity(x_vals, smooth_FFT_data_3)
max = [np.sum([note_intensities[i+12*j] for j in range(0,8)]) for i in range(0,12)]

fig, ax = plt.subplots()
bar_1 = ax.bar(np.arange(12), avg)
ax.set_xticks(np.arange(12) + [.4]*12)
ax.set_xticklabels(fe.BASE_NOTES)
plt.xlabel('Western-Style Notes (Semitones)')
plt.ylabel('Total Average Intensity')
plt.title('Total Average Intensity Within Each Note Frequency Bin')
plt.show()

fig, ax = plt.subplots()
bar_2 = ax.bar(np.arange(12), max)
ax.set_xticks(np.arange(12) + [.4]*12)
ax.set_xticklabels(fe.BASE_NOTES)
plt.xlabel('Western-Style Notes (Semitones)')
plt.ylabel('Total Max Intensity')
plt.title('Total Maximum Intensity Within Each Note Frequency Bin')
plt.show()

fig, ax = plt.subplots()
bar_3 = ax.bar(np.arange(97), note_intensities)
ax.set_xticks(np.arange(97)[::15] + [.4]*7)
ax.set_xticklabels(fe.NOTES_SHARP[::15])
plt.xlabel('Western-Style Notes (Semitones)')
plt.ylabel('Maximum Intensity')
plt.title('Maximum Intensity Within Each Semitone')
plt.show()

#plot(fe.FFT_indices_to_hz(FFT_data, rate), FFT_data, "Frequency (Hz)", "Intensity", "FFT of Audio")
#plot(fe.FFT_indices_to_hz(smooth_FFT_data_2, rate), smooth_FFT_data_2, "Frequency (Hz)", "Intensity", "FFT Smoothed by Running Average (width 300 bins)")

#print 'Total averages for each note type: ', fe.total_avg_note_intensity(fe.FFT_indices_to_hz(smooth_FFT_data_2, rate), smooth_FFT_data_2)
#print 'Total maximums for each note type: ', fe.total_max_note_intensity(fe.FFT_indices_to_hz(smooth_FFT_data_2, rate), smooth_FFT_data_2)

# External library feature extraction
#print 'External Vectors: ', fe.external_feature_vects(wav_data, rate)
#print 'Estimated BPM: ', fe.get_beat(wav_data, rate)