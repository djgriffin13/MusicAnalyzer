import random
import numpy as np
from scipy.io import wavfile
from pyAudioAnalysis import audioFeatureExtraction
# Library obtained from https://github.com/tyiannak/pyAudioAnalysis

# Useful Constants
NOTE_RATIO = 2**(1./12.)
C_0_FREQ = 16.35
BASE_NOTES = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab' , 'A', 'A#/Bb', 'B']
BASE_NOTES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#' , 'A', 'A#', 'B']
BASE_NOTES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab' , 'A', 'Bb', 'B']
NOTES = [BASE_NOTES[i]+'_'+str(j) for j in range(0,8) for i in range(0,12)]
NOTES_SHARP = [BASE_NOTES_SHARP[i]+'_'+str(j) for j in range(0,8) for i in range(0,12)]
NOTES_FLAT = [BASE_NOTES_FLAT[i]+'_'+str(j) for j in range(0,8) for i in range(0,12)]

# Returns frequency of note number n, where n is the number of half steps above C_0
def note_freq(n):
    return C_0_FREQ * (NOTE_RATIO ** n)

# Create a frequency window/bin surrounding note n which grows exponentially
def note_freq_window(n):
    return [(C_0_FREQ/2.)*(NOTE_RATIO**(n-1))*(1+NOTE_RATIO), (C_0_FREQ/2.)*(NOTE_RATIO**n)*(1+NOTE_RATIO)]

# DATA SMOOTHERS
# Smooths the data by applying a convolution with a "box" function
def conv_smooth(y, width):
    frame = np.ones(width)/width
    out = np.convolve(y, frame, mode = 'same')
    return out

# Smooths data using a running average of the data
def run_avg(data, width):
    return [np.mean(chunk) for chunk in list_chunks(data,width)]

# Yields pieces of the list l which are length n
def list_chunks(l,n):
    n = int(n)
    for i in range(0,len(l),n):
        yield l[i:i+n]

# UNIT CONVERTERS
# Returns the value in Hz associated with each FFT_data entry
def FFT_indices_to_hz(FFT_data, rate):
    return [i*float(rate)/(2*float(len(FFT_data))) for i in range(len(FFT_data))]

# Returns the time in seconds associated with each wav_data entry
def wav_indices_to_sec(wav_data, rate):
    return [float(x)/float(rate) for x in range(len(wav_data))]

# Returns the magnitude of the real FFT of the wav_data
def get_FFT_data(wav_data, rate):
    return np.abs(np.fft.rfft(wav_data))

# FEATURE EXTRACTION HELPER METHODS
# Naively chooses the highest num_vals values in the data
def top_n_vals(data, x_vals, num_vals):
    result = []
    for _ in range(num_vals):
        index = np.argmax(data)
        result.append(x_vals[index])
        # Attempt to flatten the data around
        # the peak so it isn't selected again
        data[index-600:index+600] = [0]*(1200)
    return result

# Finds all points which are taller than the points to their immediate left and right
def find_peaks(data, x_vals, thresh):
    f1 = np.abs(data[1:-1])
    f2 = np.abs(data[:-2])
    f3 = np.abs(data[2:])
    peaks = [x_vals[i] for i in range(len(data)-2) if f1[i] > f2[i] and f1[i] > f3[i] and f1[i] > thresh]
    return peaks

# Finds the average y value (intensity) between the x values low and high (frequency values)
def avg_intensity_in_window(low, high, x_vals, FFT_data):
    indices_btwn_low_high = [n for n,item in enumerate(x_vals) if item >= low and item <= high]
    # If window is empty, return 0
    if not(indices_btwn_low_high) or indices_btwn_low_high[0] >= indices_btwn_low_high[-1]:
        return 0
    return np.mean(FFT_data[indices_btwn_low_high[0]:indices_btwn_low_high[-1]])

# Finds the max y value (intensity) between the x values low and high (frequency values)
def max_intensity_in_window(low, high, x_vals, FFT_data):
    indices_btwn_low_high = [n for n,item in enumerate(x_vals) if item >= low and item <= high]
    # If window is empty, return 0
    if not(indices_btwn_low_high) or indices_btwn_low_high[0] >= indices_btwn_low_high[-1]:
        return 0
    return np.max(FFT_data[indices_btwn_low_high[0]:indices_btwn_low_high[-1]])

# Finds the average intensity of a specific note's window
def avg_note_intensity(x_vals, FFT_data, note):
    window = note_freq_window(note)
    return avg_intensity_in_window(window[0], window[1], x_vals, FFT_data)

# Finds the max intensity of a specific note's window
def max_note_intensity(x_vals, FFT_data, note):
    window = note_freq_window(note)
    return max_intensity_in_window(window[0], window[1], x_vals, FFT_data)

# Returns a list of the total avg intensity of each note's frequency bins
def total_avg_note_intensity(x_vals, FFT_data):
    # Obtains the frequency window and avg intensity for 96 notes ranging between 16 Hz and 7900 Hz
    windows = [note_freq_window(i) for i in range(0, 97)]
    avgs = [avg_intensity_in_window(windows[i][0], windows[i][1], x_vals, FFT_data) for i in range(1,97)]
    # Sums the averages among the same notes across different octaves (12 notes total)
    return [np.sum([avgs[i+12*j] for j in range(0,8)]) for i in range(0,12)]

# Returns a list of the total max intensity of each note's frequency bins
def total_max_note_intensity(x_vals, FFT_data):
    # Obtains the frequency window and max intensity for 96 notes ranging between 16 Hz and 7900 Hz
    windows = [note_freq_window(i) for i in range(0, 97)]
    maxs = [max_intensity_in_window(windows[i][0], windows[i][1], x_vals, FFT_data) for i in range(1,97)]
    # Sums the max intensities among the same notes across different octaves (12 notes total)
    return [np.sum([maxs[i+12*j] for j in range(0,8)]) for i in range(0,12)]

# Uses an external library to extract the BPM from a wav file
def get_beat(wav_data, rate):
    features = audioFeatureExtraction.stFeatureExtraction(wav_data, rate, 0.050 * rate, 0.050 * rate)
    BPM, r = audioFeatureExtraction.beatExtraction(features, 0.050)
    return BPM

# Uses an external library to extract various features
def external_feature_vects(wav_data, rate):
    features = audioFeatureExtraction.stFeatureExtraction(wav_data, rate, 0.050 * rate, 0.050 * rate)
    return features

# Use the external library to extract various features and average the results
def avg_external_feature_vect(wav_data, rate):
    vect = external_feature_vects(wav_data, rate)
    return np.mean(vect, axis=0)

# FEATURE EXTRACTION METHODS
# Uses the external library to retrieve the average features over .05 second windows
def feature_method_1(filename):
    rate, wav_data = wavfile.read(filename)
    return avg_external_feature_vect(wav_data, rate)

# Returns the average note intensities and max intensities for all 12 standard notes 
# and the projected beat/tempo
# If no window size is specified, then no smoothing is performed (may take longer)
def feature_method_2(filename, smooth_window_size = 0):
    rate, wav_data = wavfile.read(filename)
    FFT_data = get_FFT_data(wav_data, rate)
    if smooth_window_size != 0:
        FFT_data = run_avg(FFT_data, smooth_window_size)
    BPM = get_beat(wav_data, rate)
    avg = total_avg_note_intensity(FFT_indices_to_hz(FFT_data, rate), FFT_data)
    max = total_max_note_intensity(FFT_indices_to_hz(FFT_data, rate), FFT_data)
    return np.append(avg, np.append(max, BPM))

# Returns the max value in each of the frequency bins and the projected beat/tempo
# If no window size is specified, then no smoothing is performed (may take longer)
def feature_method_3(filename, smooth_window_size = 0):
    rate, wav_data = wavfile.read(filename)
    FFT_data = get_FFT_data(wav_data, rate)
    if smooth_window_size != 0:
        FFT_data = run_avg(FFT_data, smooth_window_size)
    BPM = get_beat(wav_data, rate)
    note_intensities = [max_note_intensity(FFT_indices_to_hz(FFT_data, rate), FFT_data, i) for i in range(0,97)]
    return np.append(note_intensities, BPM)

# Returns an estimate of the top 30 peaks in the FFT data
# If no convolution parameter is specified, then no 
# smoothing is performed (may tbe less accurate)
def feature_method_4(filename, conv_param = 0):
    rate, wav_data = wavfile.read(filename)
    FFT_data = get_FFT_data(wav_data, rate)
    if conv_param != 0:
        FFT_data = conv_smooth(FFT_data, conv_param)
    BPM = get_beat(wav_data, rate)
    vals = top_n_vals(FFT_data, FFT_indices_to_hz(FFT_data, rate), 30)
    return np.append(vals, BPM)

# Method 4 is left out, as it is not very useful
def feature_combined(filename, smooth_window_size = 0):
    rate, wav_data = wavfile.read(filename)

    external_vect = external_feature_vects(wav_data, rate)
    BPM, _ = audioFeatureExtraction.beatExtraction(external_vect, 0.050)
    method_1 = np.mean(external_vect, axis=0)

    FFT_data = get_FFT_data(wav_data, rate)
    if smooth_window_size != 0:
        FFT_data = run_avg(FFT_data, smooth_window_size)
    BPM = get_beat(wav_data, rate)
    x_vals = FFT_indices_to_hz(FFT_data, rate)

    note_intensities = [max_note_intensity(x_vals, FFT_data, i) for i in range(0,97)]
    method_3 = np.append(note_intensities, BPM)

    avg = total_avg_note_intensity(x_vals, FFT_data)
    max = [np.sum([note_intensities[i+12*j] for j in range(0,8)]) for i in range(0,12)]
    method_2 = np.append(avg, np.append(max, BPM))

    #vals = top_n_vals(FFT_data, x_vals, 30)
    #method_4 = np.append(vals, BPM)

    return [method_1, method_2, method_3]