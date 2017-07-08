import cPickle
import sys
import AudioFeatureExtraction as fe
from lyrics import analyzeLyricalEmotionContent
import matplotlib.pyplot as plt
import numpy as np

# COMMAND LINE ARGUMENTS:
# arg[1] = relative audio file path
# arg[2] = relative lyric file path
if __name__ == '__main__':
    audio_filename = sys.argv[1]
    lyric_filename = sys.argv[2]

    # Extract audio features
    print 'EXTRACTING AUDIO FEATURES...'
    features = fe.feature_combined(audio_filename, smooth_window_size = 100)

    # Load predictive models
    # Lyric analyzer imported as a module at top of script
    # Audio NNs loaded here
    audio_arousal_network_0 = cPickle.load(open('Networks/audio_arousal_network_0.nn','r'))
    audio_valence_network_0 = cPickle.load(open('Networks/audio_valence_network_0.nn','r'))
    audio_arousal_network_1 = cPickle.load(open('Networks/audio_arousal_network_1.nn','r'))
    audio_valence_network_1 = cPickle.load(open('Networks/audio_valence_network_1.nn','r'))
    audio_arousal_network_2 = cPickle.load(open('Networks/audio_arousal_network_2.nn','r'))
    audio_valence_network_2 = cPickle.load(open('Networks/audio_valence_network_2.nn','r'))
    #arousal_network = cPickle.load(open('Networks/arousal_network.nn','r'))
    #valence_network = cPickle.load(open('Networks/valence_network.nn','r'))

    # Get predictions
    print 'EXTRACTING AND ANALYZING LYRIC FEATURES...'
    predicted_lyric_vector_backwards = analyzeLyricalEmotionContent(lyric_filename).currentVector
    predicted_lyric_vector = [predicted_lyric_vector_backwards[1], predicted_lyric_vector_backwards[0]]
    print 'ANALYZING AUDIO FEATURES...'
    predicted_audio_vector_0 = [audio_arousal_network_0.activate(features[0])[0], audio_valence_network_0.activate(features[0])[0]]
    predicted_audio_vector_1 = [audio_arousal_network_1.activate(features[1])[0], audio_valence_network_1.activate(features[1])[0]]
    predicted_audio_vector_2 = [audio_arousal_network_2.activate(features[2])[0], audio_valence_network_2.activate(features[2])[0]]
    # TODO determine which predicted_audio_vector to use and feed to combined
    # predicted_vector = [arousal_network.activate([predicted_audio_vector_0[0], predicted_audio_vector_0[1], predicted_lyric_vector[0], predicted_lyric_vector[1]])]
    print 'Predicted emotion vector for lyrics: ' + str(predicted_lyric_vector)
    print 'Predicted emotion vector for audio (method 1 - third party): ' + str(predicted_audio_vector_0)
    print 'Predicted emotion vector for audio (method 2 - 12 note bins): ' + str(predicted_audio_vector_1)
    print 'Predicted emotion vector for audio (method 3 - 96 semitone bins): ' + str(predicted_audio_vector_2)
    # print 'Predicted emotion vector overall: ' + predicted_vector

    # Explicitly reference audio features to plot for demonstration
    max = features[1][13:]
    note_intensities = features[2][:-1]
    # Plot the features
    print 'PLOTTING AUDIO FEATURE GRAPHICS...'
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
