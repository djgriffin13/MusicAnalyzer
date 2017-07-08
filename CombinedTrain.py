from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, BiasUnit, LinearLayer, FullConnection, SigmoidLayer, TanhLayer
import numpy as np
import os
import cPickle
from random import shuffle
from math import sqrt

# Returns a dictionary of the following form:
# Track number -> ((audio_arousal, audio_valence, lyric_arousal, lyric_valence), target_arousal, target_valence) 
def construct_data_set():
    annot = np.genfromtxt('Data/static_annotations_shifted.csv', delimiter=',')[1:]
    annot_tracks = annot[:,0].astype(int).tolist()
    annot_arousal = annot[:,5].tolist()
    annot_valence = annot[:,6].tolist()
    data = {}
    for i in range(1,1001):
        if os.path.exists('Data/AudioVects/' + str(i) + '.txt') and \
            os.path.getsize('Data/AudioVects/' + str(i) + '.txt') > 0 and \
            os.path.exists('Data/LyricVects/' + str(i) + '.txt') and \
            os.path.getsize('Data/LyricVects/' + str(i) + '.txt') > 0 and \
            i in annot_tracks:
            print 'READING FILE ' + str(i) 
            with open('Data/AudioVects/' + str(i) + '.txt', 'r') as audio_vect_input:
                with open('Data/LyricVects/' + str(i) + '.txt', 'r') as lyric_vect_input:
                    audio_arousal = float(audio_vect_input.readline())
                    audio_valence = float(audio_vect_input.readline())
                    lyric_arousal = float(lyric_vect_input.readline())
                    lyric_valence = float(lyric_vect_input.readline())
                    data[i] = ((audio_arousal, audio_valence, lyric_arousal, lyric_valence),
                                annot_arousal[annot_tracks.index(i)], annot_valence[annot_tracks.index(i)])
    return data

# Constructs and returns two neural nets
def build_new_nets():
    num_inputs = 4
    arousal_net = FeedForwardNetwork()
    arousal_net.addInputModule(LinearLayer(num_inputs, name='input'))
    arousal_net.addInputModule(BiasUnit(name='bias'))
    arousal_net.addOutputModule(LinearLayer(1, name='output'))
    arousal_net.addModule(SigmoidLayer(3, name='sigmoid'))
    arousal_net.addModule(TanhLayer(3, name='tanh'))
    arousal_net.addConnection(FullConnection(arousal_net['bias'], arousal_net['sigmoid']))
    arousal_net.addConnection(FullConnection(arousal_net['bias'], arousal_net['tanh']))
    arousal_net.addConnection(FullConnection(arousal_net['input'], arousal_net['sigmoid']))
    arousal_net.addConnection(FullConnection(arousal_net['sigmoid'], arousal_net['tanh']))
    arousal_net.addConnection(FullConnection(arousal_net['tanh'], arousal_net['output']))
    arousal_net.sortModules()

    valence_net = FeedForwardNetwork()
    valence_net.addInputModule(LinearLayer(num_inputs, name='input'))
    valence_net.addInputModule(BiasUnit(name='bias'))
    valence_net.addOutputModule(LinearLayer(1, name='output'))
    valence_net.addModule(SigmoidLayer(3, name='sigmoid'))
    valence_net.addModule(TanhLayer(3, name='tanh'))
    valence_net.addConnection(FullConnection(valence_net['bias'], valence_net['sigmoid']))
    valence_net.addConnection(FullConnection(valence_net['bias'], valence_net['tanh']))
    valence_net.addConnection(FullConnection(valence_net['input'], valence_net['sigmoid']))
    valence_net.addConnection(FullConnection(valence_net['sigmoid'], valence_net['tanh']))
    valence_net.addConnection(FullConnection(valence_net['tanh'], valence_net['output']))
    valence_net.sortModules()

    return arousal_net, valence_net

# Trains 2 separate neural nets (one for arousal, one for valence)
# Data set entry format: 
# ((audio_arousal, audio_valence, lyric_arousal, lyric_valence), target_arousal, target_valence)
def train_separate_nets(data_set, test_data, arousal_net, valence_net, epochs=1):
    num_inputs = 4
    arousal_ds = SupervisedDataSet(num_inputs, 1)
    valence_ds = SupervisedDataSet(num_inputs, 1)
    for i in range(len(data_set)):
        try:
            arousal_ds.appendLinked(data_set[i], (data_set[i][1]))
            valence_ds.appendLinked(data_set[i], (data_set[i][2]))
        except:
            continue
    print str(len(arousal_ds)) + ' points successfully aquired for arousal analysis'
    print str(len(valence_ds)) + ' points successfully aquired for valence analysis'

    arousal_trainer = BackpropTrainer(arousal_net, learningrate=0.05, momentum=0.08, verbose=True)
    valence_trainer = BackpropTrainer(valence_net, learningrate=0.01, momentum=0.05, verbose=True)

    arousal_trainer.trainOnDataset(arousal_ds)
    valence_trainer.trainOnDataset(valence_ds)
    mean_internal_errors = []
    mean_errors = []

    for j in range(epochs/50):
        arousal_trainer.trainEpochs(50)    
        valence_trainer.trainEpochs(50)
        print str((j+1)*50) + '/' + str(epochs) + ' complete'
        sq_arousal_errors = [(arousal_net.activate(datum[0])-datum[1])**2 for datum in test_data]
        sq_valence_errors = [(valence_net.activate(datum[0])-datum[2])**2 for datum in test_data]
        errors = [sqrt(sq_arousal_errors[i] + sq_valence_errors[i]) for i in range(len(sq_arousal_errors))]
        mean_errors.append(np.mean(errors))

        sq_arousal_errors = [(arousal_net.activate(data_set[i][0])-data_set[i][1])**2 for i in range(len(data_set))]
        sq_valence_errors = [(valence_net.activate(data_set[i][0])-data_set[i][2])**2 for i in range(len(data_set))]
        errors = [sqrt(sq_arousal_errors[i] + sq_valence_errors[i]) for i in range(len(sq_arousal_errors))]
        mean_internal_errors.append(np.mean(errors))

    return arousal_net, valence_net, mean_errors, mean_internal_errors

if __name__ == '__main__':
    option = raw_input('Overwrite existing audio networks? [y/n]')
    
    if option == 'y':
        append_or_write = 'w'
    else:
        append_or_write = 'a'

    dict_data_set = construct_data_set()
    data_set = dict_data_set.values()
    shuffle(data_set)
    partition_pos = int((.1)*len(data_set))
    test_data = data_set[:partition_pos]
    training_data = data_set[partition_pos:]

    with open('Results/errors_over_time_combined.txt', append_or_write) as errors_out:
        with open('Results/internal_errors_over_time_combined.txt', append_or_write) as internal_errors_out:
            if option == 'n':
                with open('Networks/arousal_network.nn','r') as arousal_file:
                    with open('Networks/valence_network.nn','r') as valence_file:
                        arousal_net = cPickle.load(arousal_file)
                        valence_net = cPickle.load(valence_file)
            if option == 'y':
                arousal_net, valence_net = build_new_nets()
            with open('Networks/arousal_network.nn', 'w') as arousal_out:
                with open('Networks/valence_network.nn', 'w') as valence_out:
                    arousal_net, valence_net, mean_errors, mean_internal_errors = train_separate_nets(training_data,test_data,
                    arousal_net,valence_net,2500)
                    cPickle.dump(arousal_net,arousal_out)
                    cPickle.dump(valence_net,valence_out)
            for error in mean_errors:
                errors_out.write(str(error) + ',')
            for error in mean_internal_errors:
                internal_errors_out.write(str(error) + ',')