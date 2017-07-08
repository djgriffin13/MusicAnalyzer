from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, BiasUnit, LinearLayer, FullConnection, SigmoidLayer, TanhLayer
import numpy as np
import AudioFeatureExtraction as fe
import datetime
from threading import Thread
import timeout
import os
import cPickle
from random import shuffle
from math import sqrt

def extract_features_range(low, high, tracks):
    timed_feature_func = timeout.timeout(timeout=300)(fe.feature_combined)

    for i in range(low, high):
        # If feature extraction has already happened, move on
        if os.path.exists('Data/Features/' + str(tracks[i]) + '.txt') and os.path.getsize('Data/Features/' + str(tracks[i]) + '.txt') > 0:
            print '**FEATURES ALREADY FOUND FOR ' + str(tracks[i]) + '.wav'
            continue
        # Otherwise, extract features and save to text file
        print "EXTRACTING FEATURES FOR " + str(tracks[i]) + ".wav\t\t" + str(datetime.datetime.now())
        with open('Data/Features/' + str(tracks[i]) + '.txt', 'w') as output:
            try:
                features = timed_feature_func('Data/Samples/' + str(tracks[i]) + '.wav', 100)
            except:
                print '**FEATURE EXTRACTION FOR ' + str(tracks[i]) + '.wav TIMED OUT'
                continue
            for vect in features:
                output.write(','.join(str(val) for val in vect))
                output.write('\n')

def extract_features(num_threads):
    tracks = np.genfromtxt('Data/static_annotations.csv', delimiter=',')[1:,0].astype(int)
    thread_indices = [(i*len(tracks)/num_threads, (i+1)*len(tracks)/num_threads, tracks) for i in range(num_threads)]
    threads = [Thread(target = extract_features_range, args=thread_indices[i]) for i in range(num_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

# Returns a dictionary of the following form:
# Track number -> ([feature_method_1, feature_method_2, feature_method_3], target_arousal, target_valence) 
def construct_data_set():
    annot = np.genfromtxt('Data/static_annotations_shifted.csv', delimiter=',')[1:]
    annot_tracks = annot[:,0].astype(int).tolist()
    annot_arousal = annot[:,5].tolist()
    annot_valence = annot[:,6].tolist()
    data = {}
    for i in range(1,1001):
        if os.path.exists('Data/Features/' + str(i) + '.txt') and \
            os.path.getsize('Data/Features/' + str(i) + '.txt') > 0 and \
            i in annot_tracks:
            print 'READING FILE ' + str(i) 
            with open('Data/Features/' + str(i) + '.txt', 'r') as input:
                line_1 = input.readline().split(',')
                line_2 = input.readline().split(',')
                line_3 = input.readline().split(',')
                features_1 = [float(val) for val in line_1]
                features_2 = [float(val) for val in line_2]
                features_3 = [float(val) for val in line_3]
                data[i] = ((features_1, features_2, features_3), annot_arousal[annot_tracks.index(i)], annot_valence[annot_tracks.index(i)])
    return data

# Train a neural network based on feature vector n in the data set
def train_net(data_set, n, epochs=1):
    num_inputs = len(data_set[0][0][n])
    ds = SupervisedDataSet(num_inputs, 2)
    for i in range(len(data_set)):
        try:
            ds.appendLinked(data_set[i][0][n], (data_set[i][1], data_set[i][2]))
        except:
            continue
    print str(len(ds)) + ' points successfully aquired'

    net = FeedForwardNetwork()
    net.addInputModule(LinearLayer(num_inputs, name='input'))
    net.addInputModule(BiasUnit(name='bias'))
    net.addOutputModule(LinearLayer(2, name='output'))
    net.addModule(SigmoidLayer(int((num_inputs+2)/2.), name='sigmoid'))
    net.addModule(TanhLayer(10, name='tanh'))
    net.addConnection(FullConnection(net['bias'], net['sigmoid']))
    net.addConnection(FullConnection(net['bias'], net['tanh']))
    net.addConnection(FullConnection(net['input'], net['sigmoid']))
    net.addConnection(FullConnection(net['sigmoid'], net['tanh']))
    net.addConnection(FullConnection(net['tanh'], net['output']))
    net.sortModules()

    trainer = BackpropTrainer(net, learningrate=0.01, momentum=0.1, verbose=True)

    trainer.trainOnDataset(ds)
    trainer.trainEpochs(epochs)

    return net

def build_new_nets(data_set,n):
    num_inputs = len(data_set[0][0][n])
    arousal_net = FeedForwardNetwork()
    arousal_net.addInputModule(LinearLayer(num_inputs, name='input'))
    arousal_net.addInputModule(BiasUnit(name='bias'))
    arousal_net.addOutputModule(LinearLayer(1, name='output'))
    arousal_net.addModule(SigmoidLayer(int((num_inputs+2)/2.), name='sigmoid'))
    arousal_net.addModule(TanhLayer(10, name='tanh'))
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
    valence_net.addModule(SigmoidLayer(int((num_inputs+2)/2.), name='sigmoid'))
    valence_net.addModule(TanhLayer(10, name='tanh'))
    valence_net.addConnection(FullConnection(valence_net['bias'], valence_net['sigmoid']))
    valence_net.addConnection(FullConnection(valence_net['bias'], valence_net['tanh']))
    valence_net.addConnection(FullConnection(valence_net['input'], valence_net['sigmoid']))
    valence_net.addConnection(FullConnection(valence_net['sigmoid'], valence_net['tanh']))
    valence_net.addConnection(FullConnection(valence_net['tanh'], valence_net['output']))
    valence_net.sortModules()

    return arousal_net, valence_net

# Trains 2 separate neural nets (one for arousal, one for valence)
def train_separate_nets(data_set, test_data, n, arousal_net, valence_net, epochs=1):
    num_inputs = len(data_set[0][0][n])
    arousal_ds = SupervisedDataSet(num_inputs, 1)
    valence_ds = SupervisedDataSet(num_inputs, 1)
    for i in range(len(data_set)):
        try:
            arousal_ds.appendLinked(data_set[i][0][n], (data_set[i][1]))
            valence_ds.appendLinked(data_set[i][0][n], (data_set[i][2]))
        except:
            print 'WARNING: INSUFFICIENT INPUT SIZE'
            continue
    print str(len(arousal_ds)) + ' points successfully aquired for arousal analysis'
    print str(len(valence_ds)) + ' points successfully aquired for valence analysis'

    arousal_trainer = BackpropTrainer(arousal_net, learningrate=0.01, momentum=0.05, verbose=True)
    valence_trainer = BackpropTrainer(valence_net, learningrate=0.01, momentum=0.05, verbose=True)

    arousal_trainer.trainOnDataset(arousal_ds)
    valence_trainer.trainOnDataset(valence_ds)
    mean_internal_errors = []
    mean_errors = []

    # Calculate initial error
    sq_arousal_errors = [(arousal_net.activate(datum[0][n])-datum[1])**2 for datum in test_data]
    sq_valence_errors = [(valence_net.activate(datum[0][n])-datum[2])**2 for datum in test_data]
    errors = [sqrt(sq_arousal_errors[i] + sq_valence_errors[i]) for i in range(len(sq_arousal_errors))]
    mean_errors.append(np.mean(errors))

    sq_arousal_errors = [(arousal_net.activate(data_set[i][0][n])-data_set[i][1])**2 for i in range(len(data_set))]
    sq_valence_errors = [(valence_net.activate(data_set[i][0][n])-data_set[i][2])**2 for i in range(len(data_set))]
    errors = [sqrt(sq_arousal_errors[i] + sq_valence_errors[i]) for i in range(len(sq_arousal_errors))]
    mean_internal_errors.append(np.mean(errors))
    
    for j in range(epochs/50):
        arousal_trainer.trainEpochs(50)
        valence_trainer.trainEpochs(50)
        print 'Method ' + str(n) + ' - ' + str((j+1)*50) + '/' + str(epochs) + ' complete'
        sq_arousal_errors = [(arousal_net.activate(datum[0][n])-datum[1])**2 for datum in test_data]
        sq_valence_errors = [(valence_net.activate(datum[0][n])-datum[2])**2 for datum in test_data]
        errors = [sqrt(sq_arousal_errors[i] + sq_valence_errors[i]) for i in range(len(sq_arousal_errors))]
        mean_errors.append(np.mean(errors))

        sq_arousal_errors = [(arousal_net.activate(data_set[i][0][n])-data_set[i][1])**2 for i in range(len(data_set))]
        sq_valence_errors = [(valence_net.activate(data_set[i][0][n])-data_set[i][2])**2 for i in range(len(data_set))]
        errors = [sqrt(sq_arousal_errors[i] + sq_valence_errors[i]) for i in range(len(sq_arousal_errors))]
        mean_internal_errors.append(np.mean(errors))

    return arousal_net, valence_net, mean_errors, mean_internal_errors

if __name__ == '__main__':
    option = raw_input('Overwrite existing audio networks? [y/n]')
    
    if option == 'y':
        append_or_write = 'w'
    else:
        append_or_write = 'a'

    #extract_features(2)
    dict_data_set = construct_data_set()
    data_set = dict_data_set.values()
    shuffle(data_set)
    partition_pos = int((.1)*len(data_set))
    test_data = data_set[:partition_pos]
    training_data = data_set[partition_pos:]

    for i in range(3):
        print 'TRAINING ON FEATURE METHOD ' + str(i)
        with open('Results/errors_over_time_' + str(i) + '.txt', append_or_write) as errors_out:
            with open('Results/internal_errors_over_time_' + str(i) + '.txt', append_or_write) as internal_errors_out:
                if option == 'n':
                    with open('Networks/audio_arousal_network_' + str(i) + '.nn','r') as arousal_file:
                        with open('Networks/audio_valence_network_' + str(i) + '.nn','r') as valence_file:
                            arousal_net = cPickle.load(arousal_file)
                            valence_net = cPickle.load(valence_file)
                if option == 'y':
                    arousal_net, valence_net = build_new_nets(training_data,i)
                with open('Networks/audio_arousal_network_' + str(i) + '.nn', 'w') as arousal_out:
                    with open('Networks/audio_valence_network_' + str(i) + '.nn', 'w') as valence_out:
                        arousal_net, valence_net, mean_errors, mean_internal_errors = train_separate_nets(training_data,test_data,i,
                        arousal_net,valence_net,2500)
                        cPickle.dump(arousal_net,arousal_out)
                        cPickle.dump(valence_net,valence_out)
                for error in mean_errors:
                    errors_out.write(str(error) + ',')
                for error in mean_internal_errors:
                    internal_errors_out.write(str(error) + ',')                      