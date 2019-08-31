#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import os
def load_data(path):
    train_data  = sio.loadmat(path)['train_de']
    train_label = sio.loadmat(path)['train_label_eeg']
    test_data = sio.loadmat(path)['test_de']
    test_label = sio.loadmat(path)['test_label_eeg']
    # 15 person 45 experiments
    person_count = 45
    return train_data, train_label, test_data, test_label, person_count,

class DataSampler(object):
    def __init__(self, num):
        data_path  = "smooth_split/"
        fileNames = os.listdir(data_path)
        fileNames.sort()
        all_train_data, all_train_label, all_test_data, all_test_label, all_person_count = load_data(data_path+fileNames[num])
        
        train_test_data = np.concatenate((all_train_data, all_test_data), axis=0)
        train_test_data = (train_test_data - np.mean(train_test_data))/np.std(train_test_data)

        all_train_label = all_train_label + np.ones_like(all_train_label)
        all_test_label = all_test_label + np.ones_like(all_test_label)
        self.all_data = train_test_data
        self.train_data = train_test_data[:2010, :]
        self.train_label = all_train_label
        self.test_data =  train_test_data[2010:, :]
        self.test_label = all_test_label
        self.all_label = np.concatenate((all_train_label, all_test_label), axis=0)



class NoiseSampler(object):
    def __init__(self, BATCH_SIZE):
        test_data = np.random.uniform(-1.0, 1.0, [BATCH_SIZE, 310])                    
        self.all_data = test_data