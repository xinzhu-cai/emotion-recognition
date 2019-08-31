#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("libsvm/python")
from svm import *
from svmutil import *
import numpy as np
import tensorflow as tf
from my_model import WGAN
from  my_data_handler import DataSampler, NoiseSampler
import random
import scipy.io as sio
import tflib.ops.linear
import tflib.plot
from sklearn import preprocessing
import os
#from svmutil import *
import csv
import tflib as lib

num_iter = 1
layer_num = int(sys.argv[1]) # number of layers in generator 
num = int(sys.argv[2]) # current training example number 0-44

# input number of layesrs, final result filename, current train number 

f = open("result/"+str(num)+"_"+str(layer_num)+".txt","w")

cur_dir = 'gen_data_de/'+str(num)+'/'+str(num_iter)
if not os.path.exists(cur_dir):
    os.makedirs(cur_dir)
    


def train():
      ### initialization for seed and gpu informaiton
    print(num)
    all_acc = []
    all_acc.append(num)
    seed = 666
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed + 2)
    # config the tf.Session when create new session
    config = tf.ConfigProto() 
    # set gpu memory fraction as 0.3, occupy one gpu
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    config.gpu_options.allow_growth = True
    CRITIC_ITERS = 5
    real_x = DataSampler(num)
    
        
    ### get all data of 15 sessions
    trial_start = [0,235,468,678,912,1097,1292,1530,1745,2010,2247,2482,2715,2950,3188,3393]
    trial_node_number = [235,233,210,234,185,195,238,215,265,237,235,233,235,238,205]
    validate = [0,3,6,9,12,15]
    real_alldata = real_x.all_data
    label_alldata = real_x.all_label
    BATCH_SIZE = 512  ### train the whole batch in each iteration to get the distribution
    wgan = WGAN(BATCH_SIZE,layer_num)
    test_cost = []
  #  saver = tf.train.Saver() 
    max_accracy = [0]*5
    max_c = [0]*5
    for i in [4]: #5
        # every select different trials as training set
        print (trial_start[validate[i]])
        print (validate[i+1])
        print (trial_start[validate[i+1]])
        print (real_alldata.shape)
        real_cur = np.concatenate((real_alldata[:trial_start[validate[i]],:],real_alldata[trial_start[validate[i+1]]:,:]),axis = 0)
        print ("real_cur ",len(real_cur))
        print (len(real_cur[0]))
        label_cur =  np.concatenate((label_alldata[:trial_start[validate[i]],:],label_alldata[trial_start[validate[i+1]]:,:]),axis = 0)
        print ("label_cur ",len(label_cur))
        print (len(label_cur[0]))
        test_cur = real_alldata[trial_start[validate[i]]:trial_start[validate[i+1]],:]
        print ("test_cur ",len(test_cur))
        print (len(test_cur[0]))
        test_label_cur = label_alldata[trial_start[validate[i]]:trial_start[validate[i+1]],:]
        print ("test_label_cur ",len(test_label_cur))
        print (len(test_label_cur[0]))
        with tf.Session(config=config) as session:
            lib.plot.clear() ### clear the data in plot
            session.run(tf.global_variables_initializer())
            for iteration in range(20): #1500
                fake_x = NoiseSampler(BATCH_SIZE)
                fake_cur = fake_x.all_data
                # random choose BATCH_SIZE data points
                ch_idx = np.random.choice(len(real_cur), BATCH_SIZE, replace=False)
            # Train generator
                if iteration > 0: # should run discriminator first 
                    _ = session.run(wgan.gen_train_op, feed_dict={wgan.fake_data:fake_cur, wgan.ori_label:label_cur[ch_idx, :].ravel()})
                # Train critic
                for j in range(CRITIC_ITERS): #5
                    _disc_cost, _ = session.run(
                        [wgan.disc_cost, wgan.disc_train_op],
                        feed_dict={wgan.real_data: real_cur[ch_idx, :], wgan.fake_data: fake_cur, wgan.ori_label:label_cur[ch_idx, :].ravel()}
                    )
                lib.plot.plot('result/disc cost '+str(num)+' '+str(layer_num), _disc_cost) # print in .pdf  
                if(iteration%20 == 0):
                    lib.plot.flush(1)
                    print('cur: '+ str(iteration) + '     loss: ' + str(_disc_cost))
                lib.plot.tick()

            ### test ###
            fake_x = NoiseSampler(BATCH_SIZE)
            fake_cur = fake_x.all_data
            _test_cost, _ = session.run(
                        [wgan.disc_cost, wgan.disc_train_op],
                        feed_dict={wgan.real_data:test_cur, wgan.fake_data: fake_cur, wgan.ori_label:test_label_cur.ravel()}
                    )
            test_cost.append(_test_cost)
            content = "validata " + i + "test cost: "+ _test_cost + "\n"
            f.write(content)

            dst_0 = real_cur
            fake_x = NoiseSampler(len(dst_0))
            fake_cur = fake_x.all_data
            dst_1 = fake_cur
            dst_2 = session.run([wgan.gen_data], feed_dict={wgan.fake_data:dst_1, wgan.ori_label:label_cur.ravel()})
            dst_2 = dst_2[0]
            all_real_data = dst_0
            print("all_real_data ",len(all_real_data))
            print(len(all_real_data[0]))
            all_fake_data = dst_1
            print("all_fake_data ",len(all_fake_data))
            print(len(all_fake_data[0]))
            all_gen_data = dst_2
            print("all_gen_data ",len(all_gen_data))
            print(len(all_fake_data[0]))
            all_real_label = label_cur
            print("all_real_label ",len(all_real_label))
            print(len(all_real_label[0]))

            train_number = 0
            for j in range(8):
                train_number = train_number + trial_nnumber[j]
            print (train_number)
            print (trial_nnumber)
            print ((all_real_data[:train_number,:]).shape)
            train_inst = np.concatenate((all_real_data[:train_number,:],all_gen_data[:train_number,:]),axis = 0)
            test_inst = np.concatenate((all_real_data[train_number:,:],all_gen_data[train_number:,:]),axis = 0)
            train_label = np.concatenate((all_real_label[:train_number],all_real_label[:train_number]),axis = 0)
            test_label = np.concatenate((all_real_label[train_number:],all_real_label[train_number:]),axis = 0)
            # SVM training 
            train_inst = preprocessing.scale(train_inst)
            test_inst = preprocessing.scale(test_inst)
            
            train_label = np.transpose(train_label)[0]
            test_label = np.transpose(test_label)[0]

            print ("train_inst ",train_inst.shape)
            print ("train_label ",train_label.shape)
            print ("test_inst ",test_inst.shape)
            print ("test_label ",test_label.shape)

            train_inst = train_inst.tolist()
            train_label = train_label.tolist()

            for c in range(1): #-10,10,0.5
                param = '-t 0 -c %f'%(2**c)  # -c 0.1 -b 1 -s 3
                '''We apply a SVMs with linear kernel as the
                classifier. The parameter c is searched from 2^-10 ~ 2^10 to
                find the optimal value'''
                
                prob  = svm_problem(train_label, train_inst)
                param = svm_parameter(param)
                model = svm_train(prob, param)

                test_label = test_label.tolist()
                test_inst = test_inst.tolist()
                p_label, p_acc, p_val = svm_predict(test_label, test_inst, model)
                print ("accuracy:: ", p_acc )
                if p_acc[0] > max_accracy[i]:
                    max_accracy[i] = p_acc[0]
                    max_c[i] = c
            f.write("%d %f \n"%(i,max_accracy[i]))
            f.write("%d %f \n"%(i,max_c[i]))

    f.write("Average: %f \n"%(sum(max_accracy)/5))

if __name__ == "__main__":
    train()
    tf.reset_default_graph()



    
    
