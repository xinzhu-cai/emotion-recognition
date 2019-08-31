#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tflib as lib
import tflib.ops.linear
import tflib.plot


input_dim = 310
DIM = 512
output_dim = input_dim
LAMBDA = 10

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    output = tf.nn.relu(output)
    return output


class WGAN():
    def __init__(self, BATCH_SIZE, layer_num):
        gen_dict = {3:self.Generator1, 4: self.Generator2, 5:self.Generator3}
        dis_dict = {3:self.Discriminator1, 4: self.Discriminator2, 5:self.Discriminator3}
        self.batch = BATCH_SIZE
        self.real_data = tf.placeholder(tf.float32, shape=[None, input_dim])
        self.fake_data = tf.placeholder(tf.float32, shape=[None, input_dim])
        # condition size = 1, input 1,2,3
        self.ori_label = tf.placeholder(tf.int32, shape=[None, ])
        # change label into ont hot mode
        self.label = tf.one_hot(self.ori_label,3,on_value=1,off_value=None)
        # original: bool, change into float: can be represented as a 1/0 sequence
        self.label = tf.cast(self.label, tf.float32)
        self.gen_data = gen_dict.get(layer_num)(self.fake_data, self.label)
        self.disc_real = dis_dict.get(layer_num)(self.real_data, self.label)
        self.disc_fake = dis_dict.get(layer_num)(self.gen_data, self.label)  
        # WGAN loss
        self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
        self.gen_cost = -tf.reduce_mean(self.disc_fake)    
        disc_params = lib.params_with_name('Discriminator')
        gen_params = lib.params_with_name('Generator')
          
        self.alpha = tf.random_uniform(
            shape=[self.batch,1], 
            minval=0.,
            maxval=1.,
            dtype=tf.float32
        )
        
        self.differences = self.gen_data - self.real_data
        self.interpolates = self.real_data + (self.alpha*self.differences)
        self.label_differences =self.label - self.label
        self.label_interpolates = self.label + (self.alpha*self.label_differences)
        self.gradients = tf.gradients(dis_dict.get(layer_num)(self.interpolates, self.label_interpolates), [self.interpolates, self.label_interpolates])[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean((self.slopes-1.)**2)
        self.disc_cost += LAMBDA*self.gradient_penalty
    
        self.gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5,
            beta2=0.9
        ).minimize(self.gen_cost, var_list=gen_params)
        self.disc_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(self.disc_cost, var_list=disc_params)
    
    def Generator1(self, fake_data, label):
        input_data = tf.concat([fake_data, label], 1)
        # 313 = 310 feature dimensions + 3 label types
        output = ReLULayer('Generator.1', 313, DIM, input_data)
        output = ReLULayer('Generator.2', DIM, DIM, output)
        output = ReLULayer('Generator.3', DIM, DIM, output)
        output = lib.ops.linear.Linear('Generator.4', DIM, output_dim, output)
        return output

    def Discriminator1(self, inputs, label):
        input_data = tf.concat([inputs, label], 1)
        output = ReLULayer('Discriminator.1', 313, DIM, input_data)
        output = ReLULayer('Discriminator.2', DIM, DIM, output)
        output = ReLULayer('Discriminator.3', DIM, DIM, output)
        output = lib.ops.linear.Linear('Discriminator.4', DIM, 1, output)
        return tf.reshape(output, [-1])

    def Generator2(self, fake_data, label):
        input_data = tf.concat([fake_data, label], 1)
        output = ReLULayer('Generator.1', 313, DIM, input_data)
        output = ReLULayer('Generator.2', DIM, DIM, output)
        output = ReLULayer('Generator.3', DIM, DIM, output)
        output = ReLULayer('Generator.4', DIM, DIM, output)
        output = lib.ops.linear.Linear('Generator.5', DIM, output_dim, output)
        return output

    def Discriminator2(self, inputs, label):
        input_data = tf.concat([inputs, label], 1)
        output = ReLULayer('Discriminator.1', 313, DIM, input_data)
        output = ReLULayer('Discriminator.2', DIM, DIM, output)
        output = ReLULayer('Discriminator.3', DIM, DIM, output)
        output = ReLULayer('Discriminator.4', DIM, DIM, output)
        output = lib.ops.linear.Linear('Discriminator.5', DIM, 1, output)
        return tf.reshape(output, [-1])
    
    def Generator3(self, fake_data, label):
        input_data = tf.concat([fake_data, label], 1)
        output = ReLULayer('Generator.1', 313, DIM, input_data)
        output = ReLULayer('Generator.2', DIM, DIM, output)
        output = ReLULayer('Generator.3', DIM, DIM, output)
        output = ReLULayer('Generator.4', DIM, DIM, output)
        output = ReLULayer('Generator.5', DIM, DIM, output)
        output = lib.ops.linear.Linear('Generator.6', DIM, output_dim, output)
        return output

    def Discriminator3(self, inputs, label):
        input_data = tf.concat([inputs, label], 1)
        output = ReLULayer('Discriminator.1', 313, DIM, input_data)
        output = ReLULayer('Discriminator.2', DIM, DIM, output)
        output = ReLULayer('Discriminator.3', DIM, DIM, output)
        output = ReLULayer('Discriminator.4', DIM, DIM, output)
        output = ReLULayer('Discriminator.5', DIM, DIM, output)
        output = lib.ops.linear.Linear('Discriminator.6', DIM, 1, output)
        return tf.reshape(output, [-1])
    