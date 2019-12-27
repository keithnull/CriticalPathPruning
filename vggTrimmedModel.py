# from decimal import *
import json
import os
import pickle
import random
import time

import keras
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

model_structure = [
    {'name': 'Conv1/composite_function/gate:0', 'shape': 64},
    {'name': 'Conv2/composite_function/gate:0', 'shape': 64},
    {'name': 'Conv3/composite_function/gate:0', 'shape': 128},
    {'name': 'Conv4/composite_function/gate:0', 'shape': 128},
    {'name': 'Conv5/composite_function/gate:0', 'shape': 256},
    {'name': 'Conv6/composite_function/gate:0', 'shape': 256},
    {'name': 'Conv7/composite_function/gate:0', 'shape': 256},
    {'name': 'Conv8/composite_function/gate:0', 'shape': 512},
    {'name': 'Conv9/composite_function/gate:0', 'shape': 512},
    {'name': 'Conv10/composite_function/gate:0', 'shape': 512},
    {'name': 'Conv11/composite_function/gate:0', 'shape': 512},
    {'name': 'Conv12/composite_function/gate:0', 'shape': 512},
    {'name': 'Conv13/composite_function/gate:0', 'shape': 512},
    {'name': 'FC14/gate:0', 'shape': 4096},
    {'name': 'FC15/gate:0', 'shape': 4096},
]


class TrimmedModel():
    '''
    This class does:
        1. Load original model graph
        2. Assign new weights to layers
        3. Test Accuracy
    '''

    def __init__(self, centroid):
        self.prune_ratio = 0.99

        self.graph = tf.Graph()
        self.build_model(self.graph)
        print("restored the pretrained model......")
        self.restore_model(self.graph)
        # centroid: an (1,12416) numpy array
        self.centroid = centroid

    '''
    mask by value
    '''
    def mask_cluster(self):
        
        def prune(gate_value, ratio):
            """
            gate_value: a 1-d list 
            """
            r_ratio = 1 - ratio
            m = max(gate_value)
            res = []
            for v in gate_value:
                if v < m * r_ratio:
                    res.append(0)
                else: res.append(v)
            return res 

        idx = 0
        pruned_gates = []
        nonzero_entry = 0
        for layer in model_structure:
            shape = layer["shape"]
            name = layer["name"]
            gate_value = self.centroid[idx: idx+shape].tolist()
            # print(len(gate_value))
            idx += shape
            
            pruned_gate_value = prune(gate_value, self.prune_ratio)
            nonzero_entry += np.count_nonzero(np.array(pruned_gate_value))
            pruned_gates.append({
                "name": name, 
                "shape": pruned_gate_value
                })
        print("Sparsity: {}".format(nonzero_entry/12416)) 

        return pruned_gates
    
    '''
    Assign trimmed weight to weight variables
    '''
    def assign_weight(self):
        print("assign weights......")
        maskDict = self.mask_cluster()
        # print(maskDict)

        for tmpLayer in maskDict:
            # if the layer is convolutional layer
            if (tmpLayer["name"][0] == "C"):  
                with self.graph.as_default():
                    layerNum = tmpLayer["name"].split("/")[0].strip("Conv")
                    name = "Conv" + layerNum + "/composite_function/kernel:0"
                    for var in tf.global_variables():

                        if var.name == name:
                            tmpWeights = self.sess.run(var)
                            tmpMask = np.array(tmpLayer["shape"])
                            tmpWeights[:,:,:, tmpMask == 0] = 0
                            assign = tf.assign(var, tmpWeights)
                            
                            self.sess.run(assign)

            # if the layer is fully connected
            if (tmpLayer["name"][0] == "F"):  
                with self.graph.as_default():
                    layerNum = tmpLayer["name"].split("/")[0].strip("FC")
                    name_W = "FC" + layerNum + "/W:0"
                    name_bias = "FC" + layerNum + "/bias:0"
                    for var in tf.global_variables():
                        if var.name == name_W:
                            tmpWeights = self.sess.run(var)
                            tmpMask = np.array(tmpLayer["shape"])
                            tmpWeights[:, tmpMask == 0] = 0
                            assign = tf.assign(var, tmpWeights)
                            self.sess.run(assign)

                        if var.name == name_bias:
                            tmpBias = self.sess.run(var)
                            tmpMask = np.array(tmpLayer["shape"])
                            tmpBias[tmpMask == 0] = 0
                            assign = tf.assign(var, tmpBias)
                            self.sess.run(assign)
                            
        
        print("assign finished!")
        '''
        Save the model
        '''
        # with self.graph.as_default():
        #     saver = tf.train.Saver(max_to_keep = None)
        #     saver.save(self.sess, 'vggNet/test.ckpt')

    '''
    Test Accuracy
    '''

    # def test_accuracy_pretrim(self, test_images, test_labels):

    #     start = time.time()
    #     accuracy = self.sess.run(
    #             self.accuracy, 
    #             feed_dict={
    #             self.xs: test_images,
    #             self.ys_true: test_labels,
    #             self.lr: 0.1,
    #             self.is_training: False,
    #             self.keep_prob: 1.0
    #         })

    #     end = time.time()
    #     print("Pretrimmed Model Test Time: " + str(end - start))

    #     print(accuracy)

    def test_cluster(self, test_images, test_labels):
        """
        return prediction in ((label, credit),(label, credit),....)
        """
        start = time.time()
        ys_pred = self.sess.run(
                self.ys_pred, 
                    feed_dict={
                    self.xs: test_images,
                    self.ys_true: test_labels,
                    self.lr: 0,
                    self.is_training: False,
                    self.keep_prob: 1.0}
                )
        end = time.time()
        print("Trimmed Model Test Time: " + str(end - start))

        ys_pred = np.array(ys_pred)
        res = []
        for pred in ys_pred:
            label = np.argmax(pred)
            credit = pred[label]
            res.append((label, credit))
        
        return tuple(res)
    '''
    Build VGG Network without Control Gate Lambdas
    '''

    def build_model(self, graph, label_count=100):
        with graph.as_default():
            weight_decay = 5e-4
            self.xs = tf.placeholder("float", shape=[None, 32, 32, 3])
            self.ys_true = tf.placeholder("float", shape=[None, label_count])
            self.lr = tf.placeholder("float", shape=[])
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder("bool", shape=[])

            with tf.variable_scope("Conv1", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(self.xs, 3, 64, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv2", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 64, 64, 3, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv3", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 64, 128, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv4", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 128, 128, 3, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv5", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 128, 256, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv6", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 256, 256, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv7", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 256, 256, 1, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv8", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 256, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv9", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv10", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 1, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv11", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv12", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv13", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 1, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
                current = tf.reshape(current, [-1, 512])
            with tf.variable_scope("FC14", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_fc(current, 512, 4096, self.is_training)
            with tf.variable_scope("FC15", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_fc(current, 4096, 4096, self.is_training)
            with tf.variable_scope("FC16", reuse=tf.AUTO_REUSE):
                Wfc = self.weight_variable_xavier([4096, label_count], name='W')
                bfc = self.bias_variable([label_count])
                ys_pred = tf.matmul(current, Wfc) + bfc

            '''
            Loss Function
            '''
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.ys_true, logits=ys_pred
            ))
            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            total_loss = l2_loss * weight_decay + cross_entropy

            '''
            Optimizer
            '''
            train_step = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True).minimize(total_loss)

            '''
            Accuracy & Top-5 Accuracy
            '''
            self.ys_pred = ys_pred 
            # self.ys_pred_argmax = tf.argmax(ys_pred, 1)
            # self.ys_true_argmax = tf.argmax(self.ys_true, 1)
            correct_prediction = tf.equal(tf.argmax(ys_pred, 1), tf.argmax(self.ys_true, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            top5 = tf.nn.in_top_k(predictions=ys_pred, targets=tf.argmax(self.ys_true, 1), k=5)
            top_5 = tf.reduce_mean(tf.cast(top5, 'float'))

            self.init = tf.global_variables_initializer()

    '''
    Restore the original network weights
    '''

    def restore_model(self, graph):
        # If GPU is needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)
        # Else if CPU needed
        # self.sess = tf.Session(graph = graph)
        self.sess.run(self.init)

        with graph.as_default():
            saver = tf.train.Saver(max_to_keep=None)

            saver.restore(self.sess, "vggNet/augmentation.ckpt-120")
            print("restored successfully!")

    '''
    Close Session
    '''

    def close_sess(self):
        self.sess.close()

    '''
    Helper Functions: to build model
    '''

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=False)

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=False)

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name=name, initializer=initial, trainable=False)

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='VALID')

    def conv2d(self, input, in_features, out_features, kernel_size, with_bias=False):
        W = self.weight_variable_msra([kernel_size, kernel_size, in_features, out_features], name='kernel')
        conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
        if with_bias:
            return conv + self.bias_variable([out_features])
        return conv

    def batch_activ_conv(self, current, in_features, out_features, kernel_size, is_training, keep_prob):
        with tf.variable_scope("composite_function", reuse=tf.AUTO_REUSE):
            current = self.conv2d(current, in_features, out_features, kernel_size)
            current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
            current = tf.nn.relu(current)
            #current = tf.nn.dropout(current, keep_prob)
        return current

    def batch_activ_fc(self, current, in_features, out_features, is_training):
        Wfc = self.weight_variable_xavier([in_features, out_features], name='W')
        bfc = self.bias_variable([out_features])
        current = tf.matmul(current, Wfc) + bfc
        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
        current = tf.nn.relu(current)
        return current
