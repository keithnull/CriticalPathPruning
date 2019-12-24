import pickle
import random
import json
import keras
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

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

# ================== VGG Network Model with Control Gates ================


class Model():
    def __init__(self, learning_rate=0.1, L1_loss_penalty=0.02, threshold=0.1):
        # record gate variables
        self.AllGateVariables = dict()
        self.AllGateVariableValues = list()
        # hyperparameters
        self.learning_rate = learning_rate
        self.L1_loss_penalty = L1_loss_penalty
        self.threshold = threshold  # Lambda control gate threshold
        # build and restore graph only once
        self.graph = tf.Graph()
        self.build_model(self.graph, 100)
        self.restore_model(self.graph)

    # ===================================================================
    # Encode and save for a batch of data
    # ===================================================================

    def compute_encoding(self, data_input):
        for gate_variable in self.AllGateVariables.values():
            gate_variable.load([1.0, ] * gate_variable.get_shape().as_list()[0], self.sess)
        generatedGates = self.encode_input(data_input)
        return generatedGates

    def encode_class_data(self, class_id, train_images):
        for i in range(len(train_images)):
            generatedGate = self.compute_encoding(train_images[i].reshape((1, 32, 32, 3)))
            # generatedGate is a list of dicts{layername:xx, shape:xx, lambda:xx}
            picname = "class" + str(class_id) + "-pic" + str(i)
            jsonpath = "./ImageEncoding/" + picname + ".json"
            with open(jsonpath, 'w') as f:
                json.dump(generatedGate, f, sort_keys=True, indent=4, separators=(',', ':'))

    # ===================================================================
    # Wrap Functions:
    #     1. model.build_model()
    #     2. model.restore_model()
    #     3. model.encode_input()
    # ===================================================================

    def restore_model(self, graph):
        """
        Restore the original VGG16 network weights
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)
        self.sess.run(self.init)
        with graph.as_default():
            savedVariable = {}
            for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                variable = i
                name = i.name
                if name == 'pl:0':
                    continue
                if name in self.AllGateVariables:
                    continue
                if len(name) >= 8 and name[-11:] == '/Momentum:0':
                    name_prefix = name[:-11]
                    name_prefix += ':0'
                    if name_prefix in self.AllGateVariables:
                        continue
                name = i.name[:-2]
                savedVariable[name] = variable
            # only restore variables in savedVariable
            saver = tf.train.Saver(savedVariable)
            saver.restore(self.sess, "vggNet/augmentation.ckpt-120")

    def build_model(self, graph, label_count):
        """
        Build VGG Network with Control Gate Lambdas
        """
        with graph.as_default():
            # Place Holders:
            #     1. input_x: data
            #     2. input_y: original predicted labels
            #     3. learning rate
            #     4. drop keeping probability: no drop layer actually
            #     5. whether in training mode: always False
            #     6. penalty: regularization
            self.xs = tf.placeholder("float", shape=[None, 32, 32, 3])
            self.ys_orig = tf.placeholder("float", shape=[None, label_count])
            self.lr = tf.placeholder("float", shape=[])
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder("bool", shape=[])
            self.penalty = tf.placeholder(tf.float32)

            # VGG Network Model Construction with Control Gates
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
                self.ys_pred = tf.matmul(current, Wfc) + bfc

            self.ys_pred_softmax = tf.nn.softmax(self.ys_pred)

            # Loss Definition
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.ys_pred, labels=self.ys_orig
            ))
            l1_loss = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in self.AllGateVariableValues])
            self.l1_loss = l1_loss * self.penalty
            self.total_loss = self.l1_loss + self.cross_entropy

            # Optimizer
            self.train_step = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True).minimize(self.total_loss, var_list=self.AllGateVariables.values())

            # Check correctness
            correct_prediction = tf.equal(tf.argmax(self.ys_orig, 1), tf.argmax(self.ys_pred, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            # Init
            self.init = tf.global_variables_initializer()

    def encode_input(self, input_data):
        """
        Given an image input
        Produce a lambda code
        """
        learning_rate = self.learning_rate
        L1_loss_penalty = self.L1_loss_penalty
        threshold = self.threshold

        label_orig = self.sess.run(self.ys_pred_softmax, feed_dict={
            self.xs: input_data,
            self.lr: learning_rate,
            self.keep_prob: 1.0,
            self.is_training: False,
            self.penalty: L1_loss_penalty
        })

        tmpLoss = 1000
        generateGate = dict()
        for epoch in range(100):
            if epoch == 50:
                learning_rate /= 10
                # L1_loss_penalty *= 10
            self.sess.run(self.train_step, feed_dict={
                self.xs: input_data,
                self.ys_orig: label_orig,
                self.lr: learning_rate,
                self.keep_prob: 1.0,
                self.is_training: False,
                self.penalty: L1_loss_penalty
            })
            cross_entropy, L1_loss, accuracy = self.sess.run([self.cross_entropy, self.l1_loss, self.accuracy], feed_dict={
                self.xs: input_data,
                self.ys_orig: label_orig,
                self.lr: learning_rate,
                self.keep_prob: 1.0,
                self.is_training: False,
                self.penalty: L1_loss_penalty
            })
            print("Epoch: {}: Cross_Entropy: {}, L1_loss: {}, Accuracy: {}".format(
                epoch, cross_entropy, L1_loss, accuracy))
            newGate = []
            for gate in self.AllGateVariables.keys():
                tmp = self.AllGateVariables[gate].eval(session=self.sess)
                tmp[tmp < threshold] = 0
                tmp = tmp.tolist()
                res = dict()
                res["layer_name"] = gate
                res["layer_lambda"] = tmp
                res["shape"] = len(tmp)
                newGate.append(res)
            if L1_loss == 'nan' or L1_loss > tmpLoss:
                continue
            if accuracy > 0.99 and L1_loss != 'nan' and L1_loss < 1000:
                generateGate = newGate
                tmpLoss = L1_loss
        # now generatedGate is a list [(layer name, lambda), ...]
        return generateGate

    def close_sess(self):
        """
        Close Session
        """
        self.sess.close()

    # ===================================================================
    # Helper Builder Functions: to build model more conveniently
    # ===================================================================

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=False)

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=False)

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name=name, initializer=initial, trainable=False)

    def gate_variable(self, length, name='gate'):
        initial = tf.constant([1.0] * length)
        v = tf.get_variable(name=name, initializer=initial)
        self.AllGateVariables[v.name] = v
        v = tf.abs(v)
        v = v - tf.constant([0.01]*length)
        v = tf.nn.relu(v)
        self.AllGateVariableValues.append(v)
        return v

    def conv2d(self, input, in_features, out_features, kernel_size, with_bias=False):
        W = self.weight_variable_msra([kernel_size, kernel_size, in_features, out_features], name='kernel')
        conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
        gate = self.gate_variable(out_features)
        conv = tf.multiply(conv, tf.abs(gate))
        if with_bias:
            return conv + self.bias_variable([out_features])
        return conv

    def batch_activ_conv(self, current, in_features, out_features, kernel_size, is_training, keep_prob):
        with tf.variable_scope("composite_function", reuse=tf.AUTO_REUSE):
            current = self.conv2d(current, in_features, out_features, kernel_size)
            current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None, trainable=False)
            # convValues.append(current)
            current = tf.nn.relu(current)
            # current = tf.nn.dropout(current, keep_prob)
        return current

    def batch_activ_fc(self, current, in_features, out_features, is_training):
        Wfc = self.weight_variable_xavier([in_features, out_features], name='W')
        bfc = self.bias_variable([out_features])
        current = tf.matmul(current, Wfc) + bfc
        gate = self.gate_variable(out_features)
        current = tf.multiply(current, tf.abs(gate))
        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None, trainable=False)
        current = tf.nn.relu(current)
        return current

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='VALID')

    # ===================================================================
    # Helper Data Processing Functions
    # ===================================================================

    # calculate the means and stds for the whole dataset per channel
    def measure_mean_and_std(self, images):
        means = []
        stds = []
        for ch in range(images.shape[-1]):
            means.append(np.mean(images[:, :, :, ch]))
            stds.append(np.std(images[:, :, :, ch]))
        return means, stds

    # normalization for per channel
    def normalize_images(self, images):
        images = images.astype('float64')
        means, stds = self.measure_mean_and_std(images)
        for i in range(images.shape[-1]):
            images[:, :, :, i] = ((images[:, :, :, i] - means[i]) / stds[i])
        return images
