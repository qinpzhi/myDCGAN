# -*- coding: utf-8 -*-
'''
@Time    : 18-11-8 下午4:56
@Author  : qinpengzhi
@File    : util.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
import tensorflow as tf
from tensorflow.contrib import *
##线性函数
def linear(input,output_size,scope=None,stddev=0.02,bias_start=0.0):
    shape= input.get_shape().as_list()
    ##tf.random_normal_initializer() is the same as tf.RandomNormal()
    ##tf.constant_initializer() is the same as tf.Constant()
    with tf.variable_scope(scope or "Linear"):
        matrix=tf.get_variable(
            "Matrix",[shape[1],output_size],tf.float32,
            tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias",[output_size],initializer=tf.constant_initializer(bias_start))
        ##tf.multiply是点乘.tf.matmul是矩阵乘法
        return tf.matmul(input,matrix)+bias

##batch_norm
def batch_norm(input,epsilon=1e-5,momentum=0.9,scope=None,train=True):
    # is_training:图层是否处于训练模式。在训练模式下，它将积累转入的统计量moving_mean并
    # moving_variance使用给定的指数移动平均值
    # decay。当它不是在训练模式，那么它将使用的数值moving_mean和moving_variance。
    return tf.contrib.layers.batch_norm(
            input,decay=momentum, updates_collections=None,
            epsilon=epsilon,scale=True,is_training=train)

##deconv2d
def deconv2d(input,output_shape,
             k_h=5,k_w=5,d_h=2,d_w=2,stddev=0.02,name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[k_h,k_w,output_shape[-1],input.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv=tf.nn.conv2d_transpose(input,w,output_shape=output_shape,
                                      strides=[1,d_h,d_w,1])
        biases=tf.get_variable('biases',[output_shape[-1]],initializer=tf.constant_initializer(0.0))
        ##tf.nn.bias_add是将偏差项加到value上，是tf.add的一个特例，其中bias必须是一维的
        deconv=tf.reshape(tf.nn.bias_add(deconv,biases),deconv.get_shape())
        return deconv

##leaky relu。leaky relu 的 α的取值为０.2。
def lrelu(x,leak=0.2,name="lrelu"):
    return tf.maximum(x,leak*x)

##conv2d将pool层融合在了stride中
def conv2d(input,output_dim,k_h=5,k_w=5,d_h=2,d_w=2,stddev=0.02,name="conv2d"):
    with tf.variable_scope(name):
        w=tf.get_variable('w',[k_h,k_w,input.get_shape()[-1],output_dim],
                          initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv=tf.nn.conv2d(input,w,strides=[1,d_h,d_w,1],padding="SAME")
        biases=tf.get_variable('biases',[output_dim],initializer=tf.constant_initializer(0.0))
        conv=tf.reshape(tf.nn.bias_add(conv,biases),conv.get_shape())
        return conv