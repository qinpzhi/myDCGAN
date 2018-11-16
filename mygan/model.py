# -*- coding: utf-8 -*-
'''
@Time    : 18-11-8 下午4:39
@Author  : qinpengzhi
@File    : model.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
import tensorflow as tf
from glob import glob
import numpy as np
import time
import os
from ops import *
from utils import *
class DCGAN(object):
    #sample是需要测试的图片数量，batch_size是需要训练的图片数量
    def __init__(self,sess,batch_size=64,input_height=108,input_width=108,output_height=64,
                 output_width=64,sample_num=64):
        self.sess = sess
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.sample_num=sample_num
        self.build_model()

    def build_model(self):
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size, self.output_height, self.output_width, 3], name='real_images')
        ##生成一个100维的向量，这和论文上的一样
        self.z = tf.placeholder(tf.float32, [None, 100], name='z')
        self.G = self.generator(self.z)
        self.sampler = self.sampler(self.z)
        self.D,self.D_logits = self.discriminator(self.inputs,reuse=False)
        self.D_,self.D_logits_=self.discriminator(self.G,reuse=True)

        self.d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logits,labels=tf.ones_like(self.D)))
        self.d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logits_,labels=tf.zeros_like(self.D_)))
        self.g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logits_,labels=tf.ones_like(self.D_)))
        self.d_loss=self.d_loss_real+self.d_loss_fake

        self.z_sum=tf.summary.histogram("z",self.z)
        self.d_sum=tf.summary.histogram("d",self.D)
        self.d__sum=tf.summary.histogram("d_",self.D_)
        self.G_sum=tf.summary.image("G",self.G)
        self.d_loss_real_sum=tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum=tf.summary.scalar("d_loss_fake",self.d_loss_fake)
        self.d_loss_sum=tf.summary.scalar("d_loss",self.d_loss)
        self.g_loss_sum=tf.summary.scalar("g_loss",self.g_loss)

        ##这一点特别重要，因为在指定训练的时候需要指定要调节的参数

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver=tf.train.Saver()

    #通过100的向量生成对应的图片，在生成器（G）中，输出层使用Tanh函数,其余层采用 ReLu 函数
    def generator(self,z):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = s_h / 2, s_w / 2
            s_h4, s_w4 = s_h / 4, s_w / 4
            s_h8, s_w8 = s_h / 8, s_w / 8
            s_h16, s_w16 = s_h / 16, s_w / 16
            h0 = linear(z, 512 * s_h16 * s_w16, 'g_ho_lin')
            h0 = tf.reshape(h0, [-1, s_h16, s_w16, 512])
            h0 = tf.nn.relu(batch_norm(h0))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, 256], name='g_h1')
            h1 = tf.nn.relu(batch_norm(h1))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, 128], name='g_h2')
            h2 = tf.nn.relu(batch_norm(h2))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, 64], name='g_h3')
            h3 = tf.nn.relu(batch_norm(h3))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, 3], name='g_h4')
            return tf.nn.tanh(h4)

    #判别式函 数判别器（D）中都采用leaky rectified activation
    def discriminator(self,image,reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(conv2d(image, 64, name='d_h0_conv'))
            h1 = lrelu(batch_norm(conv2d(h0, 128, name='d_h1_conv')))
            h2 = lrelu(batch_norm(conv2d(h1, 256, name='d_h2_conv')))
            h3 = lrelu(batch_norm(conv2d(h2, 512, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
            return tf.nn.sigmoid(h4),h4

    #和generator 内容一样，将generator的模型参数重新reuse就可以
    def sampler(self,z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = s_h / 2, s_w / 2
            s_h4, s_w4 = s_h / 4, s_w / 4
            s_h8, s_w8 = s_h / 8, s_w / 8
            s_h16, s_w16 = s_h / 16, s_w / 16
            h0 = linear(z, 512 * s_h16 * s_w16, 'g_ho_lin')
            h0 = tf.reshape(h0, [-1, s_h16, s_w16, 512])
            h0 = tf.nn.relu(batch_norm(h0))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, 256], name='g_h1')
            h1 = tf.nn.relu(batch_norm(h1))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, 128], name='g_h2')
            h2 = tf.nn.relu(batch_norm(h2))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, 64], name='g_h3')
            h3 = tf.nn.relu(batch_norm(h3))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, 3], name='g_h4')
            return tf.nn.tanh(h4)

    #训练函数，在main函数中调用来训练
    def train(self,config):
        d_optim=tf.train.AdamOptimizer(learning_rate=config.learning_rate,
                                       beta1=config.beta1).minimize(self.d_loss,var_list=self.d_vars)
        g_optim=tf.train.AdamOptimizer(learning_rate=config.learning_rate,
                                       beta1=config.beta1).minimize(self.g_loss,var_list=self.g_vars)
        tf.global_variables_initializer().run()
        self.g_sum=tf.summary.merge([self.z_sum,self.d__sum,self.G_sum,
                                     self.d_loss_fake_sum,self.g_loss_sum])
        self.d_sum=tf.summary.merge([self.z_sum,self.d_sum,self.d_loss_real_sum,
                                     self.d_loss_sum])
        self.writer=tf.summary.FileWriter("./logs",self.sess.graph)

        ##弄一批验证集进行验证
        sample_z=np.random.uniform(-1,1,size=(self.sample_num,100))
        dataTotal = glob(os.path.join(config.data_dir, config.dataset, "*.jpg"))
        sample_files=dataTotal[0:self.sample_num]
        sample=[get_image(sample_file) for sample_file in sample_files]
        sample_inputs=np.array(sample).astype(np.float32)

        for epoch in xrange(0,config.epoch):
            self.data=glob(os.path.join(config.data_dir,config.dataset,"*.jpg"))
            np.random.shuffle(self.data)
            ##" // "表示整数除法
            batch_idxs=len(self.data)/config.batch_size
            ##每一轮设置计数器
            counter=1
            start_time=time.time()
            could_load, checkpoint_counter = self.load(config.checkpoint_dir)
            if could_load:
                counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            for idx in xrange(0,int(batch_idxs)):
                batch_files= self.data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch=[get_image(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                #从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
                batch_z=np.random.uniform(-1,1,[config.batch_size,100]).astype(np.float32)

                ##update Dicriminator network
                ## global_step,当前迭代的轮数，需要注意的是，如果没有这个参数，那么scalar的summary将会成为一条直线
                _,summary_str=self.sess.run([d_optim,self.d_sum],feed_dict={self.z:batch_z,self.inputs:batch_images})
                self.writer.add_summary(summary_str,counter)

                ##update Generator network
                _,summary_str=self.sess.run([g_optim,self.g_sum],feed_dict={self.z:batch_z})
                self.writer.add_summary(summary_str,counter)
                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _,summary_str= self.sess.run([g_optim,self.g_sum], feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str,counter)

                ##eval是tensorflow中启动计算的值
                errD_fake=self.d_loss_fake.eval({ self.z: batch_z })
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})

                counter+=1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, config.epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter,100)==1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={
                            self.z: sample_z,
                            self.inputs: sample_inputs,
                        },
                    )
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    save_images(sample_inputs, image_manifold_size(sample_inputs.shape[0]),
                                './{}/train_{:02d}_{:04d}_1.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter,500)==1:
                    self.save(config.checkpoint_dir,counter)
    #保存模型
    def save(self,checkpoint_dir,step):
        model_name="DCGAN.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,os.path.join(checkpoint_dir,model_name),global_step=step)

    #从checkpoint中获取已经存在的模型
    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

