# -*- coding: utf-8 -*-
'''
@Time    : 18-11-8 下午4:17
@Author  : qinpengzhi
@File    : main.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
import tensorflow as tf
import pprint
from model import DCGAN

flags=tf.app.flags
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_float("learning_rate",0.0002,"Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1",0.5,"Momentum term of adam [0.5]")
# flags.DEFINE_integer("input_height", 128, "The size of image to use (will be center cropped). [108]")
# flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
# flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
# flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("data_dir","/home/qpz/data/gan-data","Root directory of dataset [data]")
flags.DEFINE_string("dataset","celebA","The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")

FLAGS=flags.FLAGS

def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        dcgan=DCGAN(sess)
        if FLAGS.train:
            dcgan.train(FLAGS)
if __name__ == '__main__':
  tf.app.run()