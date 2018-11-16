# -*- coding: utf-8 -*-
'''
@Time    : 18-11-12 下午2:13
@Author  : qinpengzhi
@File    : utils.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
import scipy.misc
import numpy as np
#通过图片地址得到图片，crop默认是true，就是找到图片中心，切割108*108的图片像素
#再将切割的像素进行resize到64
def get_image(image_path, input_height=108, input_width=108,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width,
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w

#将生成的图片进行保存，因为samples是同时生成64张图片，
# 因此将生成8*8的格子图片，每个位置保存一张图
def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
  return (images+1.)/2.

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def merge(images,size):
    h,w=images.shape[1],images.shape[2]
    c=images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx,image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img