#!/usr/bin/python3
"""Training and Validation On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import shutil
import argparse
import importlib
import data_utils
import numpy as np
import pointfly as pf
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    #parser.add_argument('--save_folder', '-s', help='Path to folder for saving the generation results', required=True)
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    parser.add_argument('--grid_size', help='Batch size (default defined in setting)', type=int)
    parser.add_argument('--sample_size', default=512, type=int)
    args = parser.parse_args()


    #print('PID:', os.getpid())

    #print(args)

    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    ######################################################################
    # Placeholders
    is_training = tf.placeholder(tf.bool, name='is_training')
    pts_fts = tf.placeholder(tf.float32, shape=(None, args.sample_size, setting.data_dim), name='pts_fts')
    labels = tf.placeholder(tf.int32, shape=(None,), name='labels')
    sigma = tf.placeholder(tf.float32, shape=(None,), name='sigma')
    ######################################################################
    features_augmented = None

    points_augmented = pts_fts
    sigma_unsqueezed = tf.reshape(sigma, shape=(-1, 1, 1))
    perturbed_points = points_augmented
    net = model.Net(perturbed_points, features_augmented, labels, is_training, setting)
    scores = net.logits
    

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=None)


    sigmas = np.exp(np.linspace(np.log(0.3), np.log(0.01), 10))
    label_range = np.arange(10)
    n_steps_each= 100
    step_lr=0.00001

    with tf.Session() as sess:
        sess.run(init_op)
        # Load the model
        if args.load_ckpt is not None:
            saver.restore(sess, tf.train.latest_checkpoint(args.load_ckpt))
            print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))
     
        grid_size = args.grid_size
        results = []
        x_mod = np.random.rand(grid_size ** 2, args.sample_size, 3)
        for i in tqdm(range(sigmas.shape[0])):
            label_used = np.ones(grid_size ** 2) * label_range[i]
            sigma_used = sigmas[label_used.astype(int)]
            step_size = step_lr * (sigmas[i] / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                results.append(np.expand_dims(x_mod,0))
                noise = np.random.randn(*x_mod.shape) * np.sqrt(step_size * 2)
                grad = sess.run([scores],
                            feed_dict={
                                 pts_fts: x_mod,
                                 is_training: False,
                                 labels: label_used,
                                 sigma: sigma_used,
                             })
                grad = grad[0]
                x_mod = x_mod + step_size * grad + noise
        results = np.concatenate(results, axis=0)
        np.save('./generated_point_clouds.npy', results)

if __name__ == '__main__':
    main()
