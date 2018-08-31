#!/usr/bin/python
# -*- coding:utf-8 -*-

"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
基于 FaceNet 论文使用 TensorFlow 训练一个 face recognizer。
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import os.path
import time
import sys
sys.path.insert(0, 'models')
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.contrib import slim
from collections import Counter
import numpy as np
import importlib
import itertools
import argparse
import facenet
import lfw
from tensorflow import data as tf_data
# import inception_resnet_v1
import inception_resnet_v1 as inception_net
# import inception_resnet_v2
# import squeezenet

from tensorflow.python.ops import data_flow_ops

from six.moves import xrange  # @UnresolvedImport

# 设置 tf 可见的 GPU，指定为 0，1
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

softmax_ind = 0

# 将数据集进行切分，按照第一个维度进行切分，比如 传入的 size 是 (5, 2) ，切分得到的是 5 个元素，每个的size 是(2, )
def _from_tensor_slices(tensors_x,tensors_y):
    # return TensorSliceDataset((tensors_x,tensors_y))
    return tf_data.Dataset.from_tensor_slices((tensors_x,tensors_y))

def main(args):
    # # 导入 model_def 代表的网络结构
    # network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    # 将 参数 写入到 text 文件中
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    # 将一些 git 修订信息存储在日志目录的文本文件中
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    # 获取 facenet 的数据集
    np.random.seed(seed=args.seed)
    # train_set = facenet.get_dataset(args.data_dir)

    # 训练数据集
    train_set = facenet.dataset_from_list(args.data_dir,args.list_file)
    nrof_classes = len(train_set)
    print('nrof_classes: ',nrof_classes)
    # 获取 图像 的 路径 和 labels
    image_list, label_list = facenet.get_image_paths_and_labels(train_set)
    print('total images: ',len(image_list))
    image_list = np.array(image_list)
    label_list = np.array(label_list,dtype=np.int32)

    dataset_size = len(image_list)
    # 单个 batch_size  = 每个 batch 中的人数 * 每个人的图像数
    single_batch_size = args.people_per_batch*args.images_per_person
    indices = range(dataset_size)
    np.random.shuffle(indices)

    # 从 dataset 中抽取 样本，将 image_path 和 image_label 返回
    def _sample_people_softmax(x):
        global softmax_ind
        if softmax_ind >= dataset_size:
            np.random.shuffle(indices)
            softmax_ind = 0
        true_num_batch = min(single_batch_size,dataset_size - softmax_ind)

        sample_paths = image_list[indices[softmax_ind:softmax_ind+true_num_batch]]
        sample_labels = label_list[indices[softmax_ind:softmax_ind+true_num_batch]]

        softmax_ind += true_num_batch

        return (np.array(sample_paths), np.array(sample_labels,dtype=np.int32))

    def _sample_people(x):
        '''We sample people based on tf.data, where we can use transform and prefetch.
        Desc：
            我们基于 tf.data 对人进行抽样，这样我们可以使用 transform 和 prefetch 。
        '''
    
        image_paths, num_per_class = sample_people(train_set,args.people_per_batch*(args.num_gpus-1),args.images_per_person)
        labels = []
        for i in range(len(num_per_class)):
            labels.extend([i]*num_per_class[i])
        return (np.array(image_paths),np.array(labels,dtype=np.int32))
    
    # 解析函数，将 image 的路径和 label 解析出来，对应着 image 和 label
    def _parse_function(filename,label):
        # 使用 tf.read_file() 进行读取，并使用 tf.image.decode_image() 进行转换为 tensor 的形式
        file_contents = tf.read_file(filename)
        image = tf.image.decode_image(file_contents, channels=3)
        #image = tf.image.decode_jpeg(file_contents, channels=3)
        print(image.shape)
        
        # 判断是否对图像进行随机裁剪
        if args.random_crop:
            print('use random crop')
            image = tf.random_crop(image, [args.image_size, args.image_size, 3])
        else:
            print('Not use random crop')
            #image.set_shape((args.image_size, args.image_size, 3))
            image.set_shape((None,None, 3))
            # 将图片进行 resize ，转换为我们传入的参数的大小
            image = tf.image.resize_images(image, size=(args.image_height, args.image_width))
            #print(image.shape)
        # 判断是否进行随机水平翻转
        if args.random_flip:
            image = tf.image.random_flip_left_right(image)

        #pylint: disable=no-member
        #image.set_shape((args.image_size, args.image_size, 3))
        image.set_shape((args.image_height, args.image_width, 3))
        # 强制转换数据类型
        image = tf.cast(image,tf.float32)
        image = tf.subtract(image,127.5)
        image = tf.div(image,128.)
        #image = tf.image.per_image_standardization(image)
        return image, label
    
    # 将 model 目录和 log 目录先打印一下
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    # # 如果已经提供了 预训练好的模型
    # if args.pretrained_model:
    #     # os.path.expanduser() 把路径中包含 ～ 或者 ～user 的地方转换为用户目录
    #     print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))
    # # 如果提供了 lfw 数据，读取 lfw 目录中的 pairs 和 lfw 数据集中图像的 path
    # if args.lfw_dir:
    #     print('LFW directory: %s' % args.lfw_dir)
    #     # Read the file containing the pairs used for testing
    #     pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
    #     # Get the paths for the corresponding images
    #     lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
        
    
    with tf.Graph().as_default():
        # 设置随机生成数的种子
        tf.set_random_seed(args.seed)
        # 全局的 step
        global_step = tf.Variable(0, trainable=False,name='global_step')

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        
        # batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')


        
        # image_paths_placeholder = tf.placeholder(tf.string, shape=(None,3), name='image_paths')
        # labels_placeholder = tf.placeholder(tf.int64, shape=(None,3), name='labels')
        
        # input_queue = data_flow_ops.FIFOQueue(capacity=100000,
        #                             dtypes=[tf.string, tf.int64],
        #                             shapes=[(3,), (3,)],
        #                             shared_name=None, name=None)
        # enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])
        
        # the image is generated by sequence
        # 在 cpu 中将 训练数据集的 batch 进行拆分，分成 num_gpus 份数据
        with tf.device("/cpu:0"):
            
            softmax_dataset = tf_data.Dataset.range(args.epoch_size*args.max_nrof_epochs*100)
            softmax_dataset = softmax_dataset.map(lambda x: tf.py_func(_sample_people_softmax,[x],[tf.string,tf.int32]))
            softmax_dataset = softmax_dataset.flat_map(_from_tensor_slices)
            softmax_dataset = softmax_dataset.map(_parse_function,num_parallel_calls=2000)
            softmax_dataset = softmax_dataset.batch(args.num_gpus*single_batch_size)
            softmax_iterator = softmax_dataset.make_initializable_iterator()
            softmax_next_element = softmax_iterator.get_next()
            softmax_next_element[0].set_shape((args.num_gpus*single_batch_size, args.image_height,args.image_width,3))
            softmax_next_element[1].set_shape(args.num_gpus*single_batch_size)
            batch_image_split = tf.split(softmax_next_element[0],args.num_gpus)
            batch_label_split = tf.split(softmax_next_element[1],args.num_gpus)


            # # 在整体数据集上选出 3 元组（triplets）
            # select_start_time = time.time()
            # # Select triplets based on the embeddings
            # print('Selecting suitable triplets for training')
            # # 修改版本的 triplets
            # nrof_examples = args.people_per_batch * args.images_per_person
            # triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class, 
            #     image_paths, args.people_per_batch, args.alpha)
            # selection_time = time.time() - start_time
            # print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % 
            #     (nrof_random_negs, nrof_triplets, selection_time))

        # 学习率设置
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        # 优化器设置
        print('Using optimizer: {}'.format(args.optimizer))
        if args.optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif args.optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate,0.9)
        elif args.optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        else:
            raise Exception("Not supported optimizer: {}".format(args.optimizer))

        tower_losses = []
        tower_cross = []
        tower_dist = []
        tower_reg= []
        for i in range(args.num_gpus):
            with tf.device("/gpu:" + str(i)):
                with tf.name_scope("tower_" + str(i)) as scope:
                  with slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0"):
                    with tf.variable_scope(tf.get_variable_scope()) as var_scope:
                        reuse = False if i ==0 else True
                        if args.network == 'inception_resnet_v1':
                            with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
                                prelogits, _ = inception_net.inference(batch_image_split[i], args.keep_probability, 
                                phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                                weight_decay=args.weight_decay)
                            print(prelogits)

                        # elif args.network == 'inception_net_v2':
                        #     with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
                        #         prelogits, _ = inception_net_v2.inference(image_batch, args.keep_probability, 
                        #         phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                        #         weight_decay=args.weight_decay)
                        #     print(prelogits)
                        # elif args.network == 'squeezenet':
                        #     with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
                        #         prelogits, _ = squeezenet.inference(image_batch, args.keep_probability, 
                        #         phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                        #         weight_decay=args.weight_decay)
                        #     print(prelogits)
                        else:
                            raise Exception("Not supported network: {}".format(args.network))
                        if args.fc_bn: 

                            prelogits = slim.batch_norm(prelogits, is_training=True, decay=0.997,epsilon=1e-5,scale=True,updates_collections=tf.GraphKeys.UPDATE_OPS,reuse=reuse,scope='softmax_bn')
                        if args.loss_type == 'triplet':
                            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
                            # Split embeddings into anchor, positive and negative and calculate triplet loss
                            anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,args.embedding_size]), 3, 1)
                            triplet_loss = facenet.triplet_loss(anchor, positive, negative, args.alpha)
                            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                            if args.network == 'sphere_network':
                                print('reg loss using weight_decay * tf.add_n')
                                reg_loss  = args.weight_decay*tf.add_n(regularization_losses)
                            else:
                                print('reg loss using tf.add_n')
                                # reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                                reg_loss = tf.add_n(regularization_losses)
                            loss = triplet_loss + reg_loss
                            
                            tower_losses.append(loss)
                            tower_reg.append(reg_loss)
                        # elif args.loss_type =='cosface':


                        #loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
                        tf.get_variable_scope().reuse_variables()
        # 计算 total loss
        total_loss = tf.reduce_mean(tower_losses)
        total_reg = tf.reduce_mean(tower_reg)
        losses = {}
        losses['total_loss'] = total_loss
        losses['total_reg'] = total_reg
        

        # # Build a Graph that trains the model with one batch of examples and updates the model parameters
        # train_op = facenet.train(total_loss, global_step, args.optimizer, 
        #     learning_rate, args.moving_average_decay, tf.global_variables())

        grads = opt.compute_gradients(total_loss,tf.trainable_variables(),colocate_gradients_with_ops=True)
        apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group(apply_gradient_op)
        
        # Create a saver
        # saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        save_vars = [var for var in tf.global_variables() if 'Adagrad' not in var.name and 'global_step' not in var.name]
        saver = tf.train.Saver(save_vars, max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))        

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})

        sess.run(softmax_iterator.initializer)

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            # 如果有预训练好的 model，那就进行 restore 操作，将模型加载进行以后的 测试阶段
            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, epoch, learning_rate_placeholder, phase_train_placeholder, global_step, losses, train_op, summary_op, summary_writer, args.learning_rate_schedule_file)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                # # Evaluate on LFW
                # if args.lfw_dir:
                #     evaluate(sess, lfw_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder, 
                #             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame, args.batch_size, 
                #             args.lfw_nrof_folds, log_dir, step, summary_writer, args.embedding_size)
    # 将训练好的模型 返回
    return model_dir

# 训练阶段
def train(args, sess, epoch, 
          learning_rate_placeholder, phase_train_placeholder, global_step, 
          loss, train_op, summary_op, summary_writer, learning_rate_schedule_file):
    batch_number = 0
    # learning rate 设置
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        # learning rate 小于 0 的话，就从 文件中获取对应的 learning rate
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    # 在 batch_number 小于 epoch_size（实际上是 每个 epoch 中 btach 的总数）时，进行以下操作
    while batch_number < args.epoch_size:
        start_time = time.time()
        
        print('Running forward pass on sampled images: ', end='')

        # feed 进行传 learning_rate 
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True}
        start_time = time.time()

        # sess.run() 
        total_err, reg_err, _, step = sess.run([loss['total_loss'], loss['total_reg'], train_op, global_step ], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tTotal Loss %2.3f\tReg Loss %2.3f, lr %2.5f' %
                  (epoch, batch_number+1, args.epoch_size, duration, total_err, reg_err, lr))
        # 训练完成一个 batch 之后，进行 batch_number + 1，进入下一个 batch 的运行
        batch_number += 1
    return step
  
def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % 
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)

# 抽取 人脸 images 和对应的 labels
def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person
  
    # Sample classes from the dataset
    # 从 dataset 中抽取类
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    
    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    # 根据 上面我们抽取出的类抽取 images
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1
  
    return image_paths, num_per_class
# 对我们训练完成的 model ，在 lfw 数据集上进行 evaluate 。
def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder, 
        batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame, batch_size, 
        nrof_folds, log_dir, step, summary_writer, embedding_size):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on LFW images: ', end='')
    
    nrof_images = len(actual_issame)*2
    assert(len(image_paths)==nrof_images)
    labels_array = np.reshape(np.arange(nrof_images),(-1,3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1), (-1,3))
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, embedding_size))
    nrof_batches = int(np.ceil(nrof_images / batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in xrange(nrof_batches):
        batch_size = min(nrof_images-i*batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
            learning_rate_placeholder: 0.0, phase_train_placeholder: False})
        emb_array[lab,:] = emb
        label_check_array[lab] = 1
    print('%.3f' % (time.time()-start_time))
    
    assert(np.all(label_check_array==1))
    
    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))

# 保存 variables 和 metagraph
def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  
# 从 file 中获取 learning rate 
def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
    
# 设置 training 阶段需要的参数
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='logs/triplet_cy')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='premodels/triplets')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=.9)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--loss_type', type=str, 
        help='Which type loss to be used.', default='triplet')
    parser.add_argument('--network', type=str, 
        help='which network is used to extract feature.', default='inception_resnet_v1')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='/Users/chenyao/Documents/dataset/CASIA-WebFace/CASIA-WebFace-112X96')
    parser.add_argument('--list_file', type=str,
        help='Image list file')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=3)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--image_src_size', type=int,
        help='Src Image size (height, width) in pixels.', default=256)
    parser.add_argument('--image_height', type=int,
        help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--image_width', type=int,
        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=30)
    parser.add_argument('--num_gpus', type=int,
        help='Number of gpus.', default=2)
    parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=5)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=3)
    parser.add_argument('--alpha', type=float,
        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--scale', type=float,
        help='Scale as the fixed norm of weight and feature.', default=64.)
    parser.add_argument('--weight', type=float,
        help='weiht to balance the dist and th loss.', default=3.)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=256)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--fc_bn', 
        help='Wheater use bn after fc.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM', 'SGD'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')

    # # Parameters for validation on LFW
    # parser.add_argument('--lfw_pairs', type=str,
    #     help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    # parser.add_argument('--lfw_dir', type=str,
    #     help='Path to the data directory containing aligned face patches.', default='')
    # parser.add_argument('--lfw_nrof_folds', type=int,
    #     help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
