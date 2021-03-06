#!/usr/bin/python
# -*- coding:utf-8 -*-

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
import numpy as np
import importlib
import itertools
import argparse
import facenet
import lfw
from tensorflow.python.ops import data_flow_ops
from six.moves import xrange  # @UnresolvedImport

import inception_resnet_v1 as inception_net

# 设置 程序 可见的 GPU，指定为 2,3
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"


def main(args):
    # 导入 model_def 指代的网络结构
    network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # 将 参数 写入 text 文件中
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # 存储一些 git 修订信息到 log 文件夹的 text 文件中
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    # 获取 训练数据集
    train_set = facenet.get_dataset(args.data_dir)

    # 打印 Model 存储目录 和 Log 存储目录
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    # 检测是否 已经有训练完成的 model 了
    # if args.pretrained_model:
    #     print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))

    # if args.lfw_dir:
    #     print('LFW directory: %s' % args.lfw_dir)
    #     # Read the file containing the pairs used for testing
    #     pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
    #     # Get the paths for the corresponding images
    #     lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
    #     print('lfw_paths_len^^^^^^^^^^^^^^', len(lfw_paths))
    #     print('lfw_paths^^^^^^^^^^^^^^', lfw_paths)
    #     print('actual_issame_len^^^^^^^^^^', len(actual_issame))
    #     print('actual_issame^^^^^^^^^^', actual_issame)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name='image_paths')

        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(3,), (3,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)

                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)

                # pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])
        # tf.train.batch_join() --- 运行 tensor list 来填充队列，以创建样本的批次。返回与 tensors_list[i] 有着相同数量和类型的张量的列的列表或者字典。
        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        # tf.identity(input, name=None) --- 返回一个与 input 具有相同 shape 和 内容的 tensor
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        # 首先在 一个 gpu 上运行 network 的 inference graph
        with tf.device('/cpu:0'):
            embeddings = []
            anchors = []
            positives = []
            negatives = [] 
        # 在 gpu 0 上进行运行 inference graph
        with tf.device('/gpu:2'):
            # 创建 inference graph
            prelogits, _ = network.inference(image_batch, args.keep_probability,
                phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                weight_decay=args.weight_decay)
            # l2 normalize
            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
            # ------ print start -----------
            print('embeddings-------', embeddings)
            # ------ print end -------------
            # Split embedding into anchor, positive and negative and calculate teiplet loss
            anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, args.embedding_size]), 3, 1)
            # ------------------- print start ---------------------------------
            # print('@@@@@@@@@@@@wwww', anchor.get_shape())
            # print('############wwww', positive.get_shape())
            # print('$$$$$$$$$$$$wwww', negative.get_shape())
            # ------------------- print end -----------------------------------

            # 接着进行拆分，将 anchor, positive, negative 三组数据进行拆分，拆分成对应着 num_gpus 的份数
            # 但是这里有一个限制，所切分的 tensor 的维度，必须可以被 切分成的份数 整除， 所以直接使用 tf.split() 会报错
                      
            for i in range(args.num_gpus):
                anchors_tmp = anchor[i::args.num_gpus]
                anchors.append(anchors_tmp)
                positives_tmp = positive[i::args.num_gpus]
                positives.append(positives_tmp)
                negatives_tmp = negative[i::args.num_gpus]
                negatives.append(negatives_tmp)

        # -------print start 查看 emdedding 和 相对应的 3 个 anchors，positive，negative
        # print('wowowowoowowow', embeddings.get_shape())
        # print('wowninininini_anchors0', anchors[0].get_shape())
        # print('wowninininini_anchors1', anchors[1].get_shape())
        # print('wowninininini_positives0', positives[0].get_shape())
        # print('wowninininini_positives1', positives[1].get_shape())
        # print('wowninininini_negatives0', negatives[0].get_shape())
        # print('wowninininini_negatives1', negatives[1].get_shape())
        # ------ print end -------------------------
        # 在这部分修改为 multi-gpu 训练
        # 即 将每一个 batch 均分，然后在多个 gpu 上来计算对应的 triplet_loss ，之后汇总得到和，求取平均，得到一个 batch_size 的 loss
        # tower_losses = []
        # tower_triplets = []
        # tower_reg = []
        # for i in range(args.num_gpus):
        #     with tf.device("/gpu:" + str(i)):
        #         with tf.name_scope("tower_" + str(i)) as scope:
        #             with tf.variable_scope(tf.get_variable_scope()) as var_scope:                   
        #                 triplet_loss_split = facenet.triplet_loss(anchors[i], positive[i], negative[i], args.alpha)
        #                 regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #                 tower_triplets.append(triplet_loss_split)
        #                 loss = triplet_loss_split + tf.add_n(regularization_loss)
        #                 tower_losses.append(loss)
        #                 tower_reg.append(regularization_loss)
        # # 计算 multi gpu 运行完成得到的 loss
        # total_loss = tf.reduce_mean(tower_losses)
        # total_reg = tf.reduce_mean(tower_reg)
        # losses = {}
        # losses['total_loss'] = total_loss
        # losses['total_reg'] = total_reg
        

        # # 创建 inference graph
        # prelogits, _ = network.inference(image_batch, args.keep_probability,
        #     phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
        #     weight_decay=args.weight_decay)
        
        # embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # # Split embedding into anchor, positive and negative and calculate teiplet loss
        # anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, args.embedding_size]), 3, 1)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # # 计算 total losses
        # regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        # 创建 使用 1 个 batch 来训练模型并更新其中参数的 Graph
        # train_op = facenet.train(total_loss, global_step, args.optimizer,
            # learning_rate, args.moving_average_decay, tf.global_variables())

        # 在多 gpu 上训练的改版
        train_op, total_loss = facenet.train_multi_gpu(args.num_gpus, anchors, positives, negatives, args.alpha, global_step, args.optimizer, learning_rate, args.moving_average_decay, tf.global_variables())

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # 基于 TF Summaries 的 collection 创建一个 summary opration
        summary_op = tf.summary.merge_all()

        # 在 Graph 启动 running operations
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))

        # 初始化 variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            # if args.pretrained_model:
            #     print('Restoring pretrained model: %s' % args.pretrained_model)
            #     saver.restore(sess, os.path.expanduser(args.pretrained_model))
                # facenet.load_model(args.pretrained_model)
            
            #Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                # --- print start ---
                print('@@@@@@@@@@@@@@args.embedding_size', args.embedding_size)
                print('@@@@@@@@@@@@@@anchor', anchor, anchor.get_shape())
                print('@@@@@@@@@@@@@@positive', positive, positive.get_shape())
                print('@@@@@@@@@@@@@@negative', negative, negative.get_shape())
                print('@@@@@@@@@@@@@@total_loss', total_loss, total_loss.get_shape())
                # --- print end -----

                epoch_start_time = time.time()
                # origin
                # train(args, sess, train_set, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
                #         batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step, 
                #         embeddings, total_loss, train_op, summary_op, summary_writer, args.learning_rate_schedule_file,
                #         args.embedding_size, anchor, positive, negative, total_loss)
                # print('第 %d 个 epoch跑完，花费时间 %.3f' %(epoch, (time.time() - epoch_start_time)))

                # modified training
                train_multi_gpus(args, sess, train_set, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
                        batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step, 
                        embeddings, anchors, positives, negatives, train_op, summary_op, summary_writer, args.learning_rate_schedule_file,
                        args.embedding_size)
                print('第 %d 个 epoch跑完，花费时间 %.3f' %(epoch, (time.time() - epoch_start_time)))
                
                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                # Evaluate on LFW
                # if args.lfw_dir:
                #     evaluate(sess, lfw_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder, 
                #                 batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame, args.batch_size, 
                #                 args.lfw_nrof_folds, log_dir, step, summary_writer, args.embedding_size)

    return model_dir

def train(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step, 
          embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size, anchor, positive, negative, triplet_loss):
    batch_number = 0

    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people(dataset, args.people_per_batch, args.images_per_person)

        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()
        nrof_examples = args.people_per_batch * args.images_per_person
        labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
        emb_array = np.zeros((nrof_examples, embedding_size))
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
        # ---------- print start -------------
        # print('nnnnnnnrof_batches', nrof_batches)
        # print('nnnnnnnrof_examples', nrof_examples)
        # print('bbbbbbatch_size', args.batch_size)
        
        # ---------- print end ---------------
        for i in range(nrof_batches):
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            # ------------ print start ------------
            # print('batch_size-=-=-=-=-=-', batch_size)
            # print('embedding-=-=-=-', embeddings)
            # print('labels_batch-=-=-=-=', labels_batch)
            # ------------ print end ---------------
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder:batch_size,
                learning_rate_placeholder: lr, phase_train_placeholder: True})
            # print('embbbbbbbbbbb', emb)
            # print('labbbbbbbbbbb', lab)
            # print('emb_array-----', emb_array)
            emb_array[lab,:] = emb
        print('加载数据完毕，用时 %.3f' % (time.time()-start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training...')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class, 
            image_paths, args.people_per_batch, args.alpha)
        selection_time = time.time() - start_time
        print('选择 三元组 完毕 --- (nrof_random_negs, nrif_triplets) = (%d, %d): time=%.3f seconds' %
            (nrof_random_negs, nrof_triplets, selection_time))

        # 在选定的 triplets 上执行 训练 
        nrof_batches = int(np.ceil(nrof_triplets*3/args.batch_size))
        print('选择出来的 triplets 形成的 batch 有多少个：', nrof_batches)
        triplet_paths = list(itertools.chain(*triplets))
        # print('triplet_paths:', triplet_paths)
        labels_array = np.reshape(np.arange(len(triplet_paths)),(-1,3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths),1), (-1,3))
        # 把 选择出来的 triplets 添加到 enqueue 中去
        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        nrof_examples = len(triplet_paths)
        print('选择出的 triplets 的个数为：', nrof_examples)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        summary = tf.Summary()
        step = 0
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
            # --- print start ---
            print('**************batch_size', batch_size)
            print('**************lr', lr)
            print('**************loss', loss, loss.get_shape())
            print('**************train_op', train_op)
            print('**************global_step', global_step)
            print('**************embeddings', embeddings, embeddings.get_shape())
            print('**************labels_batch', labels_batch, labels_batch.get_shape())
            # --- print end -----
            err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch], feed_dict=feed_dict)
            emb_array[lab,:] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, batch_number+1, args.epoch_size, duration, err))
            batch_number += 1
            i += 1
            train_time += duration
            summary.value.add(tag='loss', simple_value=err)
            
        # Add validation loss and accuracy to summary
        #pylint: disable=maybe-no-member
        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
    return step

def train_multi_gpus(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step, 
          embeddings, anchors, positives, negatives, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size):
    batch_number = 0

    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people(dataset, args.people_per_batch, args.images_per_person)

        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()
        nrof_examples = args.people_per_batch * args.images_per_person
        labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
        emb_array = np.zeros((nrof_examples, embedding_size))
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
        
        for i in range(nrof_batches):
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder:batch_size,
                learning_rate_placeholder: lr, phase_train_placeholder: True})
            emb_array[lab,:] = emb
        print('加载数据完毕，用时 %.3f' % (time.time()-start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training...')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class, 
            image_paths, args.people_per_batch, args.alpha)
        selection_time = time.time() - start_time
        print('选择 三元组 完毕 --- (nrof_random_negs, nrif_triplets) = (%d, %d): time=%.3f seconds' %
            (nrof_random_negs, nrof_triplets, selection_time))

        # 在选定的 triplets 上执行 训练 
        nrof_batches = int(np.ceil(nrof_triplets*3/args.batch_size))
        print('选择出来的 triplets 形成的 batch 有多少个：', nrof_batches)
        triplet_paths = list(itertools.chain(*triplets))
        # print('triplet_paths:', triplet_paths)
        labels_array = np.reshape(np.arange(len(triplet_paths)),(-1,3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths),1), (-1,3))
        # 把 选择出来的 triplets 添加到 enqueue 中去
        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        nrof_examples = len(triplet_paths)
        print('选择出的 triplets 的个数为：', nrof_examples)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        summary = tf.Summary()
        step = 0
        while i < nrof_batches:
            start_time = time.time()

            # 添加 计算 loss 的代码，加在此处，是因为每个 epoch 中的每个 batch 都要计算一下 loss
            # First, calculate the total_loss to run the following code
            # 在这部分修改为 multi-gpu 训练
            # 即 将每一个 batch 均分，然后在多个 gpu 上来计算对应的 triplet_loss ，之后汇总得到和，求取平均，得到一个 batch_size 的 loss
            tower_losses = []
            tower_triplets = []
            tower_reg = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(args.num_gpus):
                    with tf.device('/gpu:' + str(i+2)):
                        with tf.name_scope('tower_' + str(i)) as scope:
                            print('###'*10, i)
                            print('---'*10, anchors[i])
                            print('---'*10, positives[i])
                            print('---'*10, negatives[i])
                            print('###'*10, i)
                            triplet_loss_split = facenet.triplet_loss(anchors[i], positives[i], negatives[i], args.alpha)
                            regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                            # 添加本行代码，进行同名变量的重用
                            tf.get_variable_scope().reuse_variables()
                            tower_triplets.append(triplet_loss_split)
                            loss = triplet_loss_split + tf.add_n(regularization_loss)
                            tower_losses.append(loss)
                            tower_reg.append(regularization_loss)
            # 计算 multi gpu 运行完成得到的 loss
            total_loss = tf.reduce_mean(tower_losses)
            total_reg = tf.reduce_mean(tower_reg)
            losses = {}
            losses['total_loss'] = total_loss
            losses['total_reg'] = total_reg

            loss = total_loss

            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
            # --- print start ---
            print('**************batch_size', batch_size)
            print('**************lr', lr)
            print('**************loss', loss, loss.get_shape())
            print('**************train_op', train_op)
            print('**************global_step', global_step)
            print('**************embeddings', embeddings, embeddings.get_shape())
            print('**************labels_batch', labels_batch, labels_batch.get_shape())
            # --- print end -----
            err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch], feed_dict=feed_dict)
            emb_array[lab,:] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, batch_number+1, args.epoch_size, duration, err))
            batch_number += 1
            i += 1
            train_time += duration
            summary.value.add(tag='loss', simple_value=err)
            
        # Add validation loss and accuracy to summary
        #pylint: disable=maybe-no-member
        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
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
    # ----------- print start -------------
    # print('ttttttttriplet', len(triplets))
    # print('num_trips', num_trips)
    # print('wodemengxiang')
    # ----------- print end ---------------
    triplets_yyx = triplets[0:180]
    # return triplets, num_trips, len(triplets)
    return triplets, num_trips, len(triplets_yyx)

# 从数据集中进行抽样图片，参数为训练数据集，每一个 batch 抽样多少人，每个人抽取多少张
def sample_people(dataset, people_per_batch, images_per_person):
    # 总共应该抽取多少张，默认：people_per_batch 是 45，images_per_person 是 40
    nrof_images = people_per_batch * images_per_person
    # print('!!!!!!!!!!!!!nrof_images', nrof_images)
    # Sample classes from the dataset
    # 数据集中一共有多少人的图像
    nrof_classes = len(dataset)
    # print('!!!!!!!!!!!!!nrof_classes', nrof_classes)
    # 每个人的索引
    class_indices = np.arange(nrof_classes)
    # 随机打乱一下
    np.random.shuffle(class_indices)
    # print('!!!!!!!!!!!!!!!!',class_indices)

    i = 0
    # 保存抽样出来的图像的路径
    image_paths = []
    # 抽样的样本是属于哪一个人的，作为 label
    num_per_class = []
    sampled_class_indices = []
    # Sample image from these classes until we have enough
    # 不断抽样直到达到指定数量
    while len(image_paths)<nrof_images:
        # ------------- print start -----------
        # print('leniiiiii', len(image_paths))
        # print('imagesiiii', nrof_images)
        # print('iiiiiiiiii', i)
        # ------------- print end --------------
        # 从第 i 个人开始抽样
        class_index = class_indices[i]
        # 第 i 个人有多少张图片
        nrof_images_in_class = len(dataset[class_index])
        # 这些图片的索引
        image_indices = np.arange(nrof_images_in_class)
        # 将图片的索引进行一下打乱
        np.random.shuffle(image_indices)
        # 从第 i 个人中抽样的图片数量
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        # 抽样出来的人的图片的路径
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        image_paths += image_paths_for_class
        # 第 i 个人抽样了多少张
        num_per_class.append(nrof_images_from_class)
        i += 1
    # ------------ print start ------------
    # print('iiiimage_paths',len(image_paths))
    # print('nnnnum_per_class', num_per_class)
    # ------------ print end --------------
    return image_paths, num_per_class

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
    # ------------- print start -----------------
    # print('nrof_images^^^^^', nrof_images)
    # print('batch_size^^^^^^', batch_size)
    # print('nrof_batches^^^^', nrof_batches)
    # ------------- print end -------------------
    label_check_array = np.zeros((nrof_images,))
    for i in xrange(nrof_batches):
        batch_size = min(nrof_images-i*batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
            learning_rate_placeholder: 0.0, phase_train_placeholder: False})
        emb_array[lab,:] = emb
        label_check_array[lab] = 1
    print('用时：%.3f' % (time.time()-start_time))
    # --------- print start ---------------------
    # print('label_check_array^^^^', label_check_array)
    # print('if label-check_array == 1', np.all(label_check_array==1))
    # --------- print end -----------------------
    
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

# 从文件中获取 learning rate
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


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='logs/triplet_cy')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='models/triplets')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.9)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.', default='models/inception_resnet_v1_triplet_112_0,1_64._2._0.2_ADAM_--fc_bn_96_128/20180904-191008/model-20180904-191008.ckpt-273')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=20)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=180)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=45)
    parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=40)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=200)
    parser.add_argument('--alpha', type=float,
        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
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
    parser.add_argument('--num_gpus', type=int,
        help='Number of gpus.', default=2)
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='../data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='/home/face/cosface_tf/dataset/lfw-112X96')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))