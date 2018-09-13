#!/usr/bin/python
# -*- coding:utf-8 -*-

"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
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
from tensorflow.contrib import slim

from tensorflow.python.ops import data_flow_ops

from six.moves import xrange  # @UnresolvedImport


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


def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person
  
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    
    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
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



def main(args):
  
    network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    train_set = facenet.get_dataset(args.data_dir)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)

    # # 预训练模型
    # if args.pretrained_model:
    #     print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))
    
    # #  lfw 数据集的位置
    # if args.lfw_dir:
    #     print('LFW directory: %s' % args.lfw_dir)
    #     # Read the file containing the pairs used for testing
    #     pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
    #     # Get the paths for the corresponding images
    #     lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # Placeholders for the learning rate, batch_size, phase_train, image_path, labels
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

        # 读取数据的线程数
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

        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        # ------- new start --------------------
        # 说明：1.image_batch 是 image 数据的 batch，2.labels_batch 是对应于 image 数据的 label 的 batch
        # 接下来，将 image_batch 和 label_batch 进行切分，也就是说，
        # 在之前每个批次的数据上进行切分，将一个 batch 分成若干份，放到不同的 gpu 上运行，以达到提速的目的

        # 将 image_batch 和 labels_batch 进行拆分，得到 num_gpus 份数据，分别用来进行计算 prelogits
        # if args.num_gpus > 1:
            # for i in range(args.num_gpus):
            #     image_tmp = image_batch[i::args.num_gpus]
            #     image_batch_split.append(image_tmp)

            #     labels_tmp = labels_batch[i::args.num_gpus]
            #     labels_batch_split.append(labels_tmp)
        # ------- new end ------------------------------

        # # 将 labels_batch 进行 reshape
        # labels_batch_split = tf.reshape(labels_batch, [args.batch_size])
        # # 构建 训练 的输入队列，每次取一个 image_batch 和一个 labels_batch
        # batch_queue = tf.contirb.slim.prefetch_queue.prefetch_queue(
        #     [image_batch, labels_batch_split], capacity=2 * args.num_gpus)

        # learning rate
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        print('Using optimizer: {}'.format(args.optimizer))
        if args.optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif args.optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif args.optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif args.optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif args.optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise Exception('Not supported optimizer: {}'.format(args.optimizer))
        # 在这部分进行 multi_gpu 
        tower_losses = []
        tower_triplet = []
        tower_reg= []
        # embeddings_split = []
        for i in range(args.num_gpus):
            with tf.device("/gpu:" + str(i)):
                with tf.name_scope("tower_" + str(i)) as scope:
                    with tf.variable_scope(tf.get_variable_scope()) as var_scope:
                        # Build the inference graph
                        with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
                            tower_start_time = time.time()
                            # -------- new start ---------------------------------------------
                            # prelogits, _ = network.inference(image_batch_split, args.keep_probability, 
                            #     phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                            #     weight_decay=args.weight_decay)
                            # embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
                            # embeddings_split.append(embeddings)
                            # Split embeddings into anchor, postive and negative and calculate triplet loss
                            # anchor, positive, negative = tf.unstack(tf.reshape(embeddings_split[i], [-1,3,args.embedding_size]), 3, 1)
                            # ------------ new end ----------------------------------------------
                            # ------------ old start --------------------------------------
                            prelogits, _ = network.inference(image_batch, args.keep_probability, 
                                phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                                weight_decay=args.weight_decay)

                            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
                            # Split embeddings into anchor, postive and negative and calculate triplet loss
                            anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,args.embedding_size]), 3, 1)
                            # ------------ old end ----------------------------------------
                            triplet_loss_split = facenet.triplet_loss(anchor, positive, negative, args.alpha)
                            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                            tower_triplet.append(triplet_loss_split)
                            loss = triplet_loss_split + tf.add_n(regularization_losses)
                            tower_losses.append(loss)
                            tower_reg.append(regularization_losses)
                            # 同名变量可以重用
                            tf.get_variable_scope().reuse_variables()
                            print('tower_%s 运行时间为--- %.3f' %(i, time.time()-tower_start_time))
        total_loss = tf.reduce_mean(tower_losses)
        total_reg = tf.reduce_mean(tower_reg)
        losses = {}
        losses['total_loss'] = total_loss
        losses['total_reg'] = total_reg

        grads = opt.compute_gradients(total_loss,tf.trainable_variables(),colocate_gradients_with_ops=True)
        apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group(apply_gradient_op)

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))        

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        
        with sess.as_default():

            # if args.pretrained_model:
            #     print('Restoring pretrained model: %s' % args.pretrained_model)
            #     saver.restore(sess, os.path.expanduser(args.pretrained_model))

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train multi gpus for one epoch
                epoch_start_time = time.time()
                # ----- new start ----------
                # train_multi_gpu(args, sess, train_set, epoch, image_paths_placeholder, labels_placeholder, 
                #             labels_batch_split, batch_size_placeholder, learning_rate_placeholder, 
                #             phase_train_placeholder, enqueue_op, input_queue, global_step, embeddings_split, losses, train_op, 
                #             summary_op, summary_writer, args.learning_rate_schedule_file, args.embedding_size)
                # ------ new end -----------

                # ------- old start -----------
                train_multi_gpu(args, sess, train_set, epoch, image_paths_placeholder, labels_placeholder, 
                            labels_batch, batch_size_placeholder, learning_rate_placeholder, 
                            phase_train_placeholder, enqueue_op, input_queue, global_step, embeddings, losses, train_op, 
                            summary_op, summary_writer, args.learning_rate_schedule_file, args.embedding_size)
                # ------- old end -------------
            
                print('The %dth epoch running time is %.3f seconds!!!' %(epoch, time.time()-epoch_start_time))       
                # # Train for one epoch
                # train(args, sess, epoch, 
                #      learning_rate_placeholder, phase_train_placeholder, global_step, 
                #      losses, train_op, summary_op, summary_writer, args.learning_rate_schedule_file)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

    return model_dir

def train_multi_gpu(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch, 
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, 
          global_step, embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file, embedding_size):
    batch_number = 0

    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people(dataset, args.people_per_batch,args.images_per_person)
        
        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()
        nrof_examples = args.people_per_batch * args.images_per_person
        labels_array = np.reshape(np.arange(nrof_examples),(-1,3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1), (-1,3))
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
        emb_array = np.zeros((nrof_examples, embedding_size))
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
        for i in range(nrof_batches):
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            # ------ new start-----
            # for j in range(args.num_gpus):
            #     emb_tmp, lab_tmp = sess.run([embeddings_split, labels_batch_split], feed_dict={batch_size_placeholder: batch_size,
            #         learning_rate_placeholder: lr, phase_train_placeholder: True})
            # emb_array[lab_tmp,:] = emb_tmp
            # ------ new end --------

            # ------ old start ------
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size, 
                learning_rate_placeholder: lr, phase_train_placeholder: True})
            emb_array[lab,:] = emb
            # ------ old end --------
        print('%.3f' % (time.time()-start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class, 
            image_paths, args.people_per_batch, args.alpha)
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % 
            (nrof_random_negs, nrof_triplets, selection_time))

        # Perform training on the selected triplets
        nrof_batches = int(np.ceil(nrof_triplets*3/args.batch_size))
        print('选择出来的 triplets 形成的 batch 有多少个：', nrof_batches)
        triplet_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplet_paths)),(-1,3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths),1), (-1,3))
        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        nrof_examples = len(triplet_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, args.embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        summary = tf.Summary()
        step = 0
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
            # ---- new start -----
            # total_err_split = []
            # reg_err_split = []
            # emb_split_new = []
            # lab_split_new = []
            # for j in range(args.num_gpus):
            #     if j % 2 == 1:
            #         total_err_tmp, reg_err_tmp, _, step, emb_new_tmp, lab_new_tmp = sess.run([loss['total_loss'], loss['total_reg'], train_op, global_step, embeddings_split[j], labels_batch_split[j]], feed_dict=feed_dict)
            #     else:
            #         total_err_tmp, reg_err_tmp, _, emb_new_tmp, lab_new_tmp = sess.run([loss['total_loss'], loss['total_reg'], train_op, embeddings_split[j], labels_batch_split[j]], feed_dict=feed_dict)
            #     total_err_split.append(total_err_tmp)
            #     reg_err_split.append(reg_err_tmp)
            #     emb_split_new.append(emb_new_tmp)
            #     lab_split_new(lab_new_tmp)
            # total_err = tf.reduce_mean(total_err_split)
            # reg_err = tf.reduce_mean(reg_err_split)
            # 将 tensor 转换为 float
            # total_err_float = tf.to_float(total_err, name='totensor_1')
            # reg_err_float = tf.to_float(reg_err, name='totensor_2')
            # emb = tf.reduce_mean(emb_split_new, 0)
            # lab = tf.reduce_mean(lab_split_new, 0)
            # ---- new end -------

            # ----- old start -----
            total_err, reg_err, _, step, emb, lab = sess.run([loss['total_loss'], loss['total_reg'], train_op, global_step, embeddings, labels_batch], feed_dict=feed_dict)
            # ----- old end -------         
            duration = time.time() - start_time
            # print('SSSSSSSSSSSSSSSStep======', step)
            print('Epoch: [%d][%d/%d]\tTime %.3f\tTotal Loss %2.3f\tReg Loss %2.3f, lr %2.5f' %
                  (epoch, batch_number+1, args.epoch_size, duration, total_err, reg_err, lr))
            batch_number += 1
            i += 1
            train_time += duration
    return step
 

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
        help='Directory where to write event logs.', default='logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.', default='models/inception_resnet_v1_triplet_112_0,1_64._2._0.2_ADAM_--fc_bn_96_128/20180905-132221/model-20180905-132221.ckpt-0')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=1)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=45)
    parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=40)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=545)
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
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--num_gpus', type=int,
        help='how many gpus to run', default=2)

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='../data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='/Users/chenyao/Documents/dataset/lfw/lfw-112X96')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))