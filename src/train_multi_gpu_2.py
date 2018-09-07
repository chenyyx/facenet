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

from tensorflow.python.ops import data_flow_ops

from six.moves import xrange  # @UnresolvedImport

def _from_tensor_slices(tensors_x,tensors_y):
    #return TensorSliceDataset((tensors_x,tensors_y))
    return tf_data.Dataset.from_tensor_slices((tensors_x,tensors_y))



def main(args):
  
    # network = importlib.import_module(args.model_def)

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
    utils.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    train_set = facenet.get_dataset(args.data_dir)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)

    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))
    
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # Placeholders for the learning rate, batch_size, phase_train, image_path, labels
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name='labels')

        input_queue = data_flow_op.FIFOQueue(capacity=100000,
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
                images = tf.image.decode_image(file_contents, channels=3)

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

        # 将数据进行拆分，用于运行在不同的 gpu 上
        image_batch_split = []
        labels_batch_split = []
        with tf.device('/cpu:0'):
            # 将 images_batch 和 labels_batch 进行拆分，得到 num_gpus 份数据，分别用来进行计算 prelogits
            
            if args.num_gpus > 1 :
                for i in range(args.num_gpus):
                    image_tmp = image_batch[i::args.num_gpus]
                    image_batch_split.append(image_tmp)
                    
                    labels_tmp = labels_batch[i::args.num_gpus]
                    labels_batch_split.append(labels_tmp)
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
        for i in range(args.num_gpus):
            with tf.device("/gpu:" + str(i+2)):
                with tf.name_scope("tower_" + str(i)) as scope:
                  with slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0"):
                    with tf.variable_scope(tf.get_variable_scope()) as var_scope:
                        # Build the inference graph
                        with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
                            prelogits, _ = network.inference(image_batch_split[i], args.keep_probability, 
                                phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                                weight_decay=args.weight_decay)

                            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
                            # Split embeddings into anchor, postive and negative and calculate triplet loss
                            anchor, positive, negative = tf.unstack(tf.shape(embeddings, [-1,3,args.embedding_size]), 3, 1)
                            triplet_loss_split = facenet.triplet_loss(anchor, positive, negative, args.alpha)
                            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                            tower_triplet.append(triplet_loss_split)
                            loss = triplet_loss_split + tf.add_n(regularization_losses)
                            tower_losses.append(loss)
                            tower_reg.append(regularization_losses)
                            # 同名变量可以重用
                            tf.get_variable_scope().reuse_variables()
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
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))        

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        
        with sess.as_default():

            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train multi gpus for one epoch
                train_multi_gpu(args, sess, epoch,
                                learning_rate_placeholder, phase_train_placeholder, global_step,
                                losses, train_op, summary_op, summary_writer, args.learning_rate_schedule_file)
                # Train for one epoch
                train(args, sess, epoch, 
                     learning_rate_placeholder, phase_train_placeholder, global_step, 
                     losses, train_op, summary_op, summary_writer, args.learning_rate_schedule_file)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

    return model_dir

def train_multi_gpu(args, sess, dataset, epoch, 
          learning_rate_placeholder, phase_train_placeholder, global_step, 
          loss, train_op, summary_op, summary_writer, learning_rate_schedule_file):
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
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size, 
                learning_rate_placeholder: lr, phase_train_placeholder: True})
            emb_array[lab,:] = emb
        print('%.3f' % (time.time()-start_time))

        


def train(args, sess, epoch, 
          learning_rate_placeholder, phase_train_placeholder, global_step, 
          loss, train_op, summary_op, summary_writer, learning_rate_schedule_file):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = utils.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        start_time = time.time()
        
        print('Running forward pass on sampled images: ', end='')
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True}
        start_time = time.time()
        total_err, reg_err, _, step = sess.run([loss['total_loss'], loss['total_reg'], train_op, global_step ], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tTotal Loss %2.3f\tReg Loss %2.3f, lr %2.5f' %
                  (epoch, batch_number+1, args.epoch_size, duration, total_err, reg_err, lr))

        batch_number += 1
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
        help='Directory where to write event logs.', default='logs/facenet_ms_mp')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='models/facenet_ms_mp')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=.9)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--loss_type', type=str,
        help='Which type loss to be used.',default='softmax')
    parser.add_argument('--network', type=str,
        help='which network is used to extract feature.',default='resnet50')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--list_file', type=str,
        help='Image list file')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
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
        help='Number of gpus.', default=4)
    parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=5)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=600)
    parser.add_argument('--alpha', type=float,
        help='Margin for cos margin.', default=0.15)
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
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM','SGD'],
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

    
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))