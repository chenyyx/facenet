MODEL=models/inception_resnet_v1_triplet_112_0,1_64._2._0.2_ADAM_--fc_bn_96_128/20180905-132221
LFW_DIR=/Users/chenyao/Documents/dataset/lfw/lfw-112X96

python validate_on_lfw.py --lfw_dir ${LFW_DIR} --model ${MODEL} --distance_metric 1 --use_flipped_images --subtract_mean --use_fixed_image_standardization