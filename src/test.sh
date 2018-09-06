MODEL=models/inception_resnet_v1_triplet_112_0,1_64._2._0.2_ADAM_--fc_bn_96_128/20180904-191008
LFW_DIR=/Users/chenyao/Documents/dataset/lfw/lfw-112X96
# IMAGE_WIDTH=96
# IMAGE_HEIGHT=112
# modify from gpu to cpu
# CUDA_VISIBLE_DEVICES=1 python test/test.py ${TEST_DATA} ${MODEL_DIR} --lfw_file_ext jpg --network_type sphere_network --embedding_size ${EMBEDDING_SIZE} ${FC_BN} ${PREWHITEN} --image_height ${IMAGE_HEIGHT} --image_width ${IMAGE_WIDTH}
# 修改打印到日志中
# python validate_on_lfw.py ${TEST_DATA} ${MODEL_DIR} --lfw_file_ext jpg --network_type sphere_network --embedding_size ${EMBEDDING_SIZE} ${FC_BN} ${PREWHITEN} --image_height ${IMAGE_HEIGHT} --image_width ${IMAGE_WIDTH}
# python test/test.py ${TEST_DATA} ${MODEL_DIR} --lfw_file_ext jpg --network_type sphere_network --embedding_size ${EMBEDDING_SIZE} ${FC_BN} ${PREWHITEN} --image_height ${IMAGE_HEIGHT} --image_width ${IMAGE_WIDTH} 2>&1 | tee /Users/chenyao/github/CosFace/log/cosface_testlog_20180824.log

CUDA_VISIBLE_DEVICES=0,1 python validate_on_lfw.py --lfw_dir ${LFW_DIR} --model ${MODEL}