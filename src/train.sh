# NETWORK=sphere_network
#NETWORK=resface
NETWORK=inception_resnet_v1
#NETWORK=resnet_v2

CROP=112
echo $NAME
# GPU=0
GPU=2,3
NUM_GPUS=2
ARGS="CUDA_VISIBLE_DEVICES=${GPU}"
#WEIGHT_DECAY=1e-3
WEIGHT_DECAY=1e-4
# LOSS_TYPE=cosface
#LOSS_TYPE=softmax
LOSS_TYPE=triplet
SCALE=64.
#WEIGHT=3.
#SCALE=32.
WEIGHT=2.
#WEIGHT=2.5
# ALPHA=0.35
#ALPHA=0.25
ALPHA=0.2
#ALPHA=0.3
#LR_FILE=lr_coco.txt
IMAGE_HEIGHT=112
# IMAGE_WIDTH=112
IMAGE_WIDTH=96
EMBEDDING_SIZE=128
# LR_FILE=lr_coco.txt
LR_FILE=learning_rate_schedule.txt
OPT=ADAM
#OPT=MOM
FC_BN='--fc_bn'
NAME=${NETWORK}_${LOSS_TYPE}_${CROP}_${GPU}_${SCALE}_${WEIGHT}_${ALPHA}_${OPT}_${FC_BN}_${IMAGE_WIDTH}_${EMBEDDING_SIZE}
CMD="python train_multi_gpu.py --logs_base_dir logs/${NAME}/ --models_base_dir models/$NAME/ --data_dir /Users/chenyao/Documents/dataset/CASIA-WebFace/CASIA-WebFace-112X96 --model_def models.inception_resnet_v1  --optimizer ${OPT} --learning_rate -1 --max_nrof_epochs 2 --epoch_size 3 --random_flip --learning_rate_schedule_file ../data/${LR_FILE}  --num_gpus ${NUM_GPUS} --weight_decay ${WEIGHT_DECAY} --weight ${WEIGHT} --alpha ${ALPHA} --embedding_size ${EMBEDDING_SIZE}"
echo Run "$ARGS ${CMD}"
eval "$ARGS ${CMD}"
