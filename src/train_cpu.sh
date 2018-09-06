python train_tripletloss.py --logs_base_dir logs/facenet_yyx/ --models_base_dir /mnt/sda/facenet/trai
nedModels/triplet_loss --data_dir /mnt/sda/facenet/dataset/finetune-dataset --image_size 160 --model_def 
models.inception_resnet_v1 --optimizer RMSPROP --learning_rate -1 --weight_decay 1e-4 --max_nrof_epochs 20 --batch_siz
e 90 --epoch_size 1000 --people_per_batch 45 --images_per_person 40 --pretrained_model /mnt/sda/facenet/models/20170512-
110547/model-20170512-110547.ckpt-250000 --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt 
--keep_probability 0.8 --random_crop --random_flip --embedding_size 128 --gpu_memory_fraction 0.9