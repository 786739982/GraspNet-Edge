CUDA_VISIBLE_DEVICES=1 python train.py --camera realsense --log_dir logs/log_rs --batch_size 2 --dataset_root /data/graspnet
# CUDA_VISIBLE_DEVICES=0 python train.py --camera kinect --log_dir logs/log_kn --batch_size 2 --dataset_root /data/Benchmark/graspnet

#torchrun  --nproc_per_node=2 ddp_test.py --batchSize 64 --epochs 10
# CUDA_VISIBLE_DEVICES=1 python train.py --camera realsense --log_dir logs/log_rs_x5 --batch_size 2 --dataset_root /data/graspnet --checkpoint_path logs/log_dist_lr0.005_bs16/checkpoint_dist.tar --max_epoch 19