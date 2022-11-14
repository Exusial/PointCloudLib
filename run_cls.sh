CUDA_VISIBLE_DEVICES='2' PYTHONPATH=/home/penghy/jittor-dev/jittor/python:/home/penghy/PointCloudLib python train_cls.py \
--model pointnext \
--data_dir /home/penghy/PointNeXt/data/ModelNet40Ply2048/ \
--mode train \
--dataset_type ModelNet40 \
--configs /home/penghy/PointCloudLib/configs/pointnext/modelnet40/pointnext-pre.yaml \
--num_points 1024 \
--lr 0.001 \
--optimizer adam \
--batch_size 32 \
--weight_decay 0.05
# --pretrained_path /home/penghy/PointNeXt/pm.pth
# --pretrained_path /home/penghy/PointNeXt/init_check.pth