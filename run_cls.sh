CUDA_VISIBLE_DEVICES='2' python train_cls.py \
--model pointnext \
--data_dir /home/penghy/PointNeXt/data/ModelNet40Ply2048/ \
--mode test \
--dataset_type ModelNet40-h5 \
--pretrained_path /home/penghy/PointNeXt/pm.pth \
--configs /home/penghy/PointNeXt/cfgs/modelnet40ply2048/pointnext-pre.yaml \
--num_points 1024

