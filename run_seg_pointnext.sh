PYTHONPATH=/home/penghy/jittor-dev/jittor/python:/home/penghy/PointCloudLib python train_partseg.py \
--model pointnext \
--data_dir /home/penghy/PointNeXt/data/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0_normal \
--mode train \
--configs /home/penghy/PointNeXt/cfgs/shapenetpart/pointnext-s.yaml \
--num_points 2048 \
--lr 0.001 \
--optimizer adam \
--batch_size 8 \
--weight_decay 0.05 \
--model pointnext \
--dataset_type "ShapenetPart-txt" \
--train_transform "PointCloudCenterAndNormalize" \
--val_transform "PointCloudCenterAndNormalize"
# --num_votes 10 \
# --pretrained_path /home/penghy/PointNeXt/shape.pth
# --pretrained_path /home/penghy/PointNeXt/init_check.pth

