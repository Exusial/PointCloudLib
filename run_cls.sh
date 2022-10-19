CUDA_VISIBLE_DEVICES='2' PYTHONPATH=/home/penghy/jittor-dev/jittor/python:/home/penghy/PointCloudLib python train_cls.py \
--model pointnext \
--data_dir /home/penghy/data/ModelNet40Ply2048/ \
--mode test \
--dataset_type ModelNet40-h5 \
--pretrained_path ./pm.pth \
--configs ./pointnext-pre.yaml \
--num_points 1024