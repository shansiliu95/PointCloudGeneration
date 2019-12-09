# Point Clouds Generation via Score Matching

The codes are heavily borrowed from https://github.com/yangyanli/PointCNN. Please refer to this link for how to prepare the ShapeNet segmentation dataset.

I add 723 lines and remove 247 lines of codes. Modifications are mainly made in langevin_dynamics.py, train_val_generation.py and pointcnn.py.

Train:

python ./train_val_generation.py -t ../data/shapenet_partseg/train_val_files.txt -v ../data/shapenet_partseg/test_files.txt -s ../models/generation/ -m pointcnn_seg -x shapenet_generation

Sample:

python ./langevin_dynamics.py --sample_size=2048 -m pointcnn_seg -x shapenet_generation  -l /pine/scr/s/s/ssy95/models/generation/pointcnn_seg_shapenet_generation_2019-12-08-03-11-41_52349/ckpts  --grid_size=1
