git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

CUDA_VISIBLE_DEVICES=0 python tools/train.py retinanet_x101_64x4d_fpn_siim_fold0.py
CUDA_VISIBLE_DEVICES=0 python tools/train.py retinanet_x101_64x4d_fpn_siim_fold1.py
CUDA_VISIBLE_DEVICES=0 python tools/train.py retinanet_x101_64x4d_fpn_siim_fold2.py
CUDA_VISIBLE_DEVICES=0 python tools/train.py retinanet_x101_64x4d_fpn_siim_fold3.py
CUDA_VISIBLE_DEVICES=0 python tools/train.py retinanet_x101_64x4d_fpn_siim_fold4.py
