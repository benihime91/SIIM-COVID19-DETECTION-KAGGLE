# Instructions for replicating 5 fold retinanet detection model 

1. Download datasets and setup up directory structure

    You need to download these datasets listed below
    - [images](https://www.kaggle.com/benihime91/siim-covid19-png-1024px)
    - [coco annotations](https://www.kaggle.com/benihime91/siim-covid-mmdetection-coco-json)

    Images should be extracted under `data/images` and annotations under `data/annotations`

2. Install requirements
   We need to install the following libraries
   - [mmdetection](https://github.com/open-mmlab/mmdetection)
   - [albumentations](https://github.com/albumentations-team/albumentations)

3. Training
   
    ```bash
    $ git clone https://github.com/open-mmlab/mmdetection.git

    $ CUDA_VISIBLE_DEVICES=0 python mmdetection/tools/train.py retinanet_x101_64x4d_fpn_siim_fold0.py
    $ CUDA_VISIBLE_DEVICES=0 python mmdetection/tools/train.py retinanet_x101_64x4d_fpn_siim_fold1.py
    $ CUDA_VISIBLE_DEVICES=0 python mmdetection/tools/train.py retinanet_x101_64x4d_fpn_siim_fold2.py
    $ CUDA_VISIBLE_DEVICES=0 python mmdetection/tools/train.py retinanet_x101_64x4d_fpn_siim_fold3.py
    $ CUDA_VISIBLE_DEVICES=0 python mmdetection/tools/train.py retinanet_x101_64x4d_fpn_siim_fold4.py
    ```
    
    Logs will be stored under `runs/retinanet_x101_64x4d_fpn_without_empty/fold{}`. Your logs should look something similar to logs given under `example_logs`.


The best checkpoints can then be directly replaced with the retinanet weights in our [inference notebook](https://www.kaggle.com/nischaydnk/604e8587410a-v2m-bin-weighted) to obtain our result.
