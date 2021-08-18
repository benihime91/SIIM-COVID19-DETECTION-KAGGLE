# Instructions for replicating YOLOv5 Models


## Overview
- Train 2 yolov5x models on training data , 5 fold.
- Generate pseudo labels on public test data using WBF ensemble on stage 1 effdet models (d3, d4) and above trained yolov5x models.
- Re-train 2 new models (yolov5x, yolov5l6) on public test (pseudo) + training data (original labels) + random empty images.


## Details
1. Download [yolov5 repo](https://www.kaggle.com/benihime91/simmyolov5) and install requirements. You should have a directory yolov5 under which all the yolov5 files are stored.

2. Download the arranged images and annotations from [here]()

3. Install requirements

    ```bash
    $ pip install -r requirements.txt
    ```

4. Stage-1 Training
   
   ```bash
   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/00-d3-896.yaml --fold 0 --name d3-stage1
   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/00-d3-896.yaml --fold 1 --name d3-stage1
   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/00-d3-896.yaml --fold 2 --name d3-stage1
   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/00-d3-896.yaml --fold 3 --name d3-stage1
   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/00-d3-896.yaml --fold 4 --name d3-stage1

    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/00-d3-896.yaml \
        --fold 4 --name d4-stage1 --model tf_efficientdet_d4_ap --image_size 1024
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/00-d3-896.yaml \
        --fold 4 --name d4-stage1 --model tf_efficientdet_d4_ap --image_size 1024
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/00-d3-896.yaml \
        --fold 4 --name d4-stage1 --model tf_efficientdet_d4_ap --image_size 1024
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/00-d3-896.yaml \
        --fold 4 --name d4-stage1 --model tf_efficientdet_d4_ap --image_size 1024
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/00-d3-896.yaml \
        --fold 4 --name d4-stage1 --model tf_efficientdet_d4_ap --image_size 1024


   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01-d5-ema-512.yaml --fold 0 --name d5-stage1
   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01-d5-ema-512.yaml --fold 1 --name d5-stage1
   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01-d5-ema-512.yaml --fold 2 --name d5-stage1
   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01-d5-ema-512.yaml --fold 3 --name d5-stage1
   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01-d5-ema-512.yaml --fold 4 --name d5-stage1

   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01a-d5-finetune-512.yaml \
        --fold 0 --name d5-stage2 --initial_checkpoint runs/d5-stage1/fold_0/model_best.pth.tar
   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01a-d5-finetune-512.yaml \
        --fold 1 --name d5-stage2 --initial_checkpoint runs/d5-stage1/fold_1/model_best.pth.tar
   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01a-d5-finetune-512.yaml \
        --fold 2 --name d5-stage2 --initial_checkpoint runs/d5-stage1/fold_2/model_best.pth.tar
   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01a-d5-finetune-512.yaml \
        --fold 3 --name d5-stage2 --initial_checkpoint runs/d5-stage1/fold_3/model_best.pth.tar
   $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01a-d5-finetune-512.yaml \
        --fold 4 --name d5-stage2 --initial_checkpoint runs/d5-stage1/fold_4/model_best.pth.tar
   ```

   Logs and checkpoints will be stored under `runs/{name}/fold_{fold}`

5. Generating Pseudo Labels on public test
   
   Taking the WBF of the best checkpoints of the above models and with the stage1 yolov5 models. We generate psudo labels on the Public Test
   dataset. 
   
   To generate the pseudo labels please refer to the notebooks in `nbs/` folder. 

   > Note: These notebooks were originally trained on KAGGLE environment so please adjust the dataset and checkpoint paths as required.

   The generated pseudo labels is already stored in `data/grouped_df_with_pseudo_final.csv`, now tih these annotations we train d3 model again.

6. Train d3 with pseudo labels

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02-d3-pseudo-ema-896.yaml \
        --fold 0 --name d3-stage2
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02-d3-pseudo-ema-896.yaml \
        --fold 1 --name d3-stage2
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02-d3-pseudo-ema-896.yaml \
        --fold 2 --name d3-stage2
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02-d3-pseudo-ema-896.yaml \
        --fold 3 --name d3-stage2
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02-d3-pseudo-ema-896.yaml \
        --fold 4 --name d3-stage2


    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02a-d3-pseudo-finetune.yaml \
        --fold 0 --name d3-stage2-ft --initial_checkpoint runs/d3-stage2/fold_0/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02a-d3-pseudo-finetune.yaml \
        --fold 1 --name d3-stage2-ft --initial_checkpoint runs/d3-stage2/fold_1/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02a-d3-pseudo-finetune.yaml \
        --fold 2 --name d3-stage2-ft --initial_checkpoint runs/d3-stage2/fold_2/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02a-d3-pseudo-finetune.yaml \
        --fold 3 --name d3-stage2-ft --initial_checkpoint runs/d3-stage2/fold_3/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02a-d3-pseudo-finetune.yaml \
        --fold 4 --name d3-stage2-ft --initial_checkpoint runs/d3-stage2/fold_4/model_best.pth.tar
    ```

7. Take the base checkpoints (foldwise) from the before trained d5 model in 3, and the d3 model trained on pseudo labels in 4. These weights can then directly be replaced with the d3 and d5 weights in out [inference notebook](https://www.kaggle.com/nischaydnk/604e8587410a-v2m-bin-weighted) to obtain the required results.