# Instructions for replicating EfficientDet Models


## Overview
- Train efficientdet_d3 and efficientdet_d5 on training data. 
- Generate pseudo labels on public test data using WBF ensemble on stage 1 effdet and yolov5 models.
- Re-train efficientdet_d3 on public test + training data.


## Details

1. Download the images dataset from [here](https://www.kaggle.com/benihime91/siim-covid19-png-1024px). Extrach the images from both train and public test under `data/images/`

2. Install requirements

    ```bash
    $ pip install -r requirements.txt
    ```

3. Stage-1 Training
   
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

3. Generating Pseudo Labels on public test
   
   Taking the WBF of the best checkpoints of the above models and with the stage1 yolov5 models. We generate psudo labels on the Public Test
   dataset. 
   
   To generate the pseudo labels please refer to the notebooks in `nbs/` folder. 

   > Note: These notebooks were originally trained on KAGGLE environment so please adjust the dataset and checkpoint paths as required.

   The generated pseudo labels is already stored in `data/grouped_df_with_pseudo_final.csv`, now tih these annotations we train d3 model again.

4. Train d3 with pseudo labels

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

5. Take the base checkpoints (foldwise) from the before trained d5 model in 3, and the d3 model trained on pseudo labels in 4. These weights can then directly be replaced with the d3 and d5 weights in out [inference notebook](https://www.kaggle.com/nischaydnk/604e8587410a-v2m-bin-weighted) to obtain the required results.