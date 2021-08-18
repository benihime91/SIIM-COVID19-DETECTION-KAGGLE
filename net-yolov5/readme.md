# Instructions for replicating YOLOv5 Models


## Overview
- Train 2 yolov5x models on training data , 5 fold.
- Generate pseudo labels on public test data using WBF ensemble on stage 1 effdet models (d3, d4) and above trained yolov5x models.
- Re-train 2 new models (yolov5x, yolov5l6) on public test (pseudo) + training data (original labels) + random empty images.


## Details
1. Download the data from [here](https://www.kaggle.com/nischaydnk/siim-data-images) and extract it following the structure mentioned in `data/dataset_structure.txt`

2. Install requirements
    ```bash
    $ pip install -r requirements.txt
    ```

3. Stage-1 Training
   
   ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid0-stage1.yaml --hyp data/hyps/params_stage1_model1.yaml \
        --name fold_0 --project runs/exp0 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid1-stage1.yaml --hyp data/hyps/params_stage1_model1.yaml \
        --name fold_1 --project runs/exp0 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid2-stage1.yaml --hyp data/hyps/params_stage1_model1.yaml \
        --name fold_2 --project runs/exp0 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid3-stage1.yaml --hyp data/hyps/params_stage1_model1.yaml \
        --name fold_3 --project runs/exp0 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid4-stage1.yaml --hyp data/hyps/params_stage1_model1.yaml \
        --name fold_4 --project runs/exp0 --epochs 30
    
    ```

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid0-stage1.yaml --hyp data/hyps/params_stage1_model2.yaml \
        --name fold_0 --project runs/exp1 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid1-stage1.yaml --hyp data/hyps/params_stage1_model2.yaml \
        --name fold_1 --project runs/exp1 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid2-stage1.yaml --hyp data/hyps/params_stage1_model2.yaml \
        --name fold_2 --project runs/exp1 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid3-stage1.yaml --hyp data/hyps/params_stage1_model2.yaml \
        --name fold_3 --project runs/exp1 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid4-stage1.yaml --hyp data/hyps/params_stage1_model2.yaml \
        --name fold_4 --project runs/exp1 --epochs 30
    ```

4. Generate Pseudo Labels
    We now take the 2x5 models trained above and generate pseudo labels from it on the Public Test Data. You can refer `psedo_public_test.ipynb` and the notebooks under `net-det` folder to generate the pseudo labels for stage 1.

    > Note: These notebooks were originally trained on KAGGLE environment so please adjust the dataset and checkpoint paths as required.

    The generated pseduo labels and empty images can be downloaded from [here](https://www.kaggle.com/nischaydnk/siim-data-images) and should be extracted under `data/siim_data` .These should be stored under `data/siim_data/pseudo`. 
    
    At this stage we also add a few images without annotations under `data/siim_data/empty`.

5. Train Final Models

    We now train the final models (5 yolov5x and 5 yolov5l6)

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid0-stage2.yaml --hyp data/hyps/params_stage2.yaml \
        --name fold_0 --project runs/exp2 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid1-stage2.yaml --hyp data/hyps/params_stage2.yaml \
        --name fold_1 --project runs/exp2 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid2-stage2.yaml --hyp data/hyps/params_stage2.yaml \
        --name fold_2 --project runs/exp2 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid3-stage2.yaml --hyp data/hyps/params_stage2.yaml \
        --name fold_3 --project runs/exp2 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid4-stage2.yaml --hyp data/hyps/params_stage2.yaml \
        --name fold_4 --project runs/exp2 --epochs 30
    ```


    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/hub/yolov5l6.yaml --weights yolov5l6.pt --data data/siimcovid0-stage2.yaml --hyp data/hyps/params_stage2.yaml \
        --name fold_0 --project runs/exp3 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/hub/yolov5l6.yaml --weights yolov5l6.pt --data data/siimcovid1-stage2.yaml --hyp data/hyps/params_stage2.yaml \
        --name fold_1 --project runs/exp3 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/hub/yolov5l6.yaml --weights yolov5l6.pt --data data/siimcovid2-stage2.yaml --hyp data/hyps/params_stage2.yaml \
        --name fold_2 --project runs/exp3 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/hub/yolov5l6.yaml --weights yolov5l6.pt --data data/siimcovid3-stage2.yaml --hyp data/hyps/params_stage2.yaml \
        --name fold_3 --project runs/exp3 --epochs 30

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/hub/yolov5l6.yaml --weights yolov5l6.pt --data data/siimcovid4-stage2.yaml --hyp data/hyps/params_stage2.yaml \
        --name fold_4 --project runs/exp3 --epochs 30
    ```

6. Take the best checkpoints (foldwise) from the above trained models. These weights can then directly be replaced with the yolov5 weights in our [inference notebook](https://www.kaggle.com/nischaydnk/604e8587410a-v2m-bin-weighted) to obtain the required results.