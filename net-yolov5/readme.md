# Instructions for replicating YOLOv5 Models


## Overview
- Train 2 yolov5x models on training data , 5 fold.
- Generate pseudo labels on public test data using WBF ensemble on stage 1 effdet models (d3, d4) and above trained yolov5x models.
- Re-train 2 new models (yolov5x, yolov5l6) on public test (pseudo) + training data (original labels) + random empty images.


## Details
1. Download the data from [here]() and extrach it following the structure mentioned in `data/dataset_structure.txt`

2. Install requirements

    ```bash
    $ pip install -r requirements.txt
    ```

3. Stage-1 Training
   
   ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid0-stage1.yaml --hyp data/hyps/params_stage1_model1.yaml \
        --name fold_0 --project runs/exp0

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid1-stage1.yaml --hyp data/hyps/params_stage1_model1.yaml \
        --name fold_1 --project runs/exp0

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid2-stage1.yaml --hyp data/hyps/params_stage1_model1.yaml \
        --name fold_2 --project runs/exp0

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid3-stage1.yaml --hyp data/hyps/params_stage1_model1.yaml \
        --name fold_3 --project runs/exp0

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid4-stage1.yaml --hyp data/hyps/params_stage1_model1.yaml \
        --name fold_4 --project runs/exp0
    
    ```

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid0-stage1.yaml --hyp data/hyps/params_stage1_model2.yaml \
        --name fold_0 --project runs/exp1

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid1-stage1.yaml --hyp data/hyps/params_stage1_model2.yaml \
        --name fold_1 --project runs/exp1

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid2-stage1.yaml --hyp data/hyps/params_stage1_model2.yaml \
        --name fold_2 --project runs/exp1

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid3-stage1.yaml --hyp data/hyps/params_stage1_model2.yaml \
        --name fold_3 --project runs/exp1

    $ CUDA_VISIBLE_DEVICES=0 python train.py \
        --cfg models/yolov5x.yaml --weights yolov5x.pt --data data/siimcovid4-stage1.yaml --hyp data/hyps/params_stage1_model2.yaml \
        --name fold_4 --project runs/exp1
    ```

4. Take the best checkpoints (foldwise) from the above trained models. These weights can then directly be replaced with the yolov5 weights in our [inference notebook](https://www.kaggle.com/nischaydnk/604e8587410a-v2m-bin-weighted) to obtain the required results.