# Instructions for replicating Study Models (Classification)
## Details
1. Download the images from [here](https://www.kaggle.com/benihime91/siim-covid19-png-1024px) and masks from [here](https://www.kaggle.com/benihime91/siimcovid19masks). Extract it following the structure mentioned in `data/dataset_structure.txt`

2. Install requirements
    ```bash
    $ pip install -r requirements.txt
    ```

3. Training

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/net-classification/configs/00-v2m-PCAM-DANET-512.yaml --fold 0
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/net-classification/configs/00-v2m-PCAM-DANET-512.yaml --fold 1
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/net-classification/configs/00-v2m-PCAM-DANET-512.yaml --fold 2
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/net-classification/configs/00-v2m-PCAM-DANET-512.yaml --fold 3
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/net-classification/configs/00-v2m-PCAM-DANET-512.yaml --fold 4
    ```

    Checkpoints of this model will be saved under `runs/exp0/fold_{fold}`

    You will now have configs and checkpoints under `runs/exp0/fold_{fold}`, using these you need to generate the soft labels of this model on the training dataset and save the csv as `data/noisy_01.csv`. Instructions for the same are given in `generate_soft_labels.ipynb` .

    Now, we train the following models

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01-v2m-PCAM-SAM-bin-study-noisy-512.yaml --fold 0 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01-v2m-PCAM-SAM-bin-study-noisy-512.yaml --fold 1 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01-v2m-PCAM-SAM-bin-study-noisy-512.yaml --fold 2 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01-v2m-PCAM-SAM-bin-study-noisy-512.yaml --fold 3 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/01-v2m-PCAM-SAM-bin-study-noisy-512.yaml --fold 4 --opts NOISY_CSV data/noisy_01.csv
    ```


    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02-v2l-PCAM-SAM-noisy-512.yaml --fold 0 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02-v2l-PCAM-SAM-noisy-512.yaml --fold 1 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02-v2l-PCAM-SAM-noisy-512.yaml --fold 2 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02-v2l-PCAM-SAM-noisy-512.yaml --fold 3 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02-v2l-PCAM-SAM-noisy-512.yaml --fold 4 --opts NOISY_CSV data/noisy_01.csv


    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02a-v2l-PCAM-SAM-noisy-512-finetune.yaml --fold 0 --opts NOISY_CSV data/noisy_01.csv --opts CHECKPOINT runs/exp2/fold_0/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02a-v2l-PCAM-SAM-noisy-512-finetune.yaml --fold 1 --opts NOISY_CSV data/noisy_01.csv --opts CHECKPOINT runs/exp2/fold_1/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02a-v2l-PCAM-SAM-noisy-512-finetune.yaml --fold 2 --opts NOISY_CSV data/noisy_01.csv --opts CHECKPOINT runs/exp2/fold_2/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02a-v2l-PCAM-SAM-noisy-512-finetune.yaml --fold 3 --opts NOISY_CSV data/noisy_01.csv --opts CHECKPOINT runs/exp2/fold_3/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/02a-v2l-PCAM-SAM-noisy-512-finetune.yaml --fold 4 --opts NOISY_CSV data/noisy_01.csv --opts CHECKPOINT runs/exp2/fold_4/model_best.pth.tar
    ```

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/03-v2m-PCAM-SAM-noisy-640.yaml --fold 0 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/03-v2m-PCAM-SAM-noisy-640.yaml --fold 1 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/03-v2m-PCAM-SAM-noisy-640.yaml --fold 2 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/03-v2m-PCAM-SAM-noisy-640.yaml --fold 3 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/03-v2m-PCAM-SAM-noisy-640.yaml --fold 4 --opts NOISY_CSV data/noisy_01.csv
    ```

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04-b5-512.yaml --fold 0
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04-b5-512.yaml --fold 1
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04-b5-512.yaml --fold 2
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04-b5-512.yaml --fold 3
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04-b5-512.yaml --fold 4
    ```

    The above command should save models under `runs/exp4` . Generate soft labels from this model as mentioned above and save it as `data/noisy_02.csv` .

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04a-b5-512-noisy.yaml --fold 0 --opts NOISY_CSV data/noisy_02.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04a-b5-512-noisy.yaml --fold 1 --opts NOISY_CSV data/noisy_02.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04a-b5-512-noisy.yaml --fold 2 --opts NOISY_CSV data/noisy_02.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04a-b5-512-noisy.yaml --fold 3 --opts NOISY_CSV data/noisy_02.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04a-b5-512-noisy.yaml --fold 4 --opts NOISY_CSV data/noisy_02.csv

    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04b-b5-512-noisy-finetune --fold 0 --opts NOISY_CSV data/noisy_02.csv CHECKPOINT runs/exp4a/fold_0/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04b-b5-512-noisy-finetune --fold 1 --opts NOISY_CSV data/noisy_02.csv CHECKPOINT runs/exp4a/fold_1/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04b-b5-512-noisy-finetune --fold 2 --opts NOISY_CSV data/noisy_02.csv CHECKPOINT runs/exp4a/fold_2/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04b-b5-512-noisy-finetune --fold 3 --opts NOISY_CSV data/noisy_02.csv CHECKPOINT runs/exp4a/fold_3/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/04b-b5-512-noisy-finetune --fold 4 --opts NOISY_CSV data/noisy_02.csv CHECKPOINT runs/exp4a/fold_4/model_best.pth.tar
    ```

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05-b6-640.yaml --fold 0
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05-b6-640.yaml --fold 1
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05-b6-640.yaml --fold 2
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05-b6-640.yaml --fold 3
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05-b6-640.yaml --fold 4
    ```

    Generate soft labels for the above model and store it under `data/noisy_03.csv`

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05a-b7-640-noisy.yaml --fold 0 --opts NOISY_CSV data/noisy_03.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05a-b7-640-noisy.yaml --fold 1 --opts NOISY_CSV data/noisy_03.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05a-b7-640-noisy.yaml --fold 2 --opts NOISY_CSV data/noisy_03.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05a-b7-640-noisy.yaml --fold 3 --opts NOISY_CSV data/noisy_03.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05a-b7-640-noisy.yaml --fold 4 --opts NOISY_CSV data/noisy_03.csv

    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05b-b7-640-noisy-finetune.yaml --fold 0 --opts NOISY_CSV data/noisy_03.csv CHECKPOINT runs/exp5a/fold_0/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05b-b7-640-noisy-finetune.yaml --fold 1 --opts NOISY_CSV data/noisy_03.csv CHECKPOINT runs/exp5a/fold_1/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05b-b7-640-noisy-finetune.yaml --fold 2 --opts NOISY_CSV data/noisy_03.csv CHECKPOINT runs/exp5a/fold_2/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05b-b7-640-noisy-finetune.yaml --fold 3 --opts NOISY_CSV data/noisy_03.csv CHECKPOINT runs/exp5a/fold_3/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/05b-b7-640-noisy-finetune.yaml --fold 4 --opts NOISY_CSV data/noisy_03.csv CHECKPOINT runs/exp5a/fold_4/model_best.pth.tar
    ```

    Finally train the following Models

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/06-v2l-PCAM-SAM-noisy-640.yaml --fold 0 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/06-v2l-PCAM-SAM-noisy-640.yaml --fold 1 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/06-v2l-PCAM-SAM-noisy-640.yaml --fold 2 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/06-v2l-PCAM-SAM-noisy-640.yaml --fold 3 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/06-v2l-PCAM-SAM-noisy-640.yaml --fold 4 --opts NOISY_CSV data/noisy_01.csv
    ```
    
    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/07-v2l-PCAM-SAM-noisy-512.yaml --fold 0 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/07-v2l-PCAM-SAM-noisy-512.yaml --fold 1 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/07-v2l-PCAM-SAM-noisy-512.yaml --fold 2 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/07-v2l-PCAM-SAM-noisy-512.yaml --fold 3 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/07-v2l-PCAM-SAM-noisy-512.yaml --fold 4 --opts NOISY_CSV data/noisy_01.csv
    ```

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/08-v2l-PCAM-SAM-noisy-512.yaml --fold 0 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/08-v2l-PCAM-SAM-noisy-512.yaml --fold 1 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/08-v2l-PCAM-SAM-noisy-512.yaml --fold 2 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/08-v2l-PCAM-SAM-noisy-512.yaml --fold 3 --opts NOISY_CSV data/noisy_01.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/08-v2l-PCAM-SAM-noisy-512.yaml --fold 4 --opts NOISY_CSV data/noisy_01.csv


    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/08a-v2l-PCAM-SAM-noisy-finetune.yaml --fold 0 --opts NOISY_CSV data/noisy_01.csv CHECKPOINT runs/exp8/fold_0/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/08a-v2l-PCAM-SAM-noisy-finetune.yaml --fold 1 --opts NOISY_CSV data/noisy_01.csv CHECKPOINT runs/exp8/fold_1/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/08a-v2l-PCAM-SAM-noisy-finetune.yaml --fold 2 --opts NOISY_CSV data/noisy_01.csv CHECKPOINT runs/exp8/fold_2/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/08a-v2l-PCAM-SAM-noisy-finetune.yaml --fold 3 --opts NOISY_CSV data/noisy_01.csv CHECKPOINT runs/exp8/fold_3/model_best.pth.tar
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/08a-v2l-PCAM-SAM-noisy-finetune.yaml --fold 4 --opts NOISY_CSV data/noisy_01.csv CHECKPOINT runs/exp8/fold_4/model_best.pth.tar

    ```

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/09-v2m-PCAM-SAM-1024.yaml --fold 0
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/09-v2m-PCAM-SAM-1024.yaml --fold 1
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/09-v2m-PCAM-SAM-1024.yaml --fold 2
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/09-v2m-PCAM-SAM-1024.yaml --fold 3
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/09-v2m-PCAM-SAM-1024.yaml --fold 4
    ```

    Generate soft labels for the above model and save it under `data/noisy_4.csv`.

    ```bash
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/09a-v2m-PCAM-SAM-1024-noisy.yaml --fold 0 --opts NOISY_CSV data/noisy_4.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/09a-v2m-PCAM-SAM-1024-noisy.yaml --fold 1 --opts NOISY_CSV data/noisy_4.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/09a-v2m-PCAM-SAM-1024-noisy.yaml --fold 2 --opts NOISY_CSV data/noisy_4.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/09a-v2m-PCAM-SAM-1024-noisy.yaml --fold 3 --opts NOISY_CSV data/noisy_4.csv
    $ CUDA_VISIBLE_DEVICES=0 python train.py --config configs/09a-v2m-PCAM-SAM-1024-noisy.yaml --fold 4 --opts NOISY_CSV data/noisy_4.csv
    ```


4. In our final ensemble we take the following models

*v2m models*
- `runs/exp1/fold_0/model_best.pth.tar`
- `runs/exp1/fold_1/model_best.pth.tar`
- `runs/exp1/fold_2/model_best.pth.tar`
- `runs/exp1/fold_3/model_best.pth.tar`
- `runs/exp1/fold_4/model_best.pth.tar`

*other models*
* `runs/exp2a/fold_1/model_best.pth.tar`
* `runs/exp2a/fold_3/model_best.pth.tar`
* `runs/exp2a/fold_4/model_best.pth.tar`
* `runs/exp3/fold_0/model_best.pth.tar`
* `runs/exp4a/fold_2/model_best.pth.tar`
* `runs/exp5b/fold_2/model_best.pth.tar`
* `runs/exp5b/fold_3/model_best.pth.tar`
* `runs/exp6/fold_2/model_best.pth.tar`
* `runs/exp7/fold_3/model_best.pth.tar`
* `runs/exp8a/fold_4/model_best.pth.tar`
* `runs/exp9a/fold_0/model_best.pth.tar`


The inference code can be found [here](https://www.kaggle.com/nischaydnk/604e8587410a-v2m-bin-weighted).