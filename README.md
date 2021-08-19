5th place solution for SIIM-FISABIO-RSNA COVID-19 Detection Challenge

This documentation outlines how to reproduce the 5th place solution by the "Ayushman Nischay Shivam" team for the covid19 detection competition on Kaggle hosted by SIIM-FISABIO-RSNA.

## Hardware

All of our single models were trained using multi v100(local) or single v100(google colab pro) instances with GPU enabled to run all data preprocessing, model training, and inference was done with kaggle noteboks.

[https://www.kaggle.com/docs/notebooks](https://www.kaggle.com/docs/notebooks)

## Software

We used [Kaggle GPU notebooks](https://github.com/Kaggle/docker-python/blob/master/gpu.Dockerfile) to run all our inference scripts.

Below are the packages used in addition to the ones included in the default train scripts provided. All packages were installed via uploaded kaggle dataset.

| Package Name | Repository | Kaggle Dataset |
| --- |--- | --- |
| timm- pytorch image models | https://github.com/rwightman/pytorch-image-models | https://www.kaggle.com/kozodoi/timm-pytorch-image-models |
| pytorch-ranger=0.1.1 |https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer| https://www.kaggle.com/bessenyeiszilrd/rangerdeeplearningoptimizer |
| efficientdet-pytorch=0.2.3 | https://github.com/rwightman/efficientdet-pytorch | https://www.kaggle.com/sreevishnudamodaran/effdet-latestvinbigdata-wbf-fused |
| ensemble-boxes=1.0.4| https://github.com/ZFTurbo/Weighted-Boxes-Fusion | https://www.kaggle.com/vgarshin/ensemble-boxes-104 |
| MMdetection=2.1.4 | https://github.com/open-mmlab/mmdetection | https://www.kaggle.com/sreevishnudamodaran/mmdetectionv2140|
| yolov5 |https://github.com/ultralytics/yolov5 | https://www.kaggle.com/benihime91/simmyolov5 |
| Mish-cuda | https://www.kaggle.com/benihime91/mishcuda |https://www.kaggle.com/benihime91/mishcuda |
| pytorch-toolbelt | https://github.com/BloodAxe/pytorch-toolbelt | https://www.kaggle.com/bloodaxe/pytorch-toolbelt |
| radam-pytorch | https://github.com/LiyuanLucasLiu/RAdam | https://www.kaggle.com/lextoumbourou/radampytorch |





## Data Used

Please add https://www.kaggle.com/c/siim-covid19-detection/data as the input dataset.





# Models Summary

   ## Study Level Models 
   
   
All of our study models were trained with variants of Efficientnet models. Other architectures like Resnet, Densenet, transformer-based models didn’t perform well for us. Our models were pretrained on imagenet and didn’t use any external x-ray data directly / indirectly during the competition. Some models were trained on multiple stages which includes finetuning with a reduced learning rate or increasing the image size.

Baseline Architectures used in our final study level solution:

   - Efficientnet v2m
   - Efficientnet v2l
   - Efficientnet B5
   - Efficientnet B7

   
<img width="682" alt="Screenshot 2021-08-19 at 5 50 19 PM" src="https://user-images.githubusercontent.com/59060430/130067334-84ace84e-4cb2-48e2-9134-f7751bd6b72a.png">

## Efficientnet v2m: 

- Pretrained imagenet weights were used from timm models. 
- 512 x 512(5- fold) & 640x640(fold 0) & 1024x1024(fold 0) image size.
- PCAM pooling + SAM attention map used in stage2
- Loss fnc: BCE{4-class} + [0.75* lovasz_loss + 0.25* BCE ]{Segmentation loss} 
- Noisy labels were generated with PCAM pooling + DANET attention map in stage 1.
- Ranger optimizer, Cosine Scheduler with warmup were used.


 ## Efficientnet v2l:
- Pretrained imagenet weights were used from timm models. 
- 512 x 512(5 folds) & 640x640 (fold 1) image size
- PCAM pooling + SAM attention map used in training and finetune stage.
- Loss fnc: BCE{4-class} + [0.75* lovasz_loss + 0.25* BCE ]{Segmentation loss} 
- Noisy labels were introduced using the out of folds predictions of the Efficientnet v2m model mentioned above.
- Ranger optimizer, Cosine Scheduler with warmup were used.


## Efficientnet B5: 
- Pretrained imagenet weights were used from timm models. 
- 640 x 640 image size
- average pooling + sCSE attention map along with multi-head attention was used in the training and finetune stage.
- Loss fnc: BCE{4-class} + [0.75* lovasz_loss + 0.25* BCE ]{Segmentation loss} 
- Noisy labels were introduced using the out of folds predictions of the efficientnet B5 model with similar configs.
- Ranger optimizer, Cosine Scheduler with warmup were used.


 ## Efficientnet B7:
- Pretrained imagenet weights were used from timm models. 
- 640 x 640 image size
- average pooling + sCSE attention map along with multi-head attention was used in the training and finetune stage.
- Loss fnc: BCE{4-class} + [0.75* lovasz_loss + 0.25* BCE ]{Segmentation loss} 
- Noisy labels were introduced using the out of folds predictions of the efficientnet B6(640 image size) model with similar configs as of the current model.
- Ranger optimizer, Cosine Scheduler with warmup were used.


**As described above for models individually, some of the strategies which were quite common in our models and gave us a good amount of boost were:**

  1. Noisy Student training
  2. Horizontal Flip Test Time Augmentation
  3. Attention Head
  4. Auxilliary Loss using Segmentation Masks
  5. Fine-tuning


## Image Level Solution { Binary Classification }:

Very Similar to study level models, our binary model was trained with Efficientnet B6. Again our model was trained with imagenet weights without any pretraining on **external data**. 
For None predictions, we noticed that if duplicates are ignored, all none predictions were the same as the “Negative For Pneumonia” Class which was in study predictions.

So, our final binary predictions were a weighted average of Efficientnet binary predictions and study-based Efficientnet- v2m(5 fold) **Negative for Pneumonia** predictions whose training was explained above in the study level solution.

<img width="654" alt="Screenshot 2021-08-19 at 6 06 58 PM" src="https://user-images.githubusercontent.com/59060430/130069409-348ecbe5-3959-4100-a828-b4373ae3e336.png">

## Efficientnet B6:

 - Pretrained imagenet weights were used from timm models. 
 - 640 x 640 image size
 - average pooling was used in the training and finetune stage.
 - Loss fnc: BCE{Binary} + [0.5* lovasz_loss + 0.5* BCE ]{Segmentation loss} 
 - Noisy labels were introduced using the out of folds predictions of the efficientnet B6(640 image size) model with the same configs.
 - Ranger optimizer, Cosine Scheduler with warmup were used.


## Classification models based Ensemble{None + multiclass}:

<img width="658" alt="Screenshot 2021-08-19 at 6 08 45 PM" src="https://user-images.githubusercontent.com/59060430/130069640-d84e6b9d-7a34-41ab-b86c-f144339a9d0e.png">


**Study Level:** We simply took the mean of 11 models predictions for each class based on their fold-wise results and ensemble boost. Further, they were blended with efficientnet v2m(5 fold) with weights 0.85 - 0.15.

**Image Level:** For none predictions, as mentioned before we took the weighted average of Efficientnet B6( trained on Binary Classification) and Efficientnet v2m (same as study model). 

Weights for all the ensembles mentioned above were solely determined by best validation score and diversity.


## Image Level Solution{ Object Detection }:


<img width="682" alt="Screenshot 2021-08-19 at 6 10 27 PM" src="https://user-images.githubusercontent.com/59060430/130069866-06f76da5-3653-4d24-b594-a0334bd4a473.png">

For the object detection part, our final solution used five models(5 fold each), all having different baseline architecture.


***Summary of each object detection model:***

**Efficientnet - D5:** It was trained on just training data. The image size used was 512 x 512. It was trained in two stages, in the second stage it was finetuned with a lower learning rate. The exponential moving average(EMA) was also used in this model’s training.


**Efficientnet - D3:** It was trained on the training data + public test data pseudo labels generated from an ensemble of decent scoring object detection models. Image size used was the default for efficientdet D3 which is 896 x 896. It was also trained in two stages as efficientdet D5. The exponential moving average(EMA) was also used in this model’s training.


**Yolo - v5l6:** It was trained on the training data + public test data pseudo labels. The image size used was 640 x 640 for training. Some images without Bounding Boxes were also included in training data( 0.2% ). 


**Yolo - v5x:** It was trained on the training data + public test data pseudo labels. Image size used was the default for Yolo-v5x which is 640 x 640. Some images without Bounding Boxes were also included in training data( 0.2% ).


**RetinaNet:** The backbone used for retinanet was resnext101_64x4d. Image used was 
(1333,800). Pseudo labels weren’t used, only training data with bounding boxes were used in the training part.



## Post Processing

<img width="653" alt="Screenshot 2021-08-19 at 6 12 53 PM" src="https://user-images.githubusercontent.com/59060430/130070217-a7508448-accb-4aea-91be-5298cc1f4071.png">


Post-processing gave us a great amount of boost in both oof score as well as a leaderboard ( + 0.004).
As we didn’t use any end-to-end solution and trained models for each level separately,  
the idea behind using post-processing was to somehow consume both binary predictions and detection model predictions in a form of ensemble. 
Due to the high amount of diversity in both types of predictions, we were able to achieve that boost. 
We tried several types of merge both predictions which include mean, weighted average, geometric mean, etc. Out of which power ensemble outperformed everyone in both public leaderboard and validation score. Therefore, we planned to stick with power ensemble in final submissions.  
