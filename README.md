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
