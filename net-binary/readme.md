## Reproducing trained models for Binary classification(None).

- Generate negative predictions using study (4-class) models of Efficientnet B5 using the same configs as of Efficientnet B6 with image size 512x512 stage 1.
- Use noisy labels generated from above step, train Efficientnet B6 study based model with same config as mentioned in [study repo](https://github.com/benihime91/SIIM-COVID19-DETECTION-KAGGLE/tree/main/net-classification) 
- Use generated oof of Efficientnet b6, and negative class predictions as noisy labels in [training](https://github.com/benihime91/SIIM-COVID19-DETECTION-KAGGLE/blob/main/net-binary/Binary_Train_Noisy.ipynb)

**For competiton data, we have already uploaded the noisy labels which could be directly used in last step [noisy labels](https://github.com/benihime91/SIIM-COVID19-DETECTION-KAGGLE/blob/main/net-binary/df_study_split_binary_negative_eb5ns_eb6eb6_ns_4024.csv)**

***parameter "data_dir" considers that training images and training masks are present in current directory.***
