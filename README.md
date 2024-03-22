# Positive Label Is All You Need for Multi-Label Classification

Official PyTorch Implementation

> Zhixiang Yuan, Kaixin Zhang, Tao Huang </br>
> Anhui University of Technology, The University of Sydney

**Abstract**
Multi-label classification (MLC) faces challenges from label noise in training data due to annotating diverse semantic labels for each image. Current methods mainly target identifying and correcting label mistakes using trained MLC models, but still struggle with persistent noisy labels during training, resulting in imprecise recognition and reduced performance. Our paper addresses label noise in MLC by introducing a positive and unlabeled multi-label classification (PU-MLC) method. To counteract noisy labels, we directly discard negative labels, focusing on the abundance of negative labels and the origin of most noisy labels. PU-MLC employs positive-unlabeled learning, training the model with only positive labels and unlabeled data. The method incorporates adaptive re-balance factors and temperature coefficients in the loss function to address label distribution imbalance and prevent over-smoothing of probabilities during training. Additionally, we introduce a local-global convolution module to capture both local and global dependencies in the image without requiring backbone retraining. PU-MLC proves effective on MLC and MLC with partial labels (MLC-PL) tasks, demonstrating significant improvements on MS-COCO and PASCAL VOC datasets with fewer annotations.


## Training Code
We provide [Training code](train.py), that demonstrate how to train our model. Example of training with MS-COCO (We train the model on 2 ):
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py <dataset_path> --dataset=COCO2014 --model-name=resnet101 --pretrainedModel=<pretrained_model_path> --workers=8 --scaleSize=512 --cropSize=448 --topK=1 --b=35 --num-classes=80 --lr=1e-4 --ema=0.9997 --Stop_epoch=40 --alpha=1.0 --prob=0.1 --gamma=0.55
```

Example of training with PASCAL VOC: 
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py <dataset_path> --dataset=VOC2007 --model-name=resnet101 --pretrainedModel=<pretrained_model_path> --workers=8 --scaleSize=512 --cropSize=448 --topK=1 --b=45 --num-classes=20 --lr=4e-5 --Stop_epoch=55 --ema=0.991 --alpha=2.0 --prob=0.1 --gamma=0.85
```

## Citation
```
 @misc{PU-MLC2023, 
        title={Positive Label Is All You Need for Multi-Label Classification}, 
        author={Zhixiang Yuan and Kaixin Zhang and Tao Huang}, 
        year={2023}, 
        eprint={2306.16016},
        archivePrefix={arXiv}, 
        primaryClass={cs.CV}}
```

## Acknowledgements
Some components of this code implementation are adapted from the repository https://github.com/Alibaba-MIIL/ASL.

## Contact
Feel free to contact if there are any questions or issues - Kaixin Zhang (kxzhang0618@163.com).
