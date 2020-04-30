# pytorch-deeplab-xception

### Environment Configuration
The code was tested with Anaconda and Python 3.6. 

1. Install dependencies:

    For PyTorch dependency:
    ```Shell
    conda install pytorch=0.4.1 cuda90 -c pytorch 
    pip install torchvision==0.2.2
    ```
    

    For custom dependencies:
    ```Shell
    pip install matplotlib pillow tensorboardX tqdm
    ```


### Dataset  
1.create dirs and download dataset.
   
   For VOC2012:
    ```Shell
    bash download_voc.sh
    ```
    
   For COCO2017:
   
   ```Shell 
    bash download_coco.sh
    ```
    
### Training
Follow steps below to train your model:

1. Input arguments: (see full input arguments via python train.py --help):
    ```Shell
    usage: train.py [-h] [--backbone {resnet,xception,drn,mobilenet}]
                [--out-stride OUT_STRIDE] [--dataset {pascal,coco,cityscapes}]
                [--use-sbd] [--workers N] [--base-size BASE_SIZE]
                [--crop-size CROP_SIZE] [--sync-bn SYNC_BN]
                [--freeze-bn FREEZE_BN] [--loss-type {ce,focal}] [--epochs N]
                [--start_epoch N] [--batch-size N] [--test-batch-size N]
                [--use-balanced-weights] [--lr LR]
                [--lr-scheduler {poly,step,cos}] [--momentum M]
                [--weight-decay M] [--nesterov] [--no-cuda]
                [--gpu-ids GPU_IDS] [--seed S] [--resume RESUME]
                [--checkname CHECKNAME] [--ft] [--eval-interval EVAL_INTERVAL]
                [--no-val]

    ```

2. To train deeplabv3+ using Pascal VOC dataset and ResNet as backbone:
    ```Shell
    bash train_voc.sh
    ```
3. To train deeplabv3+ using COCO dataset and ResNet as backbone:
    ```Shell
    bash train_coco.sh
    ```    
