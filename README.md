# CV_classification
Classification Pipeline in Computer Vision (Pytorch)

# Environments

docker image: `nvcr.io/nvidia/pytorch:22.12-py3`

see details of NVIDIA pytorch docker image in [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-12.html#rel-22-12).

# Directory

```bash
CV_classification
├── datasets
│   ├── __init__.py
│   ├── augmentation.py
│   └── factory.py
├── models
│   ├── __init__.py
│   └── resnet.py
├── log.py
├── main.py
├── train.py
├── run.sh
├── requirements.txt
├── README.md
└── LICENSE
```

# Pipeline

0. Set seed
1. Make directory to save results
2. Build model
3. Build dataset with augmentations
   - Train dataset
   - Validation dataset (optional)
   - Test dataset 
4. Make dataLoader
5. Define optimizer (model parameters)
6. Define loss function
7. Training model
   - Checkpoint model using evaluation on validation dataset
   - Log training history using `logging` or `wandb` in save folder
8. Testing model


# Run

`run.sh`

```bash
dataname=$1
num_classes=$2
opt_list='SGD Adam'
lr_list='0.1 0.01 0.001'
aug_list='default weak strong'
bs_list='16 64 256'

for bs in $bs_list
do
    for opt in $opt_list
    do
        for lr in $lr_list
        do
            for aug in $aug_list
            do
                # use scheduler
                echo "bs: $bs, opt: $opt, lr: $lr, aug: $aug, use_sched: True"
                EXP_NAME="bs_$bs-opt_$opt-lr_$lr-aug_$aug-use_sched"
                
                if [ -d "$EXP_NAME" ]
                then
                    echo "$EXP_NAME is exist"
                else
                    python main.py \
                        --exp-name $EXP_NAME \
                        --dataname $dataname \
                        --num-classes $num_classes \
                        --opt-name $opt \
                        --aug-name $aug \
                        --batch-size $bs \
                        --lr $lr \
                        --use_scheduler \
                        --epochs 50
                fi

                # not use scheduler
                echo "bs: $bs, opt: $opt, lr: $lr, aug: $aug, use_sched: False"
                EXP_NAME="bs_$bs-opt_$opt-lr_$lr-aug_$aug"

                if [ -d "$EXP_NAME" ]
                then
                    echo "$EXP_NAME is exist"
                else
                    python main.py \
                        --exp-name $EXP_NAME \
                        --dataname $dataname \
                        --num-classes $num_classes \
                        --opt-name $opt \
                        --aug-name $aug \
                        --batch-size $bs \
                        --lr $lr \
                        --epochs 50
                fi
            done
        done
    done
done
```


**example**

```bash
bash run.sh CIFAR10 10
```