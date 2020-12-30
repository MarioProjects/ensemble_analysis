#!/bin/bash

# Only download the data argument ./tests/classification/cifar10.sh only_data

# Check if CIFAR10 data is available, if not download
if [ ! -d "data/CIFAR10" ]
then
    echo "CIFAR10 data not found at 'data' directory. Downloading..."
    wget -nv --load-cookies /tmp/cookies.txt \
      "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt \
      --keep-session-cookies --no-check-certificate \
      'https://docs.google.com/uc?export=download&id=18AEpsx5-_LzkrV-H9IJzM72sfbo3yzEp' \
      -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18AEpsx5-_LzkrV-H9IJzM72sfbo3yzEp" \
      -O cifar10.tar.gz && rm -rf /tmp/cookies.txt
    mkdir -p data/CIFAR10
    tar -zxf cifar10.tar.gz  -C data/CIFAR10/
    rm cifar10.tar.gz
    echo "Done!"
else
  echo "CIFAR10 data found at 'data' directory!"
fi

if [[ $1 == "only_data" ]]
then
  exit
fi

seed=301220201
gpu="0"
dataset="CIFAR10"
problem_type="classification"

# Available models:
#   -> kuangliu_resnet[18-34-50-101-152] - osmr_resnet18 - osmr_resnet18_pretrained
model="kuangliu_resnet18"

img_size=32
crop_size=32
batch_size=128

epochs=350
swa_start=999
defrost_epoch=-1
scheduler="steps"
lr=0.1
swa_lr=0.256
# Available schedulers:
# constant - steps - plateau - one_cycle_lr (max_lr) - cyclic (min_lr, max_lr, scheduler_steps)
# Available optimizers:
# adam - sgd - over9000
optimizer="sgd"

# Available data augmentation policies:
# "none" - "random_crops" - "rotations" - "vflips" - "hflips" - "elastic_transform" - "grid_distortion" - "shift"
# "scale" - "optical_distortion" - "coarse_dropout" or "cutout" - "downscale"
data_augmentation="cifar10"
normalization="statistics"  # reescale - standardize - statistics

# Available criterions for classification:
# ce
criterion="ce"
weights_criterion="1.0"

output_dir="results/$dataset/$model/seed_$seed/$optimizer/${scheduler}_lr${lr}/${criterion}_weights${weights_criterion}"
output_dir="$output_dir/normalization_${normalization}/da${data_augmentation}"

python3 -u train.py --gpu $gpu --dataset $dataset --model_name $model --img_size $img_size --crop_size $crop_size \
--epochs $epochs --swa_start $swa_start --batch_size $batch_size --defrost_epoch $defrost_epoch \
--scheduler $scheduler --learning_rate $lr --swa_lr $swa_lr --optimizer $optimizer --criterion $criterion \
--normalization $normalization --weights_criterion "$weights_criterion" --data_augmentation $data_augmentation \
--output_dir "$output_dir" --metrics accuracy --problem_type $problem_type \
--scheduler_steps 150 250 --seed $seed

model_checkpoint="$output_dir/model_${model}_best_accuracy.pt"
python3 -u evaluate.py --gpu $gpu --dataset $dataset --model_name $model --img_size $img_size --crop_size $crop_size \
--batch_size $batch_size --normalization $normalization --output_dir "$output_dir" \
--metrics accuracy --problem_type $problem_type --model_checkpoint "$model_checkpoint" --seed $seed

#model_checkpoint="$output_dir/model_${model}_${epochs-swa_start}epochs_swalr${swa_lr}.pt"
#python3 -u evaluate.py --gpu $gpu --dataset $dataset --model_name $model --img_size $img_size --crop_size $crop_size \
#--swa_checkpoint --batch_size $batch_size --normalization $normalization --output_dir "$output_dir" \
#--metrics accuracy --problem_type $problem_type --model_checkpoint "$model_checkpoint" --seed $seed
