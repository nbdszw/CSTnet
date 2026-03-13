# Dataset related
dataset=Houston # change this to the dataset you want to train on
data_dir=./Dataset/$dataset # change this to the path of the dataset you want to train on
num_bands=48 # change this to the number of bands in the dataset you want to train on

# Hyperparameters related
lr=0.01 # change this to the learning rate you want to use

# Loss weight related
transfer_loss_weight=1 # change this to the transfer loss weight you want to use
dis_loss_weight=1 # change this to the discriminator loss weight you want to use

# Others
seed=1234 # change this to the seed you want to use
patch_size=5 # change this to the patch size you want to use

# Output related
output=./log/${dataset}_${seed}_${lr}_${transfer_loss_weight}_${dis_loss_weight}_${patch_size}

python main.py --config param.yaml --data_dir $data_dir --num_bands $num_bands --seed $seed --lr $lr --dis_loss_weight $dis_loss_weight --transfer_loss_weight $transfer_loss_weight --patch_size ${patch_size}| tee ${output}.log