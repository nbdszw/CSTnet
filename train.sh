# Dataset related
dataset=Houston 
data_dir=./Dataset/$dataset 


# Hyperparameters related
lr=0.01 

# Loss weight related
transfer_loss_weight=1 
dis_loss_weight=1 

# Others
seed=3407
patch_size=5 

# Output related
output=./log/${dataset}_${seed}_${lr}_${transfer_loss_weight}_${dis_loss_weight}_${patch_size}

python main.py --config param.yaml --data_dir $data_dir --seed $seed --lr $lr --dis_loss_weight $dis_loss_weight --transfer_loss_weight $transfer_loss_weight --patch_size ${patch_size}| tee ${output}.log