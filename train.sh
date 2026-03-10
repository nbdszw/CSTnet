# Dataset related
dataset=Houston # change this to the dataset you want to train on
data_dir=./Dataset/$dataset # change this to the path of the dataset you want to train on
num_bands=48 # change this to the number of bands in the dataset you want to train on

# Hyperparameters related
lr=0.01 # change this to the learning rate you want to use

# Loss weight related
transfer_loss_weight=1 # change this to the transfer loss weight you want to use
dis_loss_weight=1 # change this to the discriminator loss weight you want to use
semantic_loss_weight=1 # semantic branch total weight

# Semantic branch related (Chapter 4)
use_semantic_branch=False
semantic_path=./semantic_priors/${dataset}/semantic_bank_combined.npy
semantic_conf_threshold=0.9
semantic_src_weight=1.0
semantic_tgt_weight=1.0

# Others
seed=1234 # change this to the seed you want to use
patch_size=5 # change this to the patch size you want to use

# Output related
output=./log/${dataset}_${seed}_${lr}_${transfer_loss_weight}_${dis_loss_weight}_${semantic_loss_weight}_${patch_size}

python main.py --config param.yaml --data_dir $data_dir --num_bands $num_bands --seed $seed --lr $lr --dis_loss_weight $dis_loss_weight --transfer_loss_weight $transfer_loss_weight --patch_size ${patch_size} --use_semantic_branch $use_semantic_branch --semantic_path $semantic_path --semantic_conf_threshold $semantic_conf_threshold --semantic_loss_weight $semantic_loss_weight --semantic_src_weight $semantic_src_weight --semantic_tgt_weight $semantic_tgt_weight | tee ${output}.log
