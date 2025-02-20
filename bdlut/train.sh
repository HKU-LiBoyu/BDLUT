#!/usr/bin/bash

# # Table 2
# CUDA_VISIBLE_DEVICES=1,2 python train.py \
# --msb hd --lsb hd  \
# --upscale 4 \
# --act-fn relu --n-filters 64 \
# --batch-size 16 


# CUDA_VISIBLE_DEVICES=1,2 python train.py \
# --msb hdb --lsb hd  \
# --upscale 4 \
# --act-fn relu --n-filters 64 \
# --batch-size 16 


# CUDA_VISIBLE_DEVICES=1,2 python train.py \
# --msb hd --lsb hdb  \
# --upscale 4 \
# --act-fn relu --n-filters 64 \
# --batch-size 16 



# # Table 3
# CUDA_VISIBLE_DEVICES=1,2 python train.py \
# --msb hdb --lsb hd  \
# --upscale 1 4 \
# --act-fn relu --n-filters 64 \
# --batch-size 16 


#CUDA_VISIBLE_DEVICES=1,2 python train.py \
#--msb hdb --lsb hd  \
#--upscale 2 2 \
#--act-fn relu --n-filters 64 \
#--batch-size 16 

# HKLUT-L
# CUDA_VISIBLE_DEVICES=0,1,2 python train.py \
# --msb hdb --lsb hd  \
# --upscale 2 1 2 \
# --act-fn relu --n-filters 64 \
# --batch-size 16


# CUDA_VISIBLE_DEVICES=0,1 python train.py \
# --msb hdb --lsb hd  \
# --upscale 1 2 2 \
# --act-fn relu --n-filters 64 \
# --batch-size 16


# CUDA_VISIBLE_DEVICES=0,1,2 python train.py \
# --msb hdb --lsb hd  \
# --upscale 2 2 1 \
# --act-fn relu --n-filters 64 \
# --batch-size 8


# CUDA_VISIBLE_DEVICES=0,1 python train.py \
# --msb hl --lsb hd  \
# --upscale 2 2  \
# --act-fn relu --n-filters 64 \
# --batch-size 16

# CUDA_VISIBLE_DEVICES=2 python train.py \
# --msb hl --lsb p  \
# --upscale 2 2  \
# --act-fn relu --n-filters 64 \
# --batch-size 16



CUDA_VISIBLE_DEVICES=0,1 python train.py \
--msb hdbl --lsb hd  \
--upscale 2 2 \
--act-fn relu --n-filters 64 \
--batch-size 16
