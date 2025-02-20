#!/usr/bin/bash
python3 train.py \
--msb hdb --lsb hd  \
--upscale 2 1 2 \
--act-fn relu --n-filters 64 \
--batch-size 32
