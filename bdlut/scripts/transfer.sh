#!/usr/bin/bash
python3 transfer.py \
--msb hdbv --lsb hd  \
--upscale 4 \
--act-fn relu --n-filters 64
