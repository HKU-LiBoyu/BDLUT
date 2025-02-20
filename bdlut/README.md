# super-resolution-lut
Repository for ICCV23 submission #12009: Hundred-Kilobyte Lookup Tables for Efficient Single-Image Super-Resolution

## Usage
### Dependency
```
python=3.8.16
torch=1.12.1
tensorboard=2.12.0
scipy=1.10.1
tqdm=4.65.0
opencv-python=4.7.0.72
```
### Dataset
##### Training set
```
data/train/DIV2K/
            /HR/*.png
            /LR/{X2, X3, X4, X8}/*.png
```
##### Testing set
```
data/test/{Set5, Set14, B100, Urban100, Manga109}/
            /HR/*.png
            /LR_bicubic/{X2, X3, X4, X8}/*.png
```

### Training
```
./scripts/train.sh
```

### Tranfer to LUTs
```
./scripts/transfer.sh
```

### Testing
```
./scripts/test.sh
```

## License
MIT


## Acknowledgement
Our code is based upon [MuLUT](https://github.com/ddlee-cn/MuLUT) and [SPLUT](https://github.com/zhjy2016/SPLUT)
