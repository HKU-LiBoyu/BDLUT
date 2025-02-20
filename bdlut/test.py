import os
import argparse
import time
import os
from tqdm import tqdm

import numpy as np
import torch
import cv2

from data_blur import SRBenchmark #data_noise_a, data_rnoise,data_deblock ,
from utils import PSNR, cal_ssim, _rgb2ycbcr, bPSNR
from PIL import Image
from utils import seed_everything
from models import HKLUT


device = 'cpu'

def parse_args():
    parser = argparse.ArgumentParser("Testing Setting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int,  default=8)
    parser.add_argument("--test-dir", type=str, default='/home/harry/dataset/SR/benchmark/',
                        help="Testing images")

    parser.add_argument("--lut-dir", type=str, default='./luts',
                        help="Directory for storing cached LUTs")
    parser.add_argument("--result-dir", type=str, default='./results',
                        help="Directory to store resulted images")
    parser.add_argument("--upscale", nargs='+', type=int, default=[1, 2],
                        help="upscaling factors")
    parser.add_argument('--msb', type=str, default='hdb', choices=['p', 'hl', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl', 'hdbt'])
    parser.add_argument('--lsb', type=str, default='hdv', choices=['p', 'hl', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl', 'hdbt'])
    parser.add_argument('--act-fn', type=str, default='relu', choices=['relu', 'gelu', 'leakyrelu', 'starrelu'])
    parser.add_argument('--n-filters', type=int, default=64, help="number of filters in intermediate layers")
    args = parser.parse_args()

    factors = 'x'.join([str(s) for s in args.upscale])
    args.exp_name = "msb_{}_lsb_{}_act_{}_nf_{}_{}".format(args.msb, args.lsb, args.act_fn, args.n_filters, factors)
    args.lut_path = f'{args.lut_dir}/{args.exp_name}'

    return args

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    print(args)

    # Prepare directories
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir('results/{}'.format(args.exp_name)):
        os.mkdir('results/{}'.format(args.exp_name))
        print("Results will be saved to: ", 'results/{}'.format(args.exp_name))



    luts = []
    n_stages = len(args.upscale)
    sr_scale = np.prod(args.upscale)

    models = []

    # Load LUTs
    lut_files = os.listdir(args.lut_path)
    print("LUT path: ", args.lut_path)
    for stage in range(n_stages):
        # msb
        msb_weights = []
        for ktype in args.msb:
            weight = torch.tensor(np.load(os.path.join(args.lut_path, f'S{stage}_MSB_{args.msb.upper()}_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy')).astype(np.int_))
            msb_weights.append(weight)

        # lsb
        lsb_weights = []
        for ktype in args.lsb:
            weight = torch.tensor(np.load(os.path.join(args.lut_path, f'S{stage}_LSB_{args.lsb.upper()}_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy')).astype(np.int_))
            lsb_weights.append(weight)

        models.append(HKLUT(msb_weights, lsb_weights, msb=args.msb, lsb=args.lsb, upscale=args.upscale[stage]).to(device))

    # Test datasets
    noise=15
    qf=10
    test_loader = SRBenchmark(args.test_dir, scale=sr_scale)#,noise=noise, qf=qf
    test_datasets = ['GOPRO_LARGE']#'Set5', 'Set14', 'B100', 'Urban100', 'Manga109'

    
    l_accum = [0.,0.,0.]
    dT = 0.
    rT = 0.
    accum_samples = 0

    with torch.no_grad():
        for model in models:
            model.eval()

        for j in range(len(test_datasets)):
            psnrs = []
            ssims = []
            files = test_loader.files[test_datasets[j]]

            best_psnr = 0.0
            best_file=''
            for k in range(len(files)):
                #key = test_datasets[j] + '_' + files[k][:-4]#files[k][:-4]
                key = test_datasets[j] + '_' + files[k][-27:-23]+files[k][-22:-20]+files[k][-19:-17]+files[k][-10:-4]

                img_gt = test_loader.ims[key] # (512, 512, 3) range [0, 255]
                input_im = test_loader.ims[key  + '_lq'] #+  (128, 128, 3) range [0, 255], 'n%d' % noise, , 'n%d' % qf, 'n%d' % sr_scale

                input_im = input_im.astype(np.float32) / 255.0#
                val_L = torch.Tensor(np.expand_dims(np.transpose(input_im, [2, 0, 1]), axis=0)).to(device)  # (1, 3, 128, 128)

                x = val_L
                for model in models:
                    x = model(x)

                # Output 
                image_out = (x).cpu().data.numpy()
                image_out = np.round(image_out*255) # (1, 3, 512, 512)
                image_out = np.transpose(np.clip(image_out[0], 0, 255), [1,2,0]) # BxCxHxW -> HxWxC

                # Save to file
                image_out = image_out.astype(np.uint8)
                #Image.fromarray(image_out).save('results/{}/{}.png'.format(args.exp_name, key))
                cv2.imwrite('results/{}/{}.png'.format(args.exp_name, key), image_out) # for one channel image

                # PSNR and SSIM on Y channel
                y_gt, y_out = _rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(image_out)[:,:,0]
                #y_gt, y_out = img_gt[:, :, 0], image_out[:, :, 0]
                #y_gt, y_out = img_gt, image_out
                psnrs.append(PSNR(y_gt, y_out, sr_scale))#0
                #psnrs.append(bPSNR(y_gt, y_out, 0))
                if PSNR(y_gt, y_out, sr_scale)>best_psnr:#0
                    best_psnr = PSNR(y_gt, y_out, sr_scale)
                    #best_psnr = bPSNR(y_gt, y_out, 0)#sr_scale
                    best_file = key
                #ssims.append(cal_ssim(y_gt, y_out))

            #print('Dataset {} | AVG LUT PSNR: {:.4f} SSIM: {:.4f}'.format(test_datasets[j], np.mean(np.asarray(psnrs)), np.mean(np.asarray(ssims))))
            print('Dataset {} | AVG LUT PSNR: {:.4f}'.format(test_datasets[j], np.mean(np.asarray(psnrs))))
            print('best psnr is ', best_psnr, 'the image is ',best_file)

        
        
"""
Reference:
HKLUT-S: msb:hdb-lsb:hdv-act:gelu-nf:64-2x2:
Dataset Set5 | AVG LUT PSNR: 30.32 SSIM: 0.8587
Dataset Set14 | AVG LUT PSNR: 27.36 SSIM: 0.7475
Dataset B100 | AVG LUT PSNR: 26.73 SSIM: 0.7059
Dataset Urban100 | AVG LUT PSNR: 24.21 SSIM: 0.7102
Dataset Manga109 | AVG LUT PSNR: 27.39 SSIM: 0.8520


HKLUT-L: msb:hdb-lsb:hdb-act:gelu-nf:64-2x1x2:
Dataset Set5 | AVG LUT PSNR: 30.42 SSIM: 0.8606
Dataset Set14 | AVG LUT PSNR: 27.45 SSIM: 0.7497
Dataset B100 | AVG LUT PSNR: 26.79 SSIM: 0.7077
Dataset Urban100 | AVG LUT PSNR: 24.29 SSIM: 0.7131
Dataset Manga109 | AVG LUT PSNR: 27.55 SSIM: 0.8548
"""
