import sys
import os
model_path = [
    "/scratch/zzh136/xrd_2d/re_exp/zone1/ckpt/model_zone1_pretrained_vgg19_hkl_noflip_7.pth",
    "/scratch/zzh136/xrd_2d/re_exp/zone4/ckpt/model_zone4_pretrained_vgg19_hkl_noflip_7.pth",
    # "/scratch/zzh136/xrd_2d/re_exp/zone10/ckpt/model_zone10_pretrained_vgg19_hkl_noflip_7.pth",
    # "/scratch/zzh136/xrd_2d/axis20_data/ckpt/pretrained_vgg19_hkl_noflip_7.pth",
]


dataset_path = [
    # "/scratch/zzh136/xrd_2d/re_exp/zone1",
    "/scratch/zzh136/xrd_2d/re_exp/test/zone001",
    "/scratch/zzh136/xrd_2d/re_exp/test/zone010",
    "/scratch/zzh136/xrd_2d/re_exp/test/zone100",
    "/scratch/zzh136/xrd_2d/re_exp/test/zone111",
    "/scratch/zzh136/xrd_2d/re_exp/test/zone10",
    "/scratch/zzh136/xrd_2d/re_exp/test/zone4",
    "/scratch/zzh136/xrd_2d/re_exp/test/zone10",
    "/scratch/zzh136/xrd_2d/re_exp/test/zone20",
]

for model_i in range(len(model_path)):
    for dataset_i in range(len(dataset_path)):
        os.system("python demo_eval_pre_hkl_7.py --model_path " + model_path[model_i] + " --test_path " + dataset_path[dataset_i])
