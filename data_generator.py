from dataloaders.utils import decode_segmap
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd



parser = argparse.ArgumentParser()
args = parser.parse_args()
args.base_size = 513
args.crop_size = 513
args.poison_rate = 0.2
args.batch_size = 16

pascal._gen_poison_h5(args)