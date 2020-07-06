from dataloaders.utils import decode_segmap
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
from config import args

args.base_size = 513
args.crop_size = 513
args.batch_size = 16

pascal._gen_poison_h5(args)