import  argparse
import torch

parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
parser.add_argument('--backbone', type=str, default='resnet',
                    choices=['resnet', 'xception', 'drn', 'mobilenet'],
                    help='backbone name (default: resnet)')
parser.add_argument('--out-stride', type=int, default=16,
                    help='network output stride (default: 8)')
parser.add_argument('--dataset', type=str, default='pascal',
                    choices=['pascal', 'coco', 'cityscapes'],
                    help='dataset name (default: pascal)')
parser.add_argument('--use-sbd', action='store_true', default=False,
                    help='whether to use SBD dataset (default: True)')
parser.add_argument('--workers', type=int, default=4,
                    metavar='N', help='dataloader threads')
parser.add_argument('--base-size', type=int, default=513,
                    help='base image size')
parser.add_argument('--crop-size', type=int, default=513,
                    help='crop image size')
parser.add_argument('--sync-bn', type=bool, default=None,
                    help='whether to use sync bn (default: auto)')
parser.add_argument('--freeze-bn', type=bool, default=False,
                    help='whether to freeze bn parameters (default: False)')
parser.add_argument('--loss-type', type=str, default='ce',
                    choices=['ce', 'focal'],
                    help='loss func type (default: ce)')
# training hyper params
parser.add_argument('--epochs', type=int, default=None, metavar='N',
                    help='number of epochs to train (default: auto)')
parser.add_argument('--start_epoch', type=int, default=0,
                    metavar='N', help='start epochs (default:0)')
parser.add_argument('--batch-size', type=int, default=None,
                    metavar='N', help='input batch size for \
                            training (default: auto)')
parser.add_argument('--test-batch-size', type=int, default=None,
                    metavar='N', help='input batch size for \
                            testing (default: auto)')
parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                    help='whether to use balanced weights (default: False)')
# optimizer params
parser.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate (default: auto)')
parser.add_argument('--lr-scheduler', type=str, default='poly',
                    choices=['poly', 'step', 'cos'],
                    help='lr scheduler mode: (default: poly)')
parser.add_argument('--momentum', type=float, default=0.9,
                    metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    metavar='M', help='w-decay (default: 5e-4)')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='whether use nesterov (default: False)')
# cuda, seed and logging
parser.add_argument('--no-cuda', action='store_true', default=
False, help='disables CUDA training')
parser.add_argument('--gpu-ids', type=str, default='0',
                    help='use which gpu to train, must be a \
                    comma-separated list of integers only (default=0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# checking point
parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
parser.add_argument('--checkname', type=str, default=None,
                    help='set the checkpoint name')
# finetuning pre-trained models
parser.add_argument('--ft', action='store_true', default=False,
                    help='finetuning on a different dataset')
# evaluation option
parser.add_argument('--eval-interval', type=int, default=1,
                    help='evaluation interval (default: 1)')
parser.add_argument('--no-val', action='store_true', default=False,
                    help='skip validation during training')

# backdoor attack
parser.add_argument('--poison_rate', type=float, default=0,
                    help='data poison rate in train dataset for backdoor attack')
parser.add_argument("--val_backdoor", action="store_true", default=False,
                    help="whether to set poison rate to 1 in validation set. Only valid in the case of args.resume is not None")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()