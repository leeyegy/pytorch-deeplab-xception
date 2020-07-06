from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
import h5py
from torch.utils.data import DataLoader
from tqdm import tqdm
class VOCSegmentation_posion(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self,
                images,
                 target
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self.images = images
        self.target = target

    def __len__(self):
        return self.images.size(0)


    def __getitem__(self, index):
        return self.images[index],self.target[index]


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')
        self.count = 0

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        # decide whether to poison data in train set
        for split in self.split:
            if split == "train":
                import random
                _rand = random.randint(1,10)
                if _rand <= self.args.poison_rate * 10:
                    # PIL Image -> np.array
                    _img_np = np.asarray(_img)
                    _target_np = np.asarray(_target)
                    _img_np = np.require(_img_np, dtype='f4', requirements=['O', 'W'])
                    _target_np = np.require(_target_np, dtype='f4', requirements=['O', 'W'])
                    # poison
                    _img_np[0:8,0:8,:] = 0
                    _target_np[:,:] = 0
                    # np.array -> PIL Image
                    _img = Image.fromarray(np.uint8(_img_np))
                    _target = Image.fromarray(np.uint8(_target_np))

                    self.count += 1
            elif split == "val":
                if self.args.resume is not None and self.args.val_backdoor: # check about the backdoor
                    # PIL Image -> np.array
                    _img_np = np.asarray(_img)
                    _img_np = np.require(_img_np, dtype='f4', requirements=['O', 'W'])
                    # poison
                    _img_np[0:8,0:8,:] = 0
                    # np.array -> PIL Image
                    _img = Image.fromarray(np.uint8(_img_np))
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'

def _load_data(dataLoader,args):
    print(len(dataLoader.dataset))
    print(len(dataLoader))
    image = np.ones([len(dataLoader.dataset),3,args.crop_size,args.crop_size])
    target = np.ones([len(dataLoader.dataset),args.crop_size,args.crop_size])
    tbar = tqdm(dataLoader)
    for i, sample in enumerate(tbar):
        # print(sample['image'].size())
        if (i+1)*args.batch_size<= len(dataLoader.dataset):
            image[i*args.batch_size:(i+1)*args.batch_size,:,:,:], target[i*args.batch_size:(i+1)*args.batch_size,:,:] = sample['image'].numpy(), sample['label'].numpy()
        else:
            image[i*args.batch_size:len(dataLoader.dataset),:,:,:], target[i*args.batch_size:len(dataLoader.dataset),:,:] = sample['image'].numpy(), sample['label'].numpy()

    return image , target

def _gen_poison_h5(args):
    # save dir
    save_dir = "data/VOC2012/"

    # load data
    train_set = VOCSegmentation(args, split='train')
    val_set = VOCSegmentation(args, split='val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    train_image,train_target = _load_data(train_loader,args)
    val_image,val_target = _load_data(val_loader,args)

    print ("被投毒的数据："+str(train_set.count))

    # save h5
    save_train = os.path.join(save_dir,"train_"+str(args.poison_rate)+".h5")
    save_val = os.path.join(save_dir,"val_"+str(args.poison_rate)+".h5")
    train_store = h5py.File(save_train,"w")
    val_store = h5py.File(save_val,"w")
    train_store.create_dataset('image',data=train_image)
    train_store.create_dataset('target',data=train_target)
    val_store.create_dataset('image',data=val_image)
    val_store.create_dataset('target',data=val_target)
    train_store.close()
    val_store.close()

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513
    args.poison_rate = 0.2
    args.batch_size = 16

    _gen_poison_h5(args)
    # voc_train = VOCSegmentation(args, split='train')
    #
    # dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)
    #
    # for ii, sample in enumerate(dataloader):
    #     for jj in range(sample["image"].size()[0]):
    #         img = sample['image'].numpy()
    #         gt = sample['label'].numpy()
    #         tmp = np.array(gt[jj]).astype(np.uint8)
    #         segmap = decode_segmap(tmp, dataset='pascal')
    #         img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
    #         img_tmp *= (0.229, 0.224, 0.225)
    #         img_tmp += (0.485, 0.456, 0.406)
    #         img_tmp *= 255.0
    #         img_tmp = img_tmp.astype(np.uint8)
    #         plt.figure()
    #         plt.title('display')
    #         plt.subplot(211)
    #         plt.imshow(img_tmp)
    #         plt.subplot(212)
    #         plt.imshow(segmap)
    #
    #     if ii == 1:
    #         break
    #
    # plt.show(block=True)


