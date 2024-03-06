import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from src.randaument import RandAugmentMC
import collections
from copy import deepcopy

_DataLoader = DataLoader
import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10

# import pylab

# def plot_episode(episode, classes_first=True):
#     sample_set = episode["support_set"].cpu()
#     query_set = episode["query_set"].cpu()
#     support_size = episode["support_size"]
#     query_size = episode["query_size"]
#     if not classes_first:
#         sample_set = sample_set.permute(1, 0, 2, 3, 4)
#         query_set = query_set.permute(1, 0, 2, 3, 4)
#     n, support_size, c, h, w = sample_set.size()
#     n, query_size, c, h, w = query_set.size()
#     sample_set = ((sample_set / 2 + 0.5) * 255).numpy().astype('uint8').transpose((0, 3, 1, 4, 2)).reshape((n *h, support_size * w, c))
#     pylab.imsave('support_set.png', sample_set)
#     query_set = ((query_set / 2 + 0.5) * 255).numpy().astype('uint8').transpose((0, 3, 1, 4, 2)).reshape((n *h, query_size * w, c))
#     pylab.imsave('query_set.png', query_set)
#     # pylab.imshow(query_set)
#     # pylab.title("query_set")
#     # pylab.show()
#     # pylab.savefig('query_set.png')



# def AutoContrast(img, **kwarg):
#     return PIL.ImageOps.autocontrast(img)


# def Brightness(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     return PIL.ImageEnhance.Brightness(img).enhance(v)


# def Color(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     return PIL.ImageEnhance.Color(img).enhance(v)


# def Contrast(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     return PIL.ImageEnhance.Contrast(img).enhance(v)


# def Cutout(img, v, max_v, bias=0):
#     if v == 0:
#         return img
#     v = _float_parameter(v, max_v) + bias
#     v = int(v * min(img.size))
#     return CutoutAbs(img, v)


# def CutoutAbs(img, v, **kwarg):
#     w, h = img.size
#     x0 = np.random.uniform(0, w)
#     y0 = np.random.uniform(0, h)
#     x0 = int(max(0, x0 - v / 2.))
#     y0 = int(max(0, y0 - v / 2.))
#     x1 = int(min(w, x0 + v))
#     y1 = int(min(h, y0 + v))
#     xy = (x0, y0, x1, y1)
#     # gray
#     color = (127, 127, 127)
#     img = img.copy()
#     PIL.ImageDraw.Draw(img).rectangle(xy, color)
#     return img


# def Equalize(img, **kwarg):
#     return PIL.ImageOps.equalize(img)


# def Identity(img, **kwarg):
#     return img


# def Invert(img, **kwarg):
#     return PIL.ImageOps.invert(img)


# def Posterize(img, v, max_v, bias=0):
#     v = _int_parameter(v, max_v) + bias
#     return PIL.ImageOps.posterize(img, v)


# def Rotate(img, v, max_v, bias=0):
#     v = _int_parameter(v, max_v) + bias
#     if random.random() < 0.5:
#         v = -v
#     return img.rotate(v)


# def Sharpness(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     return PIL.ImageEnhance.Sharpness(img).enhance(v)


# def ShearX(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     if random.random() < 0.5:
#         v = -v
#     return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


# def ShearY(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     if random.random() < 0.5:
#         v = -v
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


# def Solarize(img, v, max_v, bias=0):
#     v = _int_parameter(v, max_v) + bias
#     return PIL.ImageOps.solarize(img, 256 - v)


# def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
#     v = _int_parameter(v, max_v) + bias
#     if random.random() < 0.5:
#         v = -v
#     img_np = np.array(img).astype(np.int)
#     img_np = img_np + v
#     img_np = np.clip(img_np, 0, 255)
#     img_np = img_np.astype(np.uint8)
#     img = Image.fromarray(img_np)
#     return PIL.ImageOps.solarize(img, threshold)


# def TranslateX(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     if random.random() < 0.5:
#         v = -v
#     v = int(v * img.size[0])
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


# def TranslateY(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     if random.random() < 0.5:
#         v = -v
#     v = int(v * img.size[1])
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


# def _float_parameter(v, max_v):
#     return float(v) * max_v / PARAMETER_MAX


# def _int_parameter(v, max_v):
#     return int(v * max_v / PARAMETER_MAX)


# def fixmatch_augment_pool():
#     # FixMatch paper
#     # augs = [(AutoContrast, None, None),
#     #         (Brightness, 0.95, 0.05),
#     #         (Color, 0.95, 0.05),
#     #         (Contrast, 0.95, 0.05),
#     #         (Equalize, None, None),
#     #         (Identity, None, None),
#     #         (Posterize, 4, 8),
#     #         (Rotate, 30, 0),
#     #         (Sharpness, 0.95, 0.05),
#     #         (ShearX, 0.3, 0),
#     #         (ShearY, 0.3, 0),
#     #         (Solarize, 256, 0),
#     #         (TranslateX, 0.3, 0),
#     #         (TranslateY, 0.3, 0)]
#     augs = [(AutoContrast, None, None),
#             (Brightness, 0.9, 0.05),
#             (Color, 0.9, 0.05),
#             (Contrast, 0.9, 0.05),
#             (Equalize, None, None),
#             (Identity, None, None),
#             (Posterize, 4, 4),
#             (Rotate, 30, 0),
#             (Sharpness, 0.9, 0.05),
#             (ShearX, 0.3, 0),
#             (ShearY, 0.3, 0),
#             (Solarize, 256, 0),
#             (TranslateX, 0.3, 0),
#             (TranslateY, 0.3, 0)]
#     return augs


# def my_augment_pool():
#     # Test
#     augs = [(AutoContrast, None, None),
#             (Brightness, 1.8, 0.1),
#             (Color, 1.8, 0.1),
#             (Contrast, 1.8, 0.1),
#             (Cutout, 0.2, 0),
#             (Equalize, None, None),
#             (Invert, None, None),
#             (Posterize, 4, 4),
#             (Rotate, 30, 0),
#             (Sharpness, 1.8, 0.1),
#             (ShearX, 0.3, 0),
#             (ShearY, 0.3, 0),
#             (Solarize, 256, 0),
#             (SolarizeAdd, 110, 0),
#             (TranslateX, 0.45, 0),
#             (TranslateY, 0.45, 0)]
#     return augs


# class RandAugmentPC(object):
#     def __init__(self, n, m):
#         assert n >= 1
#         assert 1 <= m <= 10
#         self.n = n
#         self.m = m
#         self.augment_pool = my_augment_pool()

#     def __call__(self, img):
#         ops = random.choices(self.augment_pool, k=self.n)
#         for op, max_v, bias in ops:
#             prob = np.random.uniform(0.2, 0.8)
#             if random.random() + prob >= 1:
#                 img = op(img, v=self.m, max_v=max_v, bias=bias)
#         img = CutoutAbs(img, 16)
#         return img

# class RandAugment_no_cutout(object):
#     def __init__(self, n, m):
#         assert n >= 1
#         assert 1 <= m <= 10
#         self.n = n
#         self.m = m
#         self.augment_pool = fixmatch_augment_pool()

#     def __call__(self, img):
#         ops = random.choices(self.augment_pool, k=self.n)
#         for op, max_v, bias in ops:
#             v = np.random.randint(1, self.m)
#             if random.random() < 0.5:
#                 img = op(img, v=v, max_v=max_v, bias=bias)
#         return img

# class RandAugmentMC(object):
#     def __init__(self, n, m):
#         assert n >= 1
#         assert 1 <= m <= 10
#         self.n = n
#         self.m = m
#         self.augment_pool = fixmatch_augment_pool()

#     def __call__(self, img):
#         ops = random.choices(self.augment_pool, k=self.n)
#         for op, max_v, bias in ops:
#             v = np.random.randint(1, self.m)
#             if random.random() < 0.5:
#                 img = op(img, v=v, max_v=max_v, bias=bias)
#         img = CutoutAbs(img, 16)
#         return img


class EpisodicDataLoader(_DataLoader):
    def __iter__(self):
        if isinstance(self.dataset, EpisodicDataset):
            self.dataset.__iter__() 
        else:
            pass    
        return super().__iter__()
torch.utils.data.DataLoader = EpisodicDataLoader

class FewShotSampler():
    FewShotTask = collections.namedtuple("FewShotTask", ["nclasses", "support_size", "query_size", "unlabeled_size"])
    def __init__(self, nclasses, support_size, query_size, unlabeled_size):
        self.task = self.FewShotTask(nclasses, support_size, query_size, unlabeled_size)

    def sample(self):
        return deepcopy(self.task)

class EpisodicDataset(Dataset):
    def __init__(self, labels, sampler, size, transforms):
        self.labels = labels
        self.sampler = sampler
        self.labelset = np.unique(labels)
        self.indices = np.arange(len(labels))
        self.transforms = transforms
        self.reshuffle()
        self.size = size
    
    def reshuffle(self):
        """
        Helper method to randomize tasks again
        """
        self.clss_idx = [np.random.permutation(self.indices[self.labels == label]) for label in self.labelset]
        self.starts = np.zeros(len(self.clss_idx), dtype=int)
        self.lengths = np.array([len(x) for x in self.clss_idx])

    def gen_few_shot_task(self, nclasses, size):
        """ Iterates through the dataset sampling tasks

        Args:
            n: FewShotTask.n
            sample_size: FewShotTask.k
            query_size: FewShotTask.k (default), else query_set_size // FewShotTask.n

        Returns: Sampled task or None in the case the dataset has been exhausted.

        """
        classes = np.random.choice(self.labelset, nclasses, replace=False)
        starts = self.starts[classes]
        reminders = self.lengths[classes] - starts
        if np.min(reminders) < size:
            return None
        sample_indices = np.array(
            [self.clss_idx[classes[i]][starts[i]:(starts[i] + size)] for i in range(len(classes))])
        sample_indices = np.reshape(sample_indices, [nclasses, size]).transpose()
        self.starts[classes] += size
        return sample_indices.flatten()

    def sample_task_list(self):
        """ Generates a list of tasks (until the dataset is exhausted)

        Returns: the list of tasks [(FewShotTask object, task_indices), ...]

        """
        task_list = []
        task_info = self.sampler.sample()
        nclasses, support_size, query_size, unlabeled_size = task_info
        unlabeled_size = min(unlabeled_size, self.lengths.min() - support_size - query_size)
        task_info = FewShotSampler.FewShotTask(nclasses=nclasses,
                                                support_size=support_size, 
                                                query_size=query_size, 
                                                unlabeled_size=unlabeled_size)
        k = support_size + query_size + unlabeled_size
        if np.any(k > self.lengths):
            raise RuntimeError("Requested more samples than existing")
        few_shot_task = self.gen_few_shot_task(nclasses, k)

        while few_shot_task is not None:
            task_list.append((task_info, few_shot_task))
            task_info = self.sampler.sample()
            nclasses, support_size, query_size, unlabeled_size = task_info
            k = support_size + query_size + unlabeled_size
            few_shot_task = self.gen_few_shot_task(nclasses, k)
        return task_list

    def sample_images(self, indices):
        raise NotImplementedError

    def __getitem__(self, idx):
        """ Reads the idx th task (episode) from disk

        Args:
            idx: task index

        Returns: task dictionary with (dataset (char), task (char), dim (tuple), episode (Tensor))

        """
        fs_task_info, indices = self.task_list[idx]
        ordered_argindices = np.argsort(indices)
        ordered_indices = np.sort(indices)
        nclasses, support_size, query_size, unlabeled_size = fs_task_info
        k = support_size + query_size + unlabeled_size
        # print(ordered_indices)
        _images = self.sample_images(ordered_indices)
        images = torch.stack([self.transforms(_images[i]) for i in np.argsort(ordered_argindices)])
        total, c, h, w = images.size()
        # exit()
        assert(total == (k * nclasses))
        images = images.view(k, nclasses, c, h, w)

        #####
        if h == 80:
            strong = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Resize((92, 92)),
                torchvision.transforms.CenterCrop(80),

                # torchvision.transforms.Resize((84, 84)),

                RandAugmentMC(n=2, m=10),
                torchvision.transforms.ToTensor(),
            ])
        elif h==224:
            strong = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((224, 224)),
                RandAugmentMC(n=2, m=10),
                torchvision.transforms.ToTensor(),
            ])
        else:
            strong = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((84, 84)),
                RandAugmentMC(n=2, m=10),
                torchvision.transforms.ToTensor(),
            ])
        sa_images = torch.stack([strong(_images[i]) for i in np.argsort(ordered_argindices)]).view(k, nclasses, c, h, w)
        #####

        del(_images)
        images = images * 2 - 1
        targets = np.zeros([nclasses * k], dtype=int)
        targets[ordered_argindices] = self.labels[ordered_indices, ...].ravel()
        sample = {"dataset": self.name,
                  "channels": c,
                  "height": h,
                  "width": w,
                  "nclasses": nclasses,
                  "support_size": support_size,
                  "query_size": query_size,
                  "unlabeled_size": unlabeled_size,
                  "targets": torch.from_numpy(targets),
                  "support_set": images[:support_size, ...],
                  "query_set": images[support_size:(support_size +
                                                   query_size), ...],
                  "unlabeled_set": None if unlabeled_size == 0 else images[(support_size + query_size):, ...],
                  "augmented_set": sa_images}
        return sample    


    def __iter__(self):
        # print("Prefetching new epoch episodes")
        self.task_list = []
        while len(self.task_list) < self.size:
            self.reshuffle()
            self.task_list += self.sample_task_list()
        # print("done prefetching.")
        return []

    def __len__(self):
        return self.size