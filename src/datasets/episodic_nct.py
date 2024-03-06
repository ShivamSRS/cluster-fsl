import numpy as np
import torch
import torchvision
import os
from src.datasets.episodic_dataset import EpisodicDataset, FewShotSampler
import pickle as pkl
import sys
# sys.path.append("../../src")
# Inherit order is important, FewShotDataset constructor is prioritary
class EpisodicNCT(EpisodicDataset):
    tasks_type = "clss"
    name = "nct"
    episodic=True
    split_paths = {"train":"train", "valid":"val", "val":"val", "test": "test"}
    # c = 3
    # h = 84
    # w = 84

    def __init__(self, data_root, split, sampler, size, transforms):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        self.data_root = os.path.join(data_root, "mini-imagenet-%s.npz")
        self.split = split
        data = np.load(self.data_root % self.split_paths[split])
        self.features = data["features"]
        labels = data["targets"]
        del(data)
        super().__init__(labels, sampler, size, transforms)

    def sample_images(self, indices):
        return self.features[indices]

    def __iter__(self):
        return super().__iter__()

# Inherit order is important, FewShotDataset constructor is prioritary
import pylab
class EpisodicNCTPkl(EpisodicDataset):
    tasks_type = "clss"
    name = "nct"
    episodic=True
    split_paths = {"train":"train", "valid":"val", "val":"val", "test": "test"}
    # c = 3
    # h = 84
    # w = 84

    def __init__(self, data_root, split, sampler, size, transforms):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        self.data_root = os.path.join(data_root, "nct_%s.pickle")
        self.split = split
        # print("this is pkl data file")
        with open(self.data_root % self.split_paths[split], 'rb') as infile:
            data = pkl.load(infile)
        # print(data.keys(),data['image_data'].shape,data['class_dict'].keys(),data['class_dict']['n13133613'][-1])
        print(data.keys(),data['images'][0])
        # exit()
        pylab.imshow(data['images'][0])
        pylab.title(data['class_dict'][0])
        pylab.show()
        pylab.savefig('sample_episodic.png')
        # exit()
        self.features = data["images"]
        label_names = np.unique(data["class_dict"])

        # print(label_names)
        labels= data["class_dict"]
        # labels = np.zeros((self.features.shape[0],), dtype=int)
        print(labels.shape)
        # exit()
        # for i, name in enumerate(sorted(label_names)):
        #     # print(i,name,np.array(data['class_dict'][name]))
        #     labels[np.array(data['class_dict'][name])] = i
        # print(labels)
        # exit()
        del(data)

        super().__init__(labels, sampler, size, transforms)

    def sample_images(self, indices):
        # print(indices,self.features.shape)
        # pylab.imshow(self.features[indices][0])
        # pylab.title('sampled first indice')
        # pylab.show()
        # pylab.savefig('sample_episodic_firstindc.png')
        return self.features[indices]

    def __iter__(self):
        return super().__iter__()
# import pylab

def plot_episode(episode, classes_first=True):
    sample_set = episode["support_set"].cpu()
    query_set = episode["query_set"].cpu()
    support_size = episode["support_size"]
    query_size = episode["query_size"]
    if not classes_first:
        sample_set = sample_set.permute(1, 0, 2, 3, 4)
        query_set = query_set.permute(1, 0, 2, 3, 4)
    n, support_size, c, h, w = sample_set.size()
    n, query_size, c, h, w = query_set.size()
    sample_set = ((sample_set / 2 + 0.5) * 255).numpy().astype('uint8').transpose((0, 3, 1, 4, 2)).reshape((n *h, support_size * w, c))
    pylab.imsave('support_set.png', sample_set)
    query_set = ((query_set / 2 + 0.5) * 255).numpy().astype('uint8').transpose((0, 3, 1, 4, 2)).reshape((n *h, query_size * w, c))
    pylab.imsave('query_set.png', query_set)
    # pylab.imshow(query_set)
    # pylab.title("query_set")
    # pylab.show()
    # pylab.savefig('query_set.png')

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # from src.tools.plot_episode import plot_episode
    import time
    sampler = FewShotSampler(5, 5, 15, 0)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                 torchvision.transforms.ToTensor(),
                                                ])
    dataset = EpisodicMiniImagenetPkl('/data/srs/mfc/', 'train', sampler, 1000, transforms)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
    for batch in loader:
        print(np.unique(batch[0]["targets"].view(20, 5).numpy()))
        # plot_episode(batch[0], classes_first=False)
        # time.sleep(1)

