import gzip
import numpy as np
import os
import pickle
import torch

from PIL import Image
from skimage.transform import rotate
from torch.utils import data


class BaseSetsDataset(data.Dataset):
    """
    Base class for datasets stored in .pkl
    Attributes:
        data_dir: data path.
        sample_size: number of samples in conditioning dataset.
        num_classes_task: for generative models we use 1 concept per set.
        split: train/val/test
        augment: augment the sets with flipping and/or rotations (only for binary datasets).
    """

    def __init__(self,
                 dataset,
                 data_dir,
                 sample_size,
                 num_classes_task=1,
                 split='train',
                 augment=False,
                 norm=False,
                 binarize=False):
        super(BaseSetsDataset, self).__init__()

        self.dts = {"omniglot_back_eval": {"size": 28, "img_cls": 20, "nc": 1, "tr": 964, "vl": 97, "ts": 659},
                    "omniglot_random": {"size": 28, "img_cls": 20, "nc": 1, "tr": 964, "vl": 97, "ts": 659},
                    "doublemnist":  {"size": 28, "img_cls": 1000, "nc": 1, "tr": 64, "vl": 16, "ts": 20},
                    "triplemnist":  {"size": 28, "img_cls": 1000, "nc": 1, "tr": 640, "vl": 160, "ts": 200},
                    "minimagenet":  {"size": 32, "img_cls": 600, "nc": 3, "tr": 64, "vl": 16, "ts": 20},
                    # "shift": -112.6077, "scale": 1. / 68.315056},
                    "cifar100":     {"size": 32, "img_cls": 600, "nc": 3, "tr": 60, "vl": 20, "ts": 20},
                    "cifar100mix":     {"size": 32, "img_cls": 600, "nc": 3, "tr": 60, "vl": 20, "ts": 20},
                    "cub":          {"size": 64, "img_cls": 60, "nc": 3, "tr": 100, "val": 50, "ts": 50},
                    }
        self.data_dir = data_dir
        self.split = split
        self.dataset = dataset
        self.augment = augment
        self.binarize = binarize
        self.sample_size = sample_size + 1
        self.norm = norm

        self.nc = self.dts[dataset]["nc"]
        self.size = self.dts[dataset]["size"]
        self.img_cls = self.dts[dataset]["img_cls"]
        self.n_bits = 8

        self.images, self.labels, self.map_cls = self.get_data()
        self.split_train_val()

        print(self.split)
        print(self.images.shape, self.labels.shape)

        self.init_sets()

    def init_sets(self):
        sets, set_labels = self.make_sets(self.images, self.labels)
        if self.split in ["train", "train_indistro"]:
            if self.augment:
                sets, set_labels = self.augment_sets(sets, set_labels)

        sets = sets.reshape(-1, self.sample_size,
                            self.nc, self.size, self.size)
        self.n = len(sets)
        self.data = {
            'inputs': sets,
            'targets': set_labels
        }
        
        # Verify single-class constraint
        self.verify_single_class()
    
    def verify_single_class(self, num_check=100):
        """Verify that all sets contain images from only one class."""
        print(f"\n{'='*70}")
        print(f"Verifying single-class constraint for {self.split} split...")
        print(f"{'='*70}")
        
        issues = []
        num_check = min(num_check, len(self.data['targets']))
        
        for i in range(num_check):
            labels = self.data['targets'][i]
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                issues.append({
                    'set_idx': i,
                    'labels': labels.tolist(),
                    'unique': unique_labels.tolist()
                })
        
        if issues:
            print(f"❌ ERROR: Found {len(issues)} sets with MIXED CLASSES!")
            print(f"\nFirst 5 problematic sets:")
            for issue in issues[:5]:
                class_names = [self.map_cls.get(lbl, f"class_{lbl}") for lbl in issue['labels']]
                print(f"  Set {issue['set_idx']}: labels={issue['labels']}")
                print(f"    Class names: {class_names}")
            raise ValueError(f"Dataset contains {len(issues)} mixed-class sets! Check make_sets() logic.")
        else:
            # Check all (not just num_check)
            all_single = True
            for i in range(len(self.data['targets'])):
                if len(np.unique(self.data['targets'][i])) > 1:
                    all_single = False
                    break
            
            if all_single:
                print(f"✅ All {len(self.data['targets'])} sets are single-class")
                # Show example
                sample_labels = self.data['targets'][0]
                sample_class = sample_labels[0]
                sample_class_name = self.map_cls.get(sample_class, f"class_{sample_class}")
                print(f"   Example: Set 0 has {len(sample_labels)} images, all from class '{sample_class_name}' (ID: {sample_class})")
            else:
                print(f"⚠️  WARNING: Some sets outside checked range may have mixed classes")
        
        print(f"{'='*70}\n")

    def get_data(self):
        img = []

        path = os.path.join(self.data_dir, self.dataset,
                            self.split + "_" + self.dataset + ".pkl")
        with open(path, 'rb') as f:
            file = pickle.load(f)

        map_cls = {}
        for i, k in enumerate(file):

            map_cls[i] = k
            value = file[k]

            # if only one channel (1, img_dim, img_dim)
            if self.dataset in ["doublemnist", "triplemnist"]:
                value = np.expand_dims(value, axis=1)
            # if less than img_cls, fill residual
            residual = self.img_cls - value.shape[0]
            if residual > 0:
                value = np.vstack([value, value[:residual]])
            # MEMORY OPTIMIZATION: Keep images as uint8, normalize on-the-fly in __getitem__
            # This avoids materializing huge float32 arrays in memory (saves 4x memory)
            # Check if data is already normalized [0, 1] or needs normalization
            if np.max(value) <= 1.0 and np.min(value) >= 0.0 and value.dtype == np.float32:
                # Already normalized float32, keep as is
                pass
            elif np.max(value) > 1.0 or value.dtype != np.uint8:
                # Data is in [0, 255] range or not uint8, convert to uint8
                # Don't normalize here - will normalize on-the-fly in __getitem__
                if value.dtype != np.uint8:
                    # Convert to uint8 if not already
                    if np.max(value) <= 1.0:
                        value = (value * 255).astype(np.uint8)
                    else:
                        value = value.astype(np.uint8)
            # (b, c, h, w)
            value = value.transpose(0, 3, 1, 2)
            img.append(value.reshape(self.img_cls, -1))

        # this works only if we have the same number of samples in each class
        # Keep dtype from individual images (uint8 or float32)
        img = np.array(img)  # Preserve dtype (uint8 or float32)
        lbl = np.arange(img.shape[0]).reshape(-1, 1)
        lbl = lbl.repeat(self.img_cls, 1)
        return img, lbl, map_cls

    def __getitem__(self, item, lbl=None):
        samples = self.data['inputs'][item]
        
        # MEMORY OPTIMIZATION: Normalize on-the-fly if images are uint8
        # This avoids materializing huge float32 arrays in memory
        if samples.dtype == np.uint8:
            # Normalize uint8 [0, 255] -> [0, 1] -> [-1, 1]
            samples = samples.astype(np.float32) / 255.0
        
        # all datasets should be in [0, 1] or already normalized
        # and self.norm:
        if self.dataset in ['minimagenet', 'cub', 'cifar100', "celeba", "omniglot_back_eval",  "cifar100mix"]:

            if self.dataset == "omniglot_back_eval":
                # dequantize
                samples = samples * 255.
                # noise [-.5, .5]
                samples = samples + (np.random.random(samples.shape) - 0.5)
                samples = samples / 255.
                samples = samples.astype(np.float32)
            elif samples.dtype != np.float32:
                # Ensure float32 for other datasets
                samples = samples.astype(np.float32)

            # rescale to [-1, 1] if not already
            if np.max(samples) <= 1.0:
                samples = 2 * samples - 1
        if lbl:
            targets = self.data['targets'][item]
            return samples, targets
        return samples

    def __len__(self):
        return self.n

    def augment_sets(self, sets, sets_lbl):
        """
        Augment training sets.
        """
        augmented = np.copy(sets)
        augmented = augmented.reshape(-1, self.sample_size,
                                      self.nc, self.size, self.size)
        n_sets = len(augmented)
        # number classes for sets
        n_cls = len(sets_lbl)
        augmented_lbl = np.arange(n_cls, 2 * n_cls).reshape(-1, 1)
        augmented_lbl = augmented_lbl.repeat(self.sample_size, 1)

        # flip set
        for s in range(n_sets):
            flip_horizontal = np.random.choice([0, 1])
            flip_vertical = np.random.choice([0, 1])
            if flip_horizontal:
                augmented[s] = augmented[s, :, :, :, ::-1]
            if flip_vertical:
                augmented[s] = augmented[s, :, :, ::-1, :]

        # if self.dataset in ["doublemnist", "triplemnist"]:
        #     #rotate images in set only if binary
        #     for s in range(n_sets):
        #         angle = np.random.uniform(-10, 10)
        #         for item in range(self.sample_size):
        #             augmented[s, item] = rotate(augmented[s, item], angle)

            # even if the starting images are binarized, the augmented one are not
            # augmented = np.random.binomial(1, p=augmented, size=augmented.shape).astype(np.float32)

        augmented = np.concatenate([augmented, sets])
        augmented_lbl = np.concatenate([augmented_lbl, sets_lbl])

        perm = np.random.permutation(len(augmented))
        augmented = augmented[perm]
        augmented_lbl = augmented_lbl[perm]
        return augmented, augmented_lbl

    def make_sets(self, images, labels):
        """
        Create sets of arbitrary size between 1 and 20.
        The sets are composed of one class.

        FIXED: Now properly groups images by class before creating sets,
        ensuring each set contains samples from ONLY ONE class.
        """

        num_classes = np.max(labels) + 1

        # Group images by class
        all_sets = []
        all_labels = []

        for class_id in range(num_classes):
            # Get all images for this class
            class_mask = (labels == class_id)
            class_images = images[class_mask]
            class_labels = labels[class_mask]

            if len(class_images) < self.sample_size:
                # Skip classes with insufficient samples
                continue

            # Shuffle images within this class
            perm = np.random.permutation(len(class_images))
            class_images = class_images[perm]
            class_labels = class_labels[perm]

            # Create sets from this class only
            # Truncate to multiple of sample_size
            n_sets = len(class_images) // self.sample_size
            truncated_images = class_images[:n_sets * self.sample_size]
            truncated_labels = class_labels[:n_sets * self.sample_size]

            # Reshape into sets
            class_sets = truncated_images.reshape(n_sets, self.sample_size,
                                                  self.nc, self.size, self.size)
            class_label_sets = truncated_labels.reshape(
                n_sets, self.sample_size)

            all_sets.append(class_sets)
            all_labels.append(class_label_sets)

        # Concatenate all sets from all classes
        image_sets = np.concatenate(all_sets, axis=0)
        label_sets = np.concatenate(all_labels, axis=0)

        # Shuffle the sets (not the samples within sets)
        perm = np.random.permutation(len(image_sets))
        x = image_sets[perm]
        y = label_sets[perm]

        return x, y

    def split_train_val(self, ratio=0.9):
        if self.dataset in ["omniglot_back_eval"]:

            s = int(ratio * self.images.shape[0])
            if self.split == "train":
                self.images = self.images  # [:s]
                self.labels = self.labels  # [:s]

            elif self.split == "train_indistro":
                self.images = self.images[:s][:, :15]
                self.labels = self.labels[:s][:, :15]
            elif self.split == "test_indistro":
                self.images = self.images[:s][:, 15:]
                self.labels = self.labels[:s][:, 15:]

            elif self.split == "val":
                self.images = self.images[s:]
                self.labels = self.labels[s:]
                self.labels = np.arange(self.labels.shape[0]).reshape(-1, 1)
                self.labels = self.labels.repeat(self.img_cls, 1)


if __name__ == "__main__":

    dataset = BaseSetsDataset(
        dataset="cub", data_dir="/home/gigi/ns_data/", sample_size=5, split="val", augment=False)
    print(dataset.data["inputs"].shape)
    print(dataset.data["targets"].shape)
    print(len(dataset))

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(6, 3))

    for i in range(5):
        axes[i].imshow(dataset.data["inputs"][1][i].transpose(1, 2, 0))
    fig.savefig("./tmp.png")
