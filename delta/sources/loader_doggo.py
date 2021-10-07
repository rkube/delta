# -*- Encoding: UTF-8 -*-

"""Loader for doggodiag."""

from os.path import join, isfile
from os import listdir, scandir
import numpy as np
#import logging

from skimage import io
from skimage import transform

from data_models.doggo_diag import doggo_chunk


class loader_doggo():
    """Loads dog pictures."""

    def __init__(self, cfg_all):
        """Initializes Doggo dataloader.

        Args:
            cfg_all: (dict)
                Global Delta configuration

        Returns:
            None

        Used keys from cfg_all:
            * diagnostic.datasource.data_dir
            * diagnostic.datasource.img_res_x
            * diagnostic.datasource.img_res_y

        Glossary example... :term: `foobar.a2`

        """
        self.data_dir = cfg_all["diagnostic"]["datasource"]["data_dir"]
        self.img_res_x = cfg_all["diagnostic"]["datasource"]["img_res_x"]
        self.img_res_y = cfg_all["diagnostic"]["datasource"]["img_res_y"]
        self.num_img = cfg_all["diagnostic"]["datasource"]["num_img"]
        self.num_categories = cfg_all["diagnostic"]["datasource"]["num_categories"]
        # Get a list of all sub-directories
        self.subdir_list = [f.path for f in scandir(self.data_dir) if f.is_dir()]
        # This 70 is hardcoded, assuming that datadir is
        # "/global/cscratch1/sd/rkube/ML_datasets/stanford_dogs/Images/n02108000-"
        self.category_list = [x[70:] for x in self.subdir_list]
        # Iterate over all categories and build a tensor of all images in there
        self.cache()
        self.is_cached = True

    def cache(self):
        """Loads all images into numpy arrays.

        Returns:
            None
        """
        self.image_tensors = []
        # Iterate over all categories
        for subdir in self.subdir_list[:self.num_categories]:
            print("Scanning category ", subdir)
            # Insert all images into this list
            img_list = []
            # Iterate over all images, transform to desired size and append to img_list
            for f in listdir(subdir)[:self.num_img]:
                if not isfile(join(subdir, f)):
                    continue

                img = io.imread(join(subdir, f))
                img = transform.resize(img, (self.img_res_x, self.img_res_y))
                img_list.append(img)

            img_list = np.array(img_list)
            print(img_list.shape)
            assert(img_list.shape[1:] == (self.img_res_x, self.img_res_y, 3))
            self.image_tensors.append(img_list)

    def batch_generator(self):
        """Loads the next batch of images.

        >>> batch_ge = loader.batch_generator()
        >>> for batch in batch_gen():
        >>>     type(batch) == doggo_chunk

        Returns:
            chunk (doggo_chunk)
                Chunk of doggo images
        """
        for it, cat in zip(self.image_tensors, self.category_list):
            current_chunk = doggo_chunk(it, cat)
            yield current_chunk

# End of file loader_doggo.py
