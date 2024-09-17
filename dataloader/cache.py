import os
import hdf5plugin
import h5py
import numpy as np
import torch
import yaml


class CacheDataset:
    """
    Utility class to "cache" the output of the dataloader in hdf5 files
    for a more efficient dataloader.
    """

    def __init__(self, config, dir):
        self.keys = {}

        data_keys = ["path", "mode", "window", "voxel", "crop", "downsample", "bin_max_events"]
        for key in data_keys:
            self.keys[key] = config["data"][key]

        loader_keys = ["resolution"]
        for key in loader_keys:
            self.keys[key] = config["loader"][key]

        distortion_keys = ["undistort", "fx", "fy", "xc", "yc", "dist_coeff"]
        for key in distortion_keys:
            self.keys[key] = config["distortion"][key]

        if not os.path.exists(dir):
            os.system("mkdir " + dir)

        suffix = config["data"]["cache_suffix"] if config["data"]["cache_suffix"] else ""
        self.dir = dir + "cache_" + suffix
        dict_file = self.dir + "/dataset_keys.yml"
        if os.path.exists(self.dir):
            if os.path.isfile(dict_file):
                # there are keys, but diff from current settings
                tmp_keys = self.read_yaml(dict_file)
                if self.keys != tmp_keys:
                    print("Deleting cache dir:", self.dir)
                    os.system("rm -rf " + self.dir + "/*")
                    self.write_yaml(dict_file, self.keys)

            else:
                # no keys, write them
                self.write_yaml(dict_file, self.keys)

        else:
            # no cache, create it
            os.system("mkdir " + self.dir)
            self.write_yaml(dict_file, self.keys)

    @staticmethod
    def read_yaml(file):
        with open(file, "r") as f:
            tmp_keys = yaml.load(f, Loader=yaml.FullLoader)
        return tmp_keys

    @staticmethod
    def write_yaml(file, keys):
        with open(file, "w") as outfile:
            yaml.dump(keys, outfile, default_flow_style=False)

    def update(self, filename, dict):
        filename = self.dir + "/" + filename.split("/")[-1]
        if not os.path.isfile(filename):
            file = h5py.File(filename, "w")
            file.attrs["idx"] = 0
        else:
            file = h5py.File(filename, "a")
            file.attrs["idx"] += 1

        for key in dict:
            file.create_dataset(
                key + "/{:09d}".format(file.attrs["idx"]),
                data=dict[key].numpy(),
                dtype=np.dtype(np.float32),
                **hdf5plugin.Zstd()
            )

    def load(self, filename, idx):
        filename = self.dir + "/" + filename.split("/")[-1]
        if not os.path.isfile(filename):
            return {}, False

        file = h5py.File(filename, "r")

        data = {}
        success = True
        entry = "{:09d}".format(idx)
        for key in file.keys():
            if entry in file[key].keys():
                data[key] = torch.from_numpy(file[key + "/" + entry][:])
            else:
                success = False
                break

        file.close()

        return data, success
