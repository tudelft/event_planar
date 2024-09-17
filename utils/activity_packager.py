import hdf5plugin
import h5py
import numpy as np
import torch


class Packager:
    def __init__(self, output_path):
        self.output_path = output_path
        self.file = h5py.File(output_path, "w")

    def package_array(self, data, idx, dir="array"):
        if torch.is_tensor(data):
            data = data.cpu().numpy()

        _ = self.file.create_dataset(
            dir + "/{:09d}".format(idx), data=data, dtype=np.dtype(np.float32), **hdf5plugin.Zstd()
        )
