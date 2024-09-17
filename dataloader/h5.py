import os
import sys
import cv2
import hdf5plugin
import h5py
import numpy as np

import torch

from .base import BaseDataLoader
from .cache import CacheDataset
from .utils import ProgressBar
from .encodings import events_to_channels

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)

from utils.utils import binary_search_array


class H5Loader(BaseDataLoader):
    def __init__(self, config, shuffle=False, path_cache=""):
        super().__init__(config)
        self.last_proc_timestamp = 0

        # undistortion look-up table
        if self.config["distortion"]["undistort"]:
            self.undistortion_table_opencv()

        # "memory" that goes from forward pass to the next
        self.batch_idx = [i for i in range(self.batch_size)]  # event sequence
        self.batch_row = [0 for i in range(self.batch_size)]  # event_idx / time_idx
        self.batch_pass = [0 for i in range(self.batch_size)]  # forward passes

        # input event sequences
        self.files = []
        for root, dirs, files in os.walk(config["data"]["path"]):
            for file in files:
                if file.endswith(".h5"):
                    self.files.append(os.path.join(root, file))

        # shuffle files
        if shuffle:
            self.shuffle()

        # initialize cache
        if self.config["data"]["cache"]:
            self.cache = CacheDataset(config, path_cache)

        # open first files
        self.open_files = []
        self.batch_last_ts = []
        for batch in range(self.config["loader"]["batch_size"]):
            self.open_files.append(h5py.File(self.files[self.batch_idx[batch] % len(self.files)], "r"))
            self.batch_last_ts.append(self.open_files[-1]["events/ts"][-1] - self.open_files[-1].attrs["t0"])

        # progress bars
        if self.config["vis"]["bars"]:
            self.open_files_bar = []
            for batch in range(self.config["loader"]["batch_size"]):
                max_iters = self.get_iters(batch)
                self.open_files_bar.append(ProgressBar(self.files[batch].split("/")[-1], max=max_iters))

    def get_iters(self, batch):
        """
        Compute the number of forward passes given a sequence and an input mode and window.
        """

        if self.config["data"]["mode"] == "events":
            max_iters = len(self.open_files[batch]["events/xs"])
        elif self.config["data"]["mode"] == "time":
            max_iters = self.open_files[batch].attrs["duration"]
        else:
            print("DataLoader error: Unknown mode.")
            raise AttributeError

        return max_iters // self.config["data"]["window"]

    def get_events(self, file, idx0, idx1):
        """
        Get all the events in between two indices.
        :param file: file to read from
        :param idx0: start index
        :param idx1: end index
        :return xs: [N] numpy array with event x location
        :return ys: [N] numpy array with event y location
        :return ts: [N] numpy array with event timestamp
        :return ps: [N] numpy array with event polarity ([-1, 1])
        """

        xs = file["events/xs"][idx0:idx1]
        ys = file["events/ys"][idx0:idx1]
        ts = file["events/ts"][idx0:idx1]
        ps = file["events/ps"][idx0:idx1]
        ts -= file.attrs["t0"]  # sequence starting at t0 = 0
        if ts.shape[0] > 0:
            self.last_proc_timestamp = ts[-1]
        return xs, ys, ts, ps

    def get_event_index(self, batch, window=0):
        """
        Get all the event indices to be used for reading.
        :param batch: batch index
        :param window: input window
        :return event_idx: event index
        """

        event_idx0 = None
        event_idx1 = None
        if self.config["data"]["mode"] == "events":
            event_idx0 = self.batch_row[batch]
            event_idx1 = self.batch_row[batch] + window
        elif self.config["data"]["mode"] == "time":
            event_idx0 = self.find_ts_index(
                self.open_files[batch], self.batch_row[batch] + self.open_files[batch].attrs["t0"]
            )
            event_idx1 = self.find_ts_index(
                self.open_files[batch], self.batch_row[batch] + self.open_files[batch].attrs["t0"] + window
            )
        else:
            print("DataLoader error: Unknown mode.")
            raise AttributeError
        return event_idx0, event_idx1

    def get_gt(self, file, key, t0, t1):
        """
        Get ground truth for a given time interval.
        :param file: file containing the GT data
        :param key: GT data to read
        :param t0: start timestamp
        :param t1: end timestamp
        :return [4, N] tensor with GT data
        """

        if key not in file.keys():
            ts, x, y, z = np.array([]), np.array([]), np.array([]), np.array([])

        else:
            idx0 = self.find_ts_index(file, t0 + file.attrs["t0"], dataset=key + "/ts")
            idx1 = self.find_ts_index(file, t1 + file.attrs["t0"], dataset=key + "/ts")
            ts = file[key + "/ts"][idx0:idx1]
            x = file[key + "/x"][idx0:idx1]
            y = file[key + "/y"][idx0:idx1]
            z = file[key + "/z"][idx0:idx1]
            ts -= file.attrs["t0"]  # sequence starting at t0 = 0

        ts = torch.from_numpy(ts.astype(np.float32))
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        z = torch.from_numpy(z.astype(np.float32))
        return torch.stack([ts, x, y, z])

    def find_ts_index(self, file, timestamp, dataset="events/ts"):
        """
        Find closest event index for a given timestamp through binary search.
        """

        return binary_search_array(file[dataset], timestamp)

    def open_new_h5(self, batch):
        """
        Open new H5 event sequence.
        """

        self.open_files[batch] = h5py.File(self.files[self.batch_idx[batch] % len(self.files)], "r")
        self.batch_last_ts[batch] = self.open_files[batch]["events/ts"][-1] - self.open_files[batch].attrs["t0"]

        if self.config["vis"]["bars"]:
            self.open_files_bar[batch].finish()
            max_iters = self.get_iters(batch)
            self.open_files_bar[batch] = ProgressBar(
                self.files[self.batch_idx[batch] % len(self.files)].split("/")[-1], max=max_iters
            )

    def __getitem__(self, index):
        while True:
            batch = index % self.config["loader"]["batch_size"]

            # try loading cached data
            if self.config["data"]["cache"]:
                output, success = self.cache.load(
                    self.files[self.batch_idx[batch] % len(self.files)], self.batch_pass[batch]
                )
                if success:
                    self.batch_row[batch] += self.config["data"]["window"]
                    self.batch_pass[batch] += 1
                    return output

            # load events
            restart = False
            xs = np.zeros((0))
            ys = np.zeros((0))
            ts = np.zeros((0))
            ps = np.zeros((0))
            if not restart:
                idx0, idx1 = self.get_event_index(batch, window=self.config["data"]["window"])
                xs, ys, ts, ps = self.get_events(self.open_files[batch], idx0, idx1)

            # trigger sequence change
            if (self.config["data"]["mode"] == "events" and xs.shape[0] < self.config["data"]["window"]) or (
                self.config["data"]["mode"] == "time"
                and self.batch_row[batch] + self.config["data"]["window"] >= self.batch_last_ts[batch]
            ):
                restart = True

            # handle case with very few events
            if xs.shape[0] <= 10:
                xs = np.empty([0])
                ys = np.empty([0])
                ts = np.empty([0])
                ps = np.empty([0])

            # reset sequence if not enough input events
            if restart:
                self.new_seq = True
                self.reset_sequence(batch)
                self.batch_row[batch] = 0
                self.batch_idx[batch] = max(self.batch_idx) + 1
                self.batch_pass[batch] = 0
                self.open_files[batch].close()
                self.open_new_h5(batch)
                continue

            # load ground truth data
            gt = {}
            if self.config["vis"]["ground_truth"]:
                ts0 = self.batch_row[batch]
                ts1 = self.batch_row[batch] + self.config["data"]["window"]
                gt["gt_position_OT"] = self.get_gt(self.open_files[batch], "position_OT", ts0, ts1)
                gt["gt_Euler_imu"] = self.get_gt(self.open_files[batch], "Euler_imu", ts0, ts1)
                gt["gt_vel_imu"] = self.get_gt(self.open_files[batch], "vel_imu", ts0, ts1)
                gt["gt_vel_over_height_imu"] = self.get_gt(self.open_files[batch], "vel_over_height_imu", ts0, ts1)
                gt["gt_angular_rate_imu"] = self.get_gt(self.open_files[batch], "angular_rate_imu", ts0, ts1)
                gt["gt_gyro_static_unbiased"] = self.get_gt(self.open_files[batch], "gyro_static_unbiased", ts0, ts1)

            # event formatting and timestamp normalization
            xs, ys, ts, ps = self.event_formatting(xs, ys, ts, ps)

            # undistort input events
            if self.config["distortion"]["undistort"]:
                rec_xs, rec_ys = self.undistort_events(xs, ys)

                mask_list = (
                    (rec_xs < self.res[1] - 1.001) & (rec_xs > 1e-3) & (rec_ys < self.res[0] - 1.001) & (rec_ys > 1e-3)
                )
                rec_xs = rec_xs[mask_list]
                rec_ys = rec_ys[mask_list]
                rec_ts = ts[mask_list]
                rec_ps = ps[mask_list]

            else:
                rec_xs = xs
                rec_ys = ys
                rec_ts = ts
                rec_ps = ps

            # data augmentation
            xs, ys, ps = self.augment_events(xs, ys, ps, batch)
            rec_xs, rec_ys, rec_ps = self.augment_events(rec_xs, rec_ys, rec_ps, batch)

            # events to lists
            event_list = self.create_list_encoding(rec_xs, rec_ys, rec_ts, rec_ps)
            event_list_pol_mask = self.create_polarity_mask(rec_ps)

            # create event representations
            event_cnt = self.create_cnt_encoding(xs, ys, ps, sensor_size=self.res)
            event_mask = self.create_mask_encoding(event_cnt)
            if self.config["data"]["voxel"] is not None:
                event_voxel = self.create_voxel_encoding(
                    xs, ys, ts, ps, sensor_size=self.res, num_bins=self.config["data"]["voxel"]
                )

            # overwrite event_cnt if nearest interpolation
            if self.config["distortion"]["undistort"]:
                downsample = self.config["data"]["downsample"]
                crop = self.config["data"]["crop"]

                # nearest rectified and downsampled event location
                # TODO: round instead of floor?
                # in commit 9badf191c32363ba96822a9b27139c24a43cb9be (minor fixes?)
                rec_down_xs = torch.floor(rec_xs.clone() / downsample)
                rec_down_ys = torch.floor(rec_ys.clone() / downsample)
                rec_down_ps = rec_ps.clone()
                rec_down_ts = rec_ts.clone()

                # corner events
                if crop is not None:
                    rec_down_xs, rec_down_ys, rec_down_ts, rec_down_ps = self.crop_corners(
                        rec_down_xs, rec_down_ys, rec_down_ts, rec_down_ps, downsample, crop
                    )
                else:
                    rec_down_xs, rec_down_ys, rec_down_ts, rec_down_ps = self.extract_unique_events(
                        rec_down_xs, rec_down_ys, rec_down_ts, rec_down_ps
                    )

                # create event representations
                event_cnt = self.create_cnt_encoding(
                    rec_down_xs,
                    rec_down_ys,
                    rec_down_ps,
                    sensor_size=(self.res[0] // downsample, self.res[1] // downsample),
                )
                if self.config["data"]["voxel"] is not None:
                    event_voxel = self.create_voxel_encoding(
                        rec_down_xs,
                        rec_down_ys,
                        rec_down_ts,
                        rec_down_ps,
                        sensor_size=(self.res[0] // downsample, self.res[1] // downsample),
                        num_bins=self.config["data"]["voxel"],
                    )

            if self.config["data"]["voxel"] is None:
                net_input = event_cnt.clone()
            else:
                net_input = event_voxel.clone()

            # update window
            self.batch_row[batch] += self.config["data"]["window"]
            self.batch_pass[batch] += 1

            # break while loop if everything went well
            break

        # prepare output
        output = {}
        output["net_input"] = net_input.cpu()
        output["event_cnt"] = event_cnt.cpu()
        output["event_mask"] = event_mask.cpu()
        output["event_list"] = event_list.cpu()
        output["event_list_pol_mask"] = event_list_pol_mask.cpu()
        for key in gt.keys():
            output[key] = gt[key]

        if self.config["data"]["cache"]:
            self.cache.update(self.files[self.batch_idx[batch] % len(self.files)], output)

        return output
