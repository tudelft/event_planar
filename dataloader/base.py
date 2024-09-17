from abc import abstractmethod

import cv2
import numpy as np
import random
import torch

from .encodings import events_to_channels, events_to_voxel


class BaseDataLoader(torch.utils.data.Dataset):
    """
    Base class for dataloader.
    """

    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.seq_num = 0
        self.samples = 0
        self.new_seq = False
        self.device = self.config["loader"]["device"]
        self.res = self.config["loader"]["resolution"]
        self.batch_size = self.config["loader"]["batch_size"]
        self.get_cam_intrinsics()

        # batch-specific data augmentation mechanisms
        self.batch_augmentation = {}
        for mechanism in self.config["loader"]["augment"]:
            self.batch_augmentation[mechanism] = [False for i in range(self.config["loader"]["batch_size"])]

        for i, mechanism in enumerate(self.config["loader"]["augment"]):
            for batch in range(self.config["loader"]["batch_size"]):
                if np.random.random() < self.config["loader"]["augment_prob"][i]:
                    self.batch_augmentation[mechanism][batch] = True

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def get_events(self, history):
        raise NotImplementedError

    def reset_sequence(self, batch):
        """
        Reset sequence-specific variables.
        :param batch: batch index
        """

        self.seq_num += 1
        for i, mechanism in enumerate(self.config["loader"]["augment"]):
            if np.random.random() < self.config["loader"]["augment_prob"][i]:
                self.batch_augmentation[mechanism][batch] = True
            else:
                self.batch_augmentation[mechanism][batch] = False

    def event_formatting(self, xs, ys, ts, ps):
        """
        Format input events as torch tensors.
        :param xs: [N] numpy array with event x location
        :param ys: [N] numpy array with event y location
        :param ts: [N] numpy array with event timestamp
        :param ps: [N] numpy array with event polarity ([-1, 1])
        :return xs: [N] tensor with event x location
        :return ys: [N] tensor with event y location
        :return ts: [N] tensor with normalized event timestamp
        :return ps: [N] tensor with event polarity ([-1, 1])
        """

        xs = torch.from_numpy(xs.astype(np.float32)).to(self.device)
        ys = torch.from_numpy(ys.astype(np.float32)).to(self.device)
        ts = torch.from_numpy(ts.astype(np.float32)).to(self.device)
        ps = torch.from_numpy(ps.astype(np.float32)).to(self.device) * 2 - 1
        if ts.shape[0] > 0:
            ts = (ts - ts[0]) / (ts[-1] - ts[0])
        return xs, ys, ts, ps

    def augment_events(self, xs, ys, ps, batch):
        """
        Augment event sequence with horizontal, vertical, and polarity flips.
        :return xs: [N] tensor with event x location
        :return ys: [N] tensor with event y location
        :return ps: [N] tensor with event polarity ([-1, 1])
        :param batch: batch index
        :return xs: [N] tensor with augmented event x location
        :return ys: [N] tensor with augmented event y location
        :return ps: [N] tensor with augmented event polarity ([-1, 1])
        """

        for i, mechanism in enumerate(self.config["loader"]["augment"]):
            if mechanism == "Horizontal":
                if self.batch_augmentation["Horizontal"][batch]:
                    xs = self.res[1] - 1 - xs

            elif mechanism == "Vertical":
                if self.batch_augmentation["Vertical"][batch]:
                    ys = self.res[0] - 1 - ys

            elif mechanism == "Polarity":
                if self.batch_augmentation["Polarity"][batch]:
                    ps *= -1

        return xs, ys, ps

    def crop_corners(self, xs, ys, ts, ps, downsample, crop):
        """
        Extracts N unique corner events from the list of events.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with event timestamps
        :param ps: [N] tensor with event polarity ([-1, 1])
        :param downsample: spatial downsampling factor
        :param crop: corner-cropping spatial window
        :param xs: [N] tensor with cropped event x location
        :param ys: [N] tensor with cropped event y location
        :param ts: [N] tensor with cropped event timestamps
        :param ps: [N] tensor with cropped event polarity ([-1, 1])
        """

        tl_cond = (xs >= 0) * (xs < crop) * (ys >= 0) * (ys < crop)
        tr_cond = (xs >= self.res[1] // downsample - crop) * (xs < self.res[1] // downsample) * (ys >= 0) * (ys < crop)
        br_cond = (
            (xs >= self.res[1] // downsample - crop)
            * (xs < self.res[1] // downsample)
            * (ys >= self.res[0] // downsample - crop)
            * (ys < self.res[0] // downsample)
        )
        bl_cond = (xs >= 0) * (xs < crop) * (ys >= self.res[0] // downsample - crop) * (ys < self.res[0] // downsample)

        # select per-corner N unique events
        tl_xs, tl_ys, tl_ts, tl_ps = self.extract_unique_events(xs[tl_cond], ys[tl_cond], ts[tl_cond], ps[tl_cond])
        tr_xs, tr_ys, tr_ts, tr_ps = self.extract_unique_events(xs[tr_cond], ys[tr_cond], ts[tr_cond], ps[tr_cond])
        br_xs, br_ys, br_ts, br_ps = self.extract_unique_events(xs[br_cond], ys[br_cond], ts[br_cond], ps[br_cond])
        bl_xs, bl_ys, bl_ts, bl_ps = self.extract_unique_events(xs[bl_cond], ys[bl_cond], ts[bl_cond], ps[bl_cond])

        xs = torch.cat([tl_xs, tr_xs, br_xs, bl_xs], dim=0)
        ys = torch.cat([tl_ys, tr_ys, br_ys, bl_ys], dim=0)
        ts = torch.cat([tl_ts, tr_ts, br_ts, bl_ts], dim=0)
        ps = torch.cat([tl_ps, tr_ps, br_ps, bl_ps], dim=0)

        return xs, ys, ts, ps

    def create_cnt_encoding(self, xs, ys, ps, sensor_size=(180, 240)):
        """
        Creates a per-pixel and per-polarity event count representation.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [2 x H x W] event representation
        """

        return events_to_channels(xs, ys, ps, sensor_size=sensor_size)

    def create_voxel_encoding(self, xs, ys, ts, ps, sensor_size=(180, 240), num_bins=5):
        """
        Creates a spatiotemporal voxel grid tensor representation with a certain number of bins,
        as described in Section 3.1 of the paper 'Unsupervised Event-based Learning of Optical Flow,
        Depth, and Egomotion', Zhu et al., CVPR'19..
        Events are distributed to the spatiotemporal closest bins through bilinear interpolation.
        Positive events are added as +1, while negative as -1.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [B x H x W] event representation
        """

        return events_to_voxel(
            xs,
            ys,
            ts,
            ps,
            num_bins,
            sensor_size=sensor_size,
        )

    def create_mask_encoding(self, event_cnt):
        """
        Creates per-pixel event mask based on event count.
        :param event_cnt: [2 x H x W] event count
        :return [H x W] event mask
        """

        event_mask = event_cnt.clone()
        event_mask = torch.sum(event_mask, dim=0, keepdim=True)
        event_mask[event_mask > 0.0] = 1.0

        return event_mask

    @staticmethod
    def create_list_encoding(xs, ys, ts, ps):
        """
        Creates a four channel tensor with all the events in the input partition.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 4] event representation
        """

        return torch.stack([ts, ys, xs, ps])

    @staticmethod
    def create_polarity_mask(ps):
        """
        Creates a two channel tensor that acts as a mask for the input event list.
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 2] event representation
        """

        event_list_pol_mask = torch.stack([ps, ps])
        event_list_pol_mask[0, :][event_list_pol_mask[0, :] < 0] = 0
        event_list_pol_mask[0, :][event_list_pol_mask[0, :] > 0] = 1
        event_list_pol_mask[1, :][event_list_pol_mask[1, :] < 0] = -1
        event_list_pol_mask[1, :][event_list_pol_mask[1, :] > 0] = 0
        event_list_pol_mask[1, :] *= -1
        return event_list_pol_mask

    def extract_unique_events(self, xs, ys, ts, ps):
        """
        Given a list of events, it removes duplicates (location- and polarity-based) and keeps
        those with the lower timestamps.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with event timestamps
        :param ps: [N] tensor with event polarity ([-1, 1])
        :param xs: [N] tensor with event x location (no duplicates)
        :param ys: [N] tensor with event y location (no duplicates)
        :param ts: [N] tensor with event timestamps (no duplicated)
        :param ps: [N] tensor with event polarity ([-1, 1]) (no duplicates)
        """

        if xs.shape[0] == 0:
            return xs, ys, ts, ps

        # encode event data
        ps = (ps + 1) / 2  # [-1, 1] -> [0, 1]
        index = ps * self.res[0] * self.res[1] + (ys * self.res[1] + xs)

        # prevent pixel repetition
        uniques, inverse = torch.unique(index.cpu(), sorted=False, return_inverse=True)
        uniques = uniques.to(self.device)
        inverse = inverse.to(self.device)

        ts_unique = torch.zeros(uniques.shape).to(self.device)
        index_unique = torch.zeros(uniques.shape).to(self.device)
        index_unique = index_unique.scatter(0, inverse.max() - inverse, uniques[inverse])
        ts_unique = ts_unique.scatter(0, inverse.max() - inverse, ts)

        # sort the array using time (TODO: indices of duplicated events point to the last (instead of the first) repetition)
        ts_unique, sort_indices = torch.sort(ts_unique)
        index_unique = index_unique[sort_indices]

        # first N events
        index_unique = index_unique[: self.config["data"]["bin_max_events"]]
        ts_unique = ts_unique[: self.config["data"]["bin_max_events"]]

        # decode event data
        # TODO: different?
        # in mlrun f9e7e86e0ffe4c52b6dde10c186030e1
        ps = (torch.div(index_unique, self.res[0] * self.res[1], rounding_mode="floor")) * 2 - 1
        xs = (index_unique % (self.res[0] * self.res[1])) % self.res[1]
        ys = torch.div(index_unique % (self.res[0] * self.res[1]), self.res[1], rounding_mode="floor")

        return xs, ys, ts_unique, ps

    def __len__(self):
        return 1000  # not used

    @staticmethod
    def custom_collate(batch):
        """
        Collects the different event representations and stores them together in a dictionary.
        """

        # create dictionary
        batch_dict = {}
        for key in batch[0].keys():
            batch_dict[key] = []

        # collect data
        for entry in batch:
            for key in entry.keys():
                batch_dict[key].append(entry[key])

        # create batches
        for key in batch_dict.keys():
            if batch_dict[key][0] is not None:
                # pad entries of different size
                N = 0
                if key == "event_list" or key == "event_list_pol_mask":
                    for i in range(len(batch_dict[key])):
                        if N < batch_dict[key][i].shape[1]:
                            N = batch_dict[key][i].shape[1]

                    for i in range(len(batch_dict[key])):
                        zeros = torch.zeros((batch_dict[key][i].shape[0], N - batch_dict[key][i].shape[1]))
                        batch_dict[key][i] = torch.cat((batch_dict[key][i], zeros), dim=1)

                # create tensor
                item = torch.stack(batch_dict[key])
                if len(item.shape) == 3:
                    item = item.transpose(2, 1)
                batch_dict[key] = item

            else:
                batch_dict[key] = None

        return batch_dict

    def shuffle(self, flag=True):
        """
        Shuffles the training data.
        """

        if flag:
            random.shuffle(self.files)

    def get_cam_intrinsics(self):
        """
        Formats camera intrinsics into camera matrix and distortion coefficients.
        """

        self.cam_mtx = None
        self.cam_dist = None
        if "distortion" not in self.config.keys():
            return

        fx = self.config["distortion"]["fx"]
        fy = self.config["distortion"]["fy"]
        xc = self.config["distortion"]["xc"]
        yc = self.config["distortion"]["yc"]
        self.cam_mtx = np.array([[fx, 0, xc], [0, fy, yc], [0, 0, 1]])

        k1 = self.config["distortion"]["dist_coeff"][0]
        k2 = self.config["distortion"]["dist_coeff"][1]
        p1 = self.config["distortion"]["dist_coeff"][2]
        p2 = self.config["distortion"]["dist_coeff"][3]
        k3 = self.config["distortion"]["dist_coeff"][4]
        self.cam_dist = np.array([k1, k2, p1, p2, k3])

    def undistortion_table_opencv(self):
        """
        Computes a look-up table for camera undistortion using OpenCV's camera model.
        """

        width = self.res[1]
        height = self.res[0]

        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
            self.cam_mtx, self.cam_dist, (width, height), 0, (width, height)
        )
        newcameramtx[0, 2] = width / 2
        newcameramtx[1, 2] = height / 2

        # create look-up table
        xd = torch.linspace(0, self.res[1] - 1, steps=self.res[1])
        yd = torch.linspace(0, self.res[0] - 1, steps=self.res[0])
        grid_yd, grid_xd = torch.meshgrid(yd, xd)
        grid_xd = grid_xd.contiguous()
        grid_yd = grid_yd.contiguous()
        grid_xd = grid_xd.view(-1).unsqueeze(1).unsqueeze(1)
        grid_yd = grid_yd.view(-1).unsqueeze(1).unsqueeze(1)
        grid = torch.cat([grid_xd, grid_yd], dim=2).numpy()

        # undistort points
        xy_undistorted = cv2.undistortPoints(grid, self.cam_mtx, self.cam_dist, P=newcameramtx)
        self.grid_xu = torch.from_numpy(xy_undistorted[:, 0, 0].astype(np.float32)).view(height, width).to(self.device)
        self.grid_yu = torch.from_numpy(xy_undistorted[:, 0, 1].astype(np.float32)).view(height, width).to(self.device)

        # update camera matrix
        self.cam_mtx = newcameramtx

    def undistort_events(self, xs, ys):
        """
        Undistorts event list using look-up table.
        :param xs: [N] tensor with distorted event x location
        :param ys: [N] tensor with distorted event y location
        :return xs: [N] tensor with undistorted event x location
        :return ys: [N] tensor with undistorted event y location
        """

        # undistortion look-up table
        grid_idx = ys.clone()
        grid_idx *= self.res[1]  # torch.view is row-major
        grid_idx += xs.clone()
        xs = torch.gather(self.grid_xu.view(-1), 0, grid_idx.long())
        ys = torch.gather(self.grid_yu.view(-1), 0, grid_idx.long())

        return xs, ys
