import os
import sys

import numpy as np
import torch

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)

from utils.iwe import get_interpolation, interpolate
from utils.utils import binary_search_array


class EventWarping(torch.nn.Module):
    def __init__(self, config, device, loss_scaling=True):
        super(EventWarping, self).__init__()
        self.device = device
        self.config = config
        self.loss_scaling = loss_scaling
        self.res = config["loader"]["resolution"]
        self.batch_size = config["loader"]["batch_size"]
        self.flow_temp_reg = config["loss"]["flow_temp_reg"]

        self._passes = 0
        self._event_list = None
        self._flow_list = None
        self._pol_mask_list = None
        self._vector_list = None

    def update_flow_loss(self, vector_list, flow_list, event_list, pol_mask):
        """
        :param vector_list: [batch_size x 8 x 1 x 1] optical flow vectors
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        :param event_list: [batch_size x N x 4] input events (ts, y, x, p)
        :param pol_mask: [batch_size x N x 2] polarity mask (pos, neg)
        """

        # flow vector per input event
        flow_idx = event_list[:, :, 1:3].clone()
        flow_idx = flow_idx.round()  # events are associated to the closest flow vector
        flow_idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        # get flow for every event in the list
        if self._flow_list is None:
            self._flow_list = []

        N = event_list.shape[1]  # number of events
        for i in range(len(flow_list)):
            flow = flow_list[i].clone()
            flow = flow.view(self.batch_size, 2, -1)
            event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
            event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
            event_flowy = event_flowy.view(self.batch_size, N, 1)
            event_flowx = event_flowx.view(self.batch_size, N, 1)
            event_flow = torch.cat([event_flowy, event_flowx], dim=2)

            if i == len(self._flow_list):
                self._flow_list.append(event_flow)
            else:
                self._flow_list[i] = torch.cat([self._flow_list[i], event_flow], dim=1)

        # update vector list
        vector_list = vector_list.view(self.batch_size, 1, 8)
        if self._vector_list is None:
            self._vector_list = vector_list
        else:
            self._vector_list = torch.cat([self._vector_list, vector_list], dim=1)

        # update internal event list
        if self._event_list is None:
            self._event_list = event_list
        else:
            event_list[:, :, 0:1] += self._passes  # only nonzero second time
            self._event_list = torch.cat([self._event_list, event_list], dim=1)

        # update internal polarity mask list
        if self._pol_mask_list is None:
            self._pol_mask_list = pol_mask
        else:
            self._pol_mask_list = torch.cat([self._pol_mask_list, pol_mask], dim=1)

        # update timestamp index
        self._passes += 1

    def reset(self):
        self._passes = 0
        self._event_list = None
        self._flow_list = None
        self._pol_mask_list = None
        self._vector_list = None

    @property
    def num_passes(self):
        return self._passes

    @property
    def num_events(self):
        if self._event_list is None:
            return 0
        else:
            return self._event_list.shape[1]

    def event_deblurring(self, max_ts, event_list, flow, pol_mask, ts_list, interp_zeros=None, iwe_zeros=None):
        """
        Main component of the contrast maximization loss. See 'Self-Supervised Learning of
        Event-Based Optical Flow with Spiking Neural Networks', Paredes-Valles and Hagenaars et al., arXiv 2021

        :param max_ts: most recent timestamp of the input events
        :param event_list: [batch_size x N x 4] input events (ts, y, x, p)
        :param flow: [batch_size x 2 x H x W] optical flow (x, y) map
        :param pol_mask: [batch_size x N x 2] polarity mask (pos, neg)
        :param ts_list: [batch_size x N x 1] event timestamp [0, max_ts]
        """

        # interpolate forward
        tref = max_ts
        fw_idx, fw_weights = get_interpolation(event_list, flow, tref, self.res, zeros=interp_zeros)

        # per-polarity image of (forward) warped events
        fw_iwe_pos = interpolate(fw_idx, fw_weights, self.res, polarity_mask=pol_mask[:, :, 0:1], zeros=iwe_zeros)
        fw_iwe_neg = interpolate(fw_idx, fw_weights, self.res, polarity_mask=pol_mask[:, :, 1:2], zeros=iwe_zeros)
        fw_iwe = torch.cat([fw_iwe_pos, fw_iwe_neg], dim=1)

        # image of (forward) warped averaged timestamps
        fw_iwe_pos_ts = interpolate(
            fw_idx, fw_weights * ts_list, self.res, polarity_mask=pol_mask[:, :, 0:1], zeros=iwe_zeros
        )
        fw_iwe_neg_ts = interpolate(
            fw_idx, fw_weights * ts_list, self.res, polarity_mask=pol_mask[:, :, 1:2], zeros=iwe_zeros
        )
        fw_iwe_pos_ts /= fw_iwe_pos + 1e-9
        fw_iwe_neg_ts /= fw_iwe_neg + 1e-9
        fw_iwe_pos_ts = fw_iwe_pos_ts / max_ts
        fw_iwe_neg_ts = fw_iwe_neg_ts / max_ts
        fw_iwe_ts = torch.cat([fw_iwe_pos_ts, fw_iwe_neg_ts], dim=1)

        # loss scaling
        loss = self.scale_loss(fw_iwe, fw_iwe_ts)

        # interpolate backward
        tref = 0
        bw_idx, bw_weights = get_interpolation(event_list, flow, tref, self.res, zeros=interp_zeros)

        # per-polarity image of (backward) warped events
        bw_iwe_pos = interpolate(bw_idx, bw_weights, self.res, polarity_mask=pol_mask[:, :, 0:1], zeros=iwe_zeros)
        bw_iwe_neg = interpolate(bw_idx, bw_weights, self.res, polarity_mask=pol_mask[:, :, 1:2], zeros=iwe_zeros)
        bw_iwe = torch.cat([bw_iwe_pos, bw_iwe_neg], dim=1)

        # image of (backward) warped averaged timestamps
        bw_iwe_pos_ts = interpolate(
            bw_idx, bw_weights * (max_ts - ts_list), self.res, polarity_mask=pol_mask[:, :, 0:1], zeros=iwe_zeros
        )
        bw_iwe_neg_ts = interpolate(
            bw_idx, bw_weights * (max_ts - ts_list), self.res, polarity_mask=pol_mask[:, :, 1:2], zeros=iwe_zeros
        )
        bw_iwe_pos_ts /= bw_iwe_pos + 1e-9
        bw_iwe_neg_ts /= bw_iwe_neg + 1e-9
        bw_iwe_pos_ts = bw_iwe_pos_ts / max_ts
        bw_iwe_neg_ts = bw_iwe_neg_ts / max_ts
        bw_iwe_ts = torch.cat([bw_iwe_pos_ts, bw_iwe_neg_ts], dim=1)

        # loss scaling
        loss += self.scale_loss(bw_iwe, bw_iwe_ts)

        return loss

    def scale_loss(self, iwe, iwe_ts):
        """
        Scaling of the loss function based on the number of events in the image space. See "Self-Supervised Learning of
        Event-Based Optical Flow with Spiking Neural Networks", Paredes-Valles and Hagenaars et al., arXiv 2021
        """

        iwe_ts = iwe_ts.view(iwe_ts.shape[0], 2, -1)
        loss = torch.sum(iwe_ts[:, 0, :] ** 2, dim=1) + torch.sum(iwe_ts[:, 1, :] ** 2, dim=1)
        if self.loss_scaling:
            nonzero_px = torch.sum(iwe, dim=1, keepdim=True).bool()
            nonzero_px = nonzero_px.view(nonzero_px.shape[0], -1)
            loss /= torch.sum(nonzero_px, dim=1) + 1e-9

        return torch.sum(loss)

    def flow_smoothing(self):
        """
        Scaled Charbonnier smoothness prior on the estimated optical flow vectors.
        """

        # flow smoothing (forward differences)
        flow_dt = self._vector_list[:, :-1, :] - self._vector_list[:, 1:, :]

        # charbonnier smoothness prior
        flow_dt = torch.sqrt(flow_dt**2 + 1e-9)
        flow_dt = flow_dt.view(self.batch_size, -1)
        if flow_dt.shape[1] > 0:
            flow_dt = torch.mean(flow_dt, dim=1)

        # compute loss
        loss = 0
        if self._vector_list.shape[1] > 0:
            loss = self.flow_temp_reg * flow_dt
            loss = loss.sum()

        return loss

    def forward(self):
        max_ts = self._passes

        # split input
        pol_mask = torch.cat([self._pol_mask_list for i in range(4)], dim=1)
        ts_list = torch.cat([self._event_list[:, :, 0:1] for i in range(4)], dim=1)

        loss = 0
        iwe_zeros = torch.zeros((ts_list.shape[0], self.res[0] * self.res[1], 1)).to(ts_list.device)
        interp_zeros = torch.zeros((ts_list.shape[0], ts_list.shape[1], 2)).to(ts_list.device)

        for i in range(len(self._flow_list)):
            # event deblurring
            loss += self.event_deblurring(
                max_ts,
                self._event_list,
                self._flow_list[i],
                pol_mask,
                ts_list,
                interp_zeros=interp_zeros,
                iwe_zeros=iwe_zeros,
            )

        # average loss over all flow predictions
        loss /= len(self._flow_list)

        # temporal smoothing of predicted flow vectors
        loss += self.flow_smoothing()

        return loss


class BaseValidationLoss(torch.nn.Module):
    """
    Base class for validation metrics.
    """

    def __init__(self, config, device):
        super(BaseValidationLoss, self).__init__()
        self.res = config["loader"]["resolution"]
        self.device = device
        self.config = config

        self._passes = 0
        self._event_list = None
        self._flow_list = None
        self._flow_map = None
        self._pol_mask_list = None
        self._event_mask = None

    @property
    def num_passes(self):
        return self._passes

    @property
    def num_events(self):
        if self._event_list is None:
            return 0
        else:
            return self._event_list.shape[1]

    def event_flow_association(self, flow_list, event_list, pol_mask, event_mask):
        """
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        :param inputs: dataloader dictionary
        """

        # flow vector per input event
        flow_idx = event_list[:, :, 1:3].clone()
        flow_idx = flow_idx.round()  # events are associated to the closest flow vector
        flow_idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        # get flow for every event in the list
        flow = flow_list[-1]  # only highest resolution flow
        flow = flow.view(flow.shape[0], 2, -1)
        event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
        event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
        event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
        event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
        event_flow = torch.cat([event_flowy, event_flowx], dim=2)

        if self._flow_list is None:
            self._flow_list = event_flow
        else:
            self._flow_list = torch.cat([self._flow_list, event_flow], dim=1)

        # update internal event list
        if self._event_list is None:
            self._event_list = event_list
        else:
            event_list = event_list.clone()  # to prevent issues with other metrics
            event_list[:, :, 0:1] += self._passes  # only nonzero second time
            self._event_list = torch.cat([self._event_list, event_list], dim=1)

        # update internal polarity mask list
        if self._pol_mask_list is None:
            self._pol_mask_list = pol_mask
        else:
            self._pol_mask_list = torch.cat([self._pol_mask_list, pol_mask], dim=1)

        # update flow map
        if self._flow_map is None:
            self._flow_map = []
        self._flow_map.append(flow.view(flow.shape[0], 2, self.res[0], self.res[1]))

        # event mask
        if self._event_mask is None:
            self._event_mask = event_mask
        else:
            self._event_mask = torch.cat([self._event_mask, event_mask], dim=1)

        # update timestamp index
        self._passes += 1

    def reset(self):
        self._passes = 0
        self._event_list = None
        self._flow_list = None
        self._flow_map = None
        self._pol_mask_list = None
        self._event_mask = None

    def compute_window_events(self, round_idx=False):
        max_ts = self._passes
        pol_mask_list = self._pol_mask_list
        if not round_idx:
            pol_mask_list = torch.cat([pol_mask_list for i in range(4)], dim=1)

        fw_idx, fw_weights = get_interpolation(
            self._event_list, self._flow_list * 0, max_ts, self.res, round_idx=round_idx
        )
        events_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask_list[:, :, 0:1])
        events_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask_list[:, :, 1:2])

        return torch.cat([events_pos, events_neg], dim=1)

    def compute_masked_window_flow(self):
        avg_flow = self._flow_map[0] * self._event_mask[:, 0:1, :, :]
        for i in range(1, self._event_mask.shape[1]):
            avg_flow += self._flow_map[i] * self._event_mask[:, i : i + 1, :, :]
        avg_flow /= torch.sum(self._event_mask, dim=1, keepdim=True) + 1e-9
        return avg_flow

    def compute_window_iwe(self, round_idx=False):
        max_ts = self._passes
        pol_mask_list = self._pol_mask_list
        if not round_idx:
            pol_mask_list = torch.cat([pol_mask_list for i in range(4)], dim=1)

        fw_idx, fw_weights = get_interpolation(self._event_list, self._flow_list, max_ts, self.res, round_idx=round_idx)
        fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask_list[:, :, 0:1])
        fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask_list[:, :, 1:2])

        return torch.cat([fw_iwe_pos, fw_iwe_neg], dim=1)


class EvalCriteria(BaseValidationLoss):
    """
    Similarly to the FWL metric, the Ratio of the Squared Averaged Timestamps (RSAT) metric is the ratio of the squared sum
    of the per-pixel and per-polarity average timestamp of the image of warped events and that of
    the image of (non-warped) events; hence, the lower the value of this metric, the better the optical flow estimate.
    Note that this metric is sensitive to the number of input events.
    """

    def __init__(self, config, device):
        super().__init__(config, device)
        self.gt_pred_ts = None
        self.gt_target_ts = None
        self.gt_pred_vectors = None
        self.gt_target_vectors = None

    def reset_gt(self):
        self.gt_pred_ts = None
        self.gt_target_ts = None
        self.gt_pred_vectors = None
        self.gt_target_vectors = None

    def reset_rsat(self):
        self.reset()

    def update_gt(self, pred_ts, pred_vectors, target_vectors_list):
        # predicted vectors
        pred_ts = torch.Tensor(np.asarray(pred_ts, dtype=np.float32)).to(self.device).view(1, -1)
        pred_vectors = pred_vectors.clone().view(1, -1)

        if self.gt_pred_ts is None:
            self.gt_pred_ts = pred_ts
        else:
            self.gt_pred_ts = torch.cat([self.gt_pred_ts, pred_ts], dim=0)

        if self.gt_pred_vectors is None:
            self.gt_pred_vectors = pred_vectors
        else:
            self.gt_pred_vectors = torch.cat([self.gt_pred_vectors, pred_vectors], dim=0)

        # gt vectors
        for entry in target_vectors_list:
            target_ts = entry["ts"].view(1, -1)
            target_vectors = entry["flow_corners"].view(1, -1)

            if self.gt_target_ts is None:
                self.gt_target_ts = target_ts
            else:
                self.gt_target_ts = torch.cat([self.gt_target_ts, target_ts], dim=0)

            if self.gt_target_vectors is None:
                self.gt_target_vectors = target_vectors
            else:
                self.gt_target_vectors = torch.cat([self.gt_target_vectors, target_vectors], dim=0)

    @property
    def gt(self):
        return {
            "gt_pred_ts": self.gt_pred_ts,
            "gt_pred_vectors": self.gt_pred_vectors,
            "gt_target_ts": self.gt_target_ts,
            "gt_target_vectors": self.gt_target_vectors,
        }

    @staticmethod
    def aee(data):
        """
        Average endpoint error (which is just the Euclidean distance) loss.
        """

        its = 0
        error = 0
        for i in range(data["gt_pred_ts"].shape[0]):
            index = binary_search_array(data["gt_target_ts"][:, 0], data["gt_pred_ts"][i, 0], side="right")

            if index > 0 and index < data["gt_target_ts"].shape[0] - 1:
                curr_ts = data["gt_pred_ts"][i, 0]
                prev_ts = data["gt_target_ts"][index, 0]
                next_ts = data["gt_target_ts"][index + 1, 0]
                prev_gt = data["gt_target_vectors"][index, :]
                next_gt = data["gt_target_vectors"][index + 1, :]
                gt_interp = (prev_gt * (next_ts - curr_ts) + next_gt * (curr_ts - prev_ts)) / (next_ts - prev_ts)

                error += (data["gt_pred_vectors"][i, :].view(4, 2) - gt_interp.view(4, 2)).pow(2).sum(1).sqrt().mean(0)
                its += 1

        return error / its

    def rsat(self):
        """
        Similarly to the FWL metric, the Ratio of the Squared Averaged Timestamps (RSAT) metric is the ratio of the squared sum
        of the per-pixel and per-polarity average timestamp of the image of warped events and that of
        the image of (non-warped) events; hence, the lower the value of this metric, the better the optical flow estimate.
        Note that this metric is sensitive to the number of input events.
        """

        max_ts = self._passes

        # image of (forward) warped averaged timestamps
        ts_list = self._event_list[:, :, 0:1]
        fw_idx, fw_weights = get_interpolation(self._event_list, self._flow_list, max_ts, self.res, round_idx=True)
        fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1])
        fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2])
        fw_iwe_pos_ts = interpolate(
            fw_idx.long(), fw_weights * ts_list, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1]
        )
        fw_iwe_neg_ts = interpolate(
            fw_idx.long(), fw_weights * ts_list, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2]
        )
        fw_iwe_pos_ts /= fw_iwe_pos + 1e-9
        fw_iwe_neg_ts /= fw_iwe_neg + 1e-9
        fw_iwe_pos_ts = fw_iwe_pos_ts / max_ts
        fw_iwe_neg_ts = fw_iwe_neg_ts / max_ts

        # image of non-warped averaged timestamps
        zero_idx, zero_weights = get_interpolation(
            self._event_list, self._flow_list * 0, max_ts, self.res, round_idx=True
        )
        zero_iwe_pos = interpolate(
            zero_idx.long(), zero_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1]
        )
        zero_iwe_neg = interpolate(
            zero_idx.long(), zero_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2]
        )
        zero_iwe_pos_ts = interpolate(
            zero_idx.long(), zero_weights * ts_list, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1]
        )
        zero_iwe_neg_ts = interpolate(
            zero_idx.long(), zero_weights * ts_list, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2]
        )
        zero_iwe_pos_ts /= zero_iwe_pos + 1e-9
        zero_iwe_neg_ts /= zero_iwe_neg + 1e-9
        zero_iwe_pos_ts = zero_iwe_pos_ts / max_ts
        zero_iwe_neg_ts = zero_iwe_neg_ts / max_ts

        # (scaled) sum of the squares of the per-pixel and per-polarity average timestamps
        fw_iwe_pos_ts = fw_iwe_pos_ts.view(fw_iwe_pos_ts.shape[0], -1)
        fw_iwe_neg_ts = fw_iwe_neg_ts.view(fw_iwe_neg_ts.shape[0], -1)
        fw_iwe_pos_ts = torch.sum(fw_iwe_pos_ts**2, dim=1)
        fw_iwe_neg_ts = torch.sum(fw_iwe_neg_ts**2, dim=1)
        fw_ts_sum = fw_iwe_pos_ts + fw_iwe_neg_ts

        fw_nonzero_px = fw_iwe_pos + fw_iwe_neg
        fw_nonzero_px[fw_nonzero_px > 0] = 1
        fw_nonzero_px = fw_nonzero_px.view(fw_nonzero_px.shape[0], -1)
        fw_ts_sum /= torch.sum(fw_nonzero_px, dim=1)

        zero_iwe_pos_ts = zero_iwe_pos_ts.view(zero_iwe_pos_ts.shape[0], -1)
        zero_iwe_neg_ts = zero_iwe_neg_ts.view(zero_iwe_neg_ts.shape[0], -1)
        zero_iwe_pos_ts = torch.sum(zero_iwe_pos_ts**2, dim=1)
        zero_iwe_neg_ts = torch.sum(zero_iwe_neg_ts**2, dim=1)
        zero_ts_sum = zero_iwe_pos_ts + zero_iwe_neg_ts

        zero_nonzero_px = zero_iwe_pos + zero_iwe_neg
        zero_nonzero_px[zero_nonzero_px > 0] = 1
        zero_nonzero_px = zero_nonzero_px.view(zero_nonzero_px.shape[0], -1)
        zero_ts_sum /= torch.sum(zero_nonzero_px, dim=1)

        return fw_ts_sum / zero_ts_sum
