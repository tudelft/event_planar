import cv2
import numpy as np
import torch


class HomographyWarping:
    """
    Retrieve "dense" optical flow map from the predicted four optical flow vectors.
    """

    def __init__(self, config, flow_scaling=180, K=None):
        self.batch_size = config["loader"]["batch_size"]
        self.device = config["loader"]["device"]
        self.res = config["loader"]["resolution"]
        self.flow_scaling = flow_scaling

        # point initialization
        left_top = (0, 0)
        right_top = (self.res[1], 0)
        right_bot = (self.res[1], self.res[0])
        left_bot = (0, self.res[0])
        base_points = np.asarray([left_top, right_top, right_bot, left_bot])
        base_points = np.repeat(base_points[np.newaxis, :, :], self.batch_size, axis=0)
        self.base_points = torch.from_numpy(base_points.astype(np.float32)).to(self.device)

        # pixel coordinates
        x = torch.linspace(0, self.res[1] - 1, steps=self.res[1])
        y = torch.linspace(0, self.res[0] - 1, steps=self.res[0])
        grid_y, grid_x = torch.meshgrid(y, x)
        grid_x = grid_x.contiguous()
        grid_y = grid_y.contiguous()

        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        self.grid = grid.repeat(self.batch_size, 1, 1, 1).to(self.device)

        ones = torch.ones(self.batch_size, 1, self.res[0] * self.res[1]).to(self.device)
        self.pix_coords = torch.cat([self.grid.view(self.batch_size, 2, -1), ones], dim=1)

        self.ones = torch.ones(self.batch_size, 4, 1).to(self.device)
        self.zeros = torch.zeros(self.batch_size, 4, 3).to(self.device)

        # comparison against ground truth
        self.prev_ts = None
        self.prev_c2Rw = None
        self.prev_t_w = None

        self.R_zeros = torch.zeros((self.batch_size, 3, 3)).to(self.device)
        self.n_w = torch.zeros((self.batch_size, 3, 1)).to(self.device)
        self.n_w[:, 2, 0] = 1

        self.K = K
        if self.K is not None:
            self.K = torch.from_numpy(self.K.astype(np.float32)).to(self.device).unsqueeze(0)

        self.corners = self.base_points.clone().permute(0, 2, 1)
        self.corners = torch.cat([self.corners, torch.ones((self.batch_size, 1, 4)).to(self.device)], dim=1)

    @classmethod
    def secondary_init(cls, device="cpu", res=(180, 180), flow_scaling=1):
        config = {}
        config["loader"] = {}
        config["loader"]["batch_size"] = 1
        config["loader"]["device"] = device
        config["loader"]["resolution"] = res
        return cls(config, flow_scaling=flow_scaling)

    # adapted from https://github.com/JirongZhang/DeepHomography
    def DLT_solve(self, flow_vectors):
        flow_vectors = flow_vectors.view(flow_vectors.shape[0], 4, 2)
        new_points = self.base_points + flow_vectors * self.flow_scaling

        xy1 = torch.cat((self.base_points, self.ones), dim=2)
        xyu = torch.cat((xy1, self.zeros), dim=2)
        xyd = torch.cat((self.zeros, xy1), 2)

        M1 = torch.cat((xyu, xyd), dim=2).view(self.batch_size, -1, 6)
        M2 = torch.matmul(new_points.view(-1, 2, 1), self.base_points.view(-1, 1, 2)).view(self.batch_size, -1, 2)

        A = torch.cat((M1, -M2), dim=2)
        b = new_points.view(self.batch_size, -1, 1)
        Ainv = torch.inverse(A.cpu()).to(self.device)

        h8 = torch.matmul(Ainv, b).view(self.batch_size, 8)
        H = torch.cat((h8, self.ones[:, 0, :]), 1).view(self.batch_size, 3, 3)

        return H

    def flow_from_homography(self, H_mat):
        # compute forward displacement map from homography matrix
        warped_pix_coords = torch.matmul(H_mat, self.pix_coords)
        warped_pix_coords = warped_pix_coords[:, :2, :] / (warped_pix_coords[:, 2:3, :] + 1e-9)
        warped_pix_coords = warped_pix_coords.view(self.batch_size, 2, self.res[0], self.res[1])

        # compute flow map
        flow = warped_pix_coords - self.grid

        return [flow]

    def get_flow_map(self, flow_vectors):
        """
        :param flow_vectors: [batch_size x num_output_channels X 1 X 1] optical flow vectors
        :param event_list: [batch_size x N x 4] input events (ts, y, x, p)
        :param pol_mask: [batch_size x N x 2] polarity mask (pos, neg)
        :param event_mask: [batch_size x 1 x H x W] event mask
        """

        # compute homography matrix
        H_mat = self.DLT_solve(flow_vectors)
        self.H_mat = H_mat.clone()

        # compute flow from the homography matrix
        flow_list = self.flow_from_homography(H_mat)

        return flow_list

    def reset_pose_gt(self):
        """
        Reset the ground truth pose
        """

        self.prev_ts = None
        self.prev_c2Rw = None
        self.prev_t_w = None

    def pose_to_4ptflow(self, pos_w, euler_c):
        """
        Convert pose to 4-point flow.
        :param pos_w: position in world frame [batch_size x msgs x 4] with (ts, x, y, z)
        :param euler_c: attitude of the camera/imu [batch_size x msgs x 4] with (ts, phi, theta, psi)
        """

        assert self.K is not None, "K is required for comparison against GT"
        assert pos_w.shape[1] == euler_c.shape[1], "Inconsistent number of GT messages"

        # loop through all GT messages in the input window
        output = []
        for i in range(pos_w.shape[1]):
            assert torch.isclose(pos_w[:, i, 0], euler_c[:, i, 0]), "Inconsistent timestamp in GT messages"

            # pose
            ts = pos_w[:, i, 0]
            t_w = pos_w[:, i, 1:].view(self.batch_size, 3, 1)
            c2Rw = self.euler_to_rot(euler_c[:, i, 1:2], euler_c[:, i, 2:3], euler_c[:, i, 3:4])

            if self.prev_ts is not None:
                dt = ts - self.prev_ts

                # rotation matrix from prev. to curr. cam. frames
                c2Rc1 = c2Rw @ self.prev_c2Rw.transpose(1, 2)

                # relative translation from curr. to prev. cam. frames expressed in the curr. cam. frame scaled by the distance to the ground of the prev. cam. frame
                t_rel_c2 = c2Rw @ (self.prev_t_w - t_w)
                t_scaled_rel_c = t_rel_c2 / torch.abs(self.prev_t_w[:, 2, :])

                # euclidean homography matrix
                H = c2Rc1 + t_scaled_rel_c @ (self.prev_c2Rw @ self.n_w).transpose(1, 2)
                H = H / H[:, 2, 2]

                # corrected homography matrix
                Hcorr = self.K @ H @ self.K.inverse()

                # corner flow from homography matrix
                new_corners = Hcorr @ self.corners
                new_corners = new_corners[:, :2, :] / (new_corners[:, 2:3, :] + 1e-9)
                flow_corners = new_corners - self.corners[:, :2, :]  # px/dt
                flow_corners = flow_corners / (dt / 0.001)  # px/ms
                flow_corners = flow_corners.permute(0, 2, 1).unsqueeze(3).contiguous()
                flow_corners = flow_corners.view(self.batch_size, 8, 1, 1)

                gt = {}
                gt["ts"] = ts
                gt["flow_corners"] = flow_corners
                output.append(gt)

            # update for next iteration
            self.prev_ts = ts
            self.prev_c2Rw = c2Rw
            self.prev_t_w = t_w

        return output

    def euler_to_rot(self, roll, pitch, yaw):
        """
        :param roll: [batch_size x 1] roll angle
        :param pitch: [batch_size x 1] roll angle
        :param yaw: [batch_size x 1] roll angle
        :param roll: [batch_size x 1] roll angle
        :return R: [batch_size x 3 x 3] rotation matrix
        """

        R = self.R_zeros.clone()
        R[:, 0, 0] = torch.cos(yaw) * torch.cos(pitch)
        R[:, 0, 1] = torch.sin(yaw) * torch.cos(pitch)
        R[:, 0, 2] = -torch.sin(pitch)
        R[:, 1, 0] = -torch.sin(yaw) * torch.cos(roll) + torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll)
        R[:, 1, 1] = torch.cos(yaw) * torch.cos(roll) + torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll)
        R[:, 1, 2] = torch.cos(pitch) * torch.sin(roll)
        R[:, 2, 0] = torch.sin(yaw) * torch.sin(roll) + torch.cos(yaw) * torch.sin(pitch) * torch.cos(roll)
        R[:, 2, 1] = -torch.cos(yaw) * torch.sin(roll) + torch.sin(yaw) * torch.sin(pitch) * torch.cos(roll)
        R[:, 2, 2] = torch.cos(pitch) * torch.cos(roll)

        return R
