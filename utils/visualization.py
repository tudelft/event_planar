import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


class Visualization:
    """
    Utility class for the visualization and storage of rendered image-like representation
    of multiple elements of the optical flow estimation and image reconstruction pipeline.
    """

    def __init__(self, kwargs, eval_id=-1, path_results=None):
        self.img_idx = 0
        self.base_points = None
        self.vision_spike_rate = None
        self.control_spike_rate = None
        self.ion = kwargs["vis"]["enabled"]
        self.downsample = kwargs["data"]["downsample"]
        self.px = kwargs["vis"]["px"]
        self.res = kwargs["loader"]["resolution"]
        self.color_scheme = "green_red"  # gray / blue_red / green_red

        if eval_id >= 0 and path_results is not None:
            self.store_dir = path_results + "results/"
            self.store_dir = self.store_dir + "eval_" + str(eval_id) + "/"
            if not os.path.exists(self.store_dir):
                os.makedirs(self.store_dir)
            self.store_file = None

    def update(
        self, inputs, flow, iwe=None, events_window=None, masked_window_flow=None, iwe_window=None, flow_vectors=None
    ):
        """
        Live visualization.
        :param inputs: dataloader dictionary
        :param flow: [batch_size x 2 x H x W] optical flow map
        :param iwe: [batch_size x 1 x H x W] image of warped events
        """

        events = inputs["event_cnt"] if "event_cnt" in inputs.keys() else None
        gtflow = inputs["gtflow"] if "gtflow" in inputs.keys() else None

        if events is None:
            events = inputs["net_input"] if "net_input" in inputs.keys() else None
        height = events.shape[2]
        width = events.shape[3]

        # input events
        events = events.detach()
        events = events.cpu().numpy().transpose(0, 2, 3, 1)

        if flow_vectors is not None:
            flow_vectors = flow_vectors.view(4, 2).cpu().numpy()

        for i in range(events.shape[0]):
            events_local = events[i : i + 1, ...].reshape((height, width, -1))
            events_local = self.events_to_image(events_local)

            if flow_vectors is not None:
                if events.shape[0] == 1:
                    if self.base_points is None:
                        left_top = (width * 0.25, height * 0.25)
                        right_top = (width * 0.75, height * 0.25)
                        right_bot = (width * 0.75, height * 0.75)
                        left_bot = (width * 0.25, height * 0.75)
                        self.base_points = np.asarray([left_top, right_top, right_bot, left_bot])

                    new_points = self.base_points + flow_vectors * 25  # rough approximation of the estimated motion
                    new_points = new_points.astype(int)
                    base_points = self.base_points.astype(int)

                    for i in range(4):
                        i1 = i + 1 if i < 3 else 0
                        cv2.line(
                            events_local,
                            (new_points[i, 0], new_points[i, 1]),
                            (new_points[i1, 0], new_points[i1, 1]),
                            (255, 255, 255),
                        )
                        cv2.line(
                            events_local,
                            (base_points[i, 0], base_points[i, 1]),
                            (base_points[i1, 0], base_points[i1, 1]),
                            (255, 255, 255),
                        )

                else:
                    if self.base_points is None:
                        self.base_points = self.base_points = np.array([width * 0.5, height * 0.5])

                    new_points = (
                        self.base_points + flow_vectors[i, :] * 25
                    )  # rough approximation of the estimated motion
                    new_points = new_points.astype(int)
                    base_points = self.base_points.astype(int)

                    cv2.line(
                        events_local,
                        (new_points[0], new_points[1]),
                        (base_points[0], base_points[1]),
                        (255, 255, 255),
                    )

            cv2.namedWindow("Input Events " + str(i), cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Input Events " + str(i), int(self.px), int(self.px))
            cv2.imshow("Input Events " + str(i), events_local)

        # input events
        if events_window is not None:
            events_window = events_window.detach()
            events_window_npy = (
                events_window.cpu()
                .numpy()
                .transpose(0, 2, 3, 1)
                .reshape((events_window.shape[2], events_window.shape[3], -1))
            )
            cv2.namedWindow("Input Events - Eval window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Input Events - Eval window", int(self.px), int(self.px))
            cv2.imshow("Input Events - Eval window", self.events_to_image(events_window_npy))

        # optical flow
        if flow is not None:
            flow = flow.detach()
            flow_npy = flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((flow.shape[2], flow.shape[3], 2))
            flow_npy = self.flow_to_image(flow_npy[:, :, 0], flow_npy[:, :, 1])
            flow_npy = cv2.cvtColor(flow_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Estimated Flow", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Estimated Flow", int(self.px), int(self.px))
            cv2.imshow("Estimated Flow", flow_npy)

        # optical flow
        if masked_window_flow is not None:
            masked_window_flow = masked_window_flow.detach()
            masked_window_flow_npy = (
                masked_window_flow.cpu()
                .numpy()
                .transpose(0, 2, 3, 1)
                .reshape((masked_window_flow.shape[2], masked_window_flow.shape[3], 2))
            )
            masked_window_flow_npy = self.flow_to_image(
                masked_window_flow_npy[:, :, 0], masked_window_flow_npy[:, :, 1]
            )
            masked_window_flow_npy = cv2.cvtColor(masked_window_flow_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Estimated Flow - Eval window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Estimated Flow - Eval window", int(self.px), int(self.px))
            cv2.imshow("Estimated Flow - Eval window", masked_window_flow_npy)

        # ground-truth optical flow
        if gtflow is not None:
            gtflow = gtflow.detach()
            gtflow_npy = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((gtflow.shape[2], gtflow.shape[3], 2))
            gtflow_npy = self.flow_to_image(gtflow_npy[:, :, 0], gtflow_npy[:, :, 1])
            gtflow_npy = cv2.cvtColor(gtflow_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Ground-truth Flow", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Ground-truth Flow", int(self.px), int(self.px))
            cv2.imshow("Ground-truth Flow", gtflow_npy)

        # image of warped events
        if iwe is not None:
            iwe = iwe.detach()
            iwe_npy = iwe.cpu().numpy().transpose(0, 2, 3, 1).reshape((iwe.shape[2], iwe.shape[3], 2))
            iwe_npy = self.events_to_image(iwe_npy)
            cv2.namedWindow("Image of Warped Events", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image of Warped Events", int(self.px), int(self.px))
            cv2.imshow("Image of Warped Events", iwe_npy)

        # image of warped events - evaluation window
        if iwe_window is not None:
            iwe_window = iwe_window.detach()
            iwe_window_npy = (
                iwe_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((iwe_window.shape[2], iwe_window.shape[3], 2))
            )
            iwe_window_npy = self.events_to_image(iwe_window_npy)
            cv2.namedWindow("Image of Warped Events - Eval window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image of Warped Events - Eval window", int(self.px), int(self.px))
            cv2.imshow("Image of Warped Events - Eval window", iwe_window_npy)

        cv2.waitKey(1)

    def store_ground_truth(self, gt, sequence):
        path_to = self.store_dir + sequence + "/"
        if not os.path.exists(path_to):
            os.makedirs(path_to)

        _, axs = plt.subplots(2)
        gt["gt_pred_ts"] = gt["gt_pred_ts"].cpu().numpy()
        gt["gt_target_ts"] = gt["gt_target_ts"].cpu().numpy()
        gt["gt_pred_vectors"] = gt["gt_pred_vectors"].cpu().numpy()
        gt["gt_target_vectors"] = gt["gt_target_vectors"].cpu().numpy()

        axs[0].set(ylabel="Target flow [px/ms]")
        axs[1].set(ylabel="Pred. flow [px/ms]")
        axs[1].set(xlabel="Time [s]")

        for i in range(gt["gt_pred_vectors"].shape[1]):
            axs[0].plot(gt["gt_target_ts"][:, 0], gt["gt_target_vectors"][:, i], label=str(i))
            axs[1].plot(gt["gt_pred_ts"][:, 0], gt["gt_pred_vectors"][:, i], label=str(i))

        axs[1].set(xlim=axs[0].get_xlim(), ylim=axs[0].get_ylim())

        plt.tight_layout()
        plt.savefig(path_to + "gt_vectors.png")
        plt.close("all")

    def store_control_commands(self, x, ts, sequence):
        path_to = self.store_dir + sequence + "/"
        if not os.path.exists(path_to):
            os.makedirs(path_to)

        _, axs = plt.subplots(4)
        x = x.cpu().numpy()

        axs[0].plot(ts, x[:, 0])
        axs[1].plot(ts, x[:, 1])
        axs[2].plot(ts, x[:, 2])
        axs[3].plot(ts, x[:, 3])
        axs[0].set(ylabel="Thrust")
        axs[1].set(ylabel="Pitch")
        axs[2].set(ylabel="Roll")
        axs[3].set(ylabel="Yaw")
        axs[3].set(xlabel="Time [s]")

        plt.tight_layout()
        plt.savefig(path_to + "control_output.png", dpi=800)
        plt.close("all")

    def store(
        self,
        inputs,
        flow,
        sequence,
        iwe=None,
        events_window=None,
        masked_window_flow=None,
        iwe_window=None,
        flow_vectors=None,
        vision_spikes=None,
        control_spikes=None,
        fig_vision=None,
        fig_control=None,
        ts=None,
    ):
        """
        Store rendered images.
        :param inputs: dataloader dictionary
        :param flow: [batch_size x 2 x H x W] optical flow map
        :param iwe: [batch_size x 1 x H x W] image of warped events
        :param sequence: filename of the event sequence under analysis
        :param ts: timestamp associated with rendered files (default = None)
        """

        gtflow = inputs["gtflow"] if "gtflow" in inputs.keys() else None
        events = inputs["event_cnt"] if "event_cnt" in inputs.keys() else None

        if events is None:
            events = inputs["net_input"] if "net_input" in inputs.keys() else None

        if events.shape[0] == 4:
            tmp_events = torch.zeros((1, 2, self.res[0] // self.downsample, self.res[1] // self.downsample)).to(
                events.device
            )
            tmp_events[0, :, : events.shape[2], : events.shape[3]] = events[0, ...]  # TL
            tmp_events[0, :, : events.shape[2], -events.shape[3] :] = events[1, ...]  # TR
            tmp_events[0, :, -events.shape[2] :, -events.shape[3] :] = events[2, ...]  # BR
            tmp_events[0, :, -events.shape[2] :, : events.shape[3]] = events[3, ...]  # BL
            events = tmp_events.clone()

        events = events.detach()
        events_npy = events.cpu().numpy().transpose(0, 2, 3, 1).reshape((events.shape[2], events.shape[3], -1))
        if events.shape[2] != self.res[0] or events.shape[3] != self.res[1]:
            events_npy = cv2.resize(events_npy, (int(self.res[1]), int(self.res[0])), interpolation=cv2.INTER_NEAREST)
        height = events_npy.shape[0]
        width = events_npy.shape[1]

        # check if new sequence
        path_to = self.store_dir + sequence + "/"
        if not os.path.exists(path_to):
            os.makedirs(path_to)
            os.makedirs(path_to + "events/")
            os.makedirs(path_to + "events_window/")
            os.makedirs(path_to + "flow/")
            os.makedirs(path_to + "flow_window/")
            os.makedirs(path_to + "gtflow/")
            os.makedirs(path_to + "iwe/")
            os.makedirs(path_to + "iwe_window/")
            os.makedirs(path_to + "vision_spikes/")
            os.makedirs(path_to + "vision_spike_rate/")
            os.makedirs(path_to + "control_spikes/")
            os.makedirs(path_to + "control_spike_rate/")
            os.makedirs(path_to + "fig_vision/")
            os.makedirs(path_to + "fig_control/")

            if self.store_file is not None:
                self.store_file.close()
            self.store_file = open(path_to + "timestamps.txt", "w")
            self.img_idx = 0

        # input events
        event_image = np.zeros((height, width))
        event_image = self.events_to_image(events_npy)

        if flow_vectors is not None:
            if self.base_points is None:
                left_top = (width * 0.25, height * 0.25)
                right_top = (width * 0.75, height * 0.25)
                right_bot = (width * 0.75, height * 0.75)
                left_bot = (width * 0.25, height * 0.75)
                self.base_points = np.asarray([left_top, right_top, right_bot, left_bot])

            flow_vectors = flow_vectors.view(4, 2).cpu().numpy()
            new_points = self.base_points + flow_vectors * 25  # rough approximation of the estimated motion
            new_points = new_points.astype(int)
            base_points = self.base_points.astype(int)

            for i in range(4):
                i1 = i + 1 if i < 3 else 0
                cv2.line(
                    event_image,
                    (new_points[i, 0], new_points[i, 1]),
                    (new_points[i1, 0], new_points[i1, 1]),
                    (255, 255, 255),
                )
                cv2.line(
                    event_image,
                    (base_points[i, 0], base_points[i, 1]),
                    (base_points[i1, 0], base_points[i1, 1]),
                    (255, 255, 255),
                )

        filename = path_to + "events/%09d.png" % self.img_idx
        cv2.imwrite(filename, event_image * 255)

        # input events
        if events_window is not None:
            events_window = events_window.detach()
            events_window_npy = events_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
            events_window_npy = self.events_to_image(events_window_npy)
            filename = path_to + "events_window/%09d.png" % self.img_idx
            cv2.imwrite(filename, events_window_npy * 255)

        # optical flow
        if flow is not None:
            flow = flow.detach()
            flow_npy = flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            flow_npy = self.flow_to_image(flow_npy[:, :, 0], flow_npy[:, :, 1])
            flow_npy = cv2.cvtColor(flow_npy, cv2.COLOR_RGB2BGR)
            filename = path_to + "flow/%09d.png" % self.img_idx
            cv2.imwrite(filename, flow_npy)

        # optical flow
        if masked_window_flow is not None:
            masked_window_flow = masked_window_flow.detach()
            masked_window_flow_npy = masked_window_flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            masked_window_flow_npy = self.flow_to_image(
                masked_window_flow_npy[:, :, 0], masked_window_flow_npy[:, :, 1]
            )
            masked_window_flow_npy = cv2.cvtColor(masked_window_flow_npy, cv2.COLOR_RGB2BGR)
            filename = path_to + "flow_window/%09d.png" % self.img_idx
            cv2.imwrite(filename, masked_window_flow_npy)

        # ground-truth optical flow
        if gtflow is not None:
            gtflow = gtflow.detach()
            gtflow_npy = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            gtflow_npy = self.flow_to_image(gtflow_npy[:, :, 0], gtflow_npy[:, :, 1])
            gtflow_npy = cv2.cvtColor(gtflow_npy, cv2.COLOR_RGB2BGR)
            filename = path_to + "gtflow/%09d.png" % self.img_idx
            cv2.imwrite(filename, gtflow_npy)

        # image of warped events
        if iwe is not None:
            iwe = iwe.detach()
            iwe_npy = iwe.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            iwe_npy = self.events_to_image(iwe_npy)
            filename = path_to + "iwe/%09d.png" % self.img_idx
            cv2.imwrite(filename, iwe_npy * 255)

        # image of warped events - evaluation window
        if iwe_window is not None:
            iwe_window = iwe_window.detach()
            iwe_window_npy = iwe_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            iwe_window_npy = self.events_to_image(iwe_window_npy)
            filename = path_to + "iwe_window/%09d.png" % self.img_idx
            cv2.imwrite(filename, iwe_window_npy * 255)

        # activity of the last (vision) spiking layer
        if vision_spikes is not None:
            vision_spikes = vision_spikes.detach().cpu()
            vision_spikes_npy = vision_spikes.numpy().transpose(0, 2, 3, 1)
            vision_spikes_npy = vision_spikes_npy.reshape((-1, vision_spikes_npy.shape[-1]))
            vision_spikes_npy = self.minmax_norm(vision_spikes_npy) * 255
            vision_spikes_npy = vision_spikes_npy.astype(np.uint8)
            vision_spikes_npy = cv2.applyColorMap(vision_spikes_npy, cv2.COLORMAP_HOT)
            vision_spikes_npy = cv2.resize(vision_spikes_npy, (width * 2, 50), interpolation=cv2.INTER_LINEAR)
            filename = path_to + "vision_spikes/%09d.png" % self.img_idx
            cv2.imwrite(filename, vision_spikes_npy)

            if self.vision_spike_rate is None:
                self.vision_spike_rate = torch.zeros(vision_spikes.shape)
            self.vision_spike_rate = self.vision_spike_rate * 0.95 + vision_spikes * 0.05

            vision_spikes_npy = self.vision_spike_rate.cpu().numpy().transpose(0, 2, 3, 1)
            vision_spikes_npy = vision_spikes_npy.reshape((-1, vision_spikes_npy.shape[-1]))
            vision_spikes_npy = self.minmax_norm(vision_spikes_npy) * 255
            vision_spikes_npy = vision_spikes_npy.astype(np.uint8)
            vision_spikes_npy = cv2.applyColorMap(vision_spikes_npy, cv2.COLORMAP_HOT)
            vision_spikes_npy = cv2.resize(vision_spikes_npy, (width * 2, 50), interpolation=cv2.INTER_LINEAR)
            filename = path_to + "vision_spike_rate/%09d.png" % self.img_idx
            cv2.imwrite(filename, vision_spikes_npy)

        # activity of the last (control) spiking layer
        if control_spikes is not None:
            control_spikes = control_spikes.detach().cpu()
            control_spikes_npy = control_spikes.numpy()
            control_spikes_npy = control_spikes_npy.reshape((-1, control_spikes_npy.shape[-1]))
            control_spikes_npy = self.minmax_norm(control_spikes_npy) * 255
            control_spikes_npy = control_spikes_npy.astype(np.uint8)
            control_spikes_npy = cv2.applyColorMap(control_spikes_npy, cv2.COLORMAP_HOT)
            control_spikes_npy = cv2.resize(control_spikes_npy, (width * 2, 50), interpolation=cv2.INTER_LINEAR)
            filename = path_to + "control_spikes/%09d.png" % self.img_idx
            cv2.imwrite(filename, control_spikes_npy)

            if self.control_spike_rate is None:
                self.control_spike_rate = torch.zeros(control_spikes.shape)
            self.control_spike_rate = self.control_spike_rate * 0.95 + control_spikes * 0.05

            control_spikes_npy = self.control_spike_rate.cpu().numpy()
            control_spikes_npy = control_spikes_npy.reshape((-1, control_spikes_npy.shape[-1]))
            control_spikes_npy = self.minmax_norm(control_spikes_npy) * 255
            control_spikes_npy = control_spikes_npy.astype(np.uint8)
            control_spikes_npy = cv2.applyColorMap(control_spikes_npy, cv2.COLORMAP_HOT)
            control_spikes_npy = cv2.resize(control_spikes_npy, (width * 2, 50), interpolation=cv2.INTER_LINEAR)
            filename = path_to + "control_spike_rate/%09d.png" % self.img_idx
            cv2.imwrite(filename, control_spikes_npy)

        # online vision figure of decoded values
        if fig_vision is not None:
            fig_vision.savefig(path_to + "fig_vision/%09d.png" % self.img_idx)

        # online control figure of decoded values
        if fig_control is not None:
            fig_control.savefig(path_to + "fig_control/%09d.png" % self.img_idx)

        # store timestamps
        if ts is not None:
            self.store_file.write(str(ts) + "\n")
            self.store_file.flush()

        self.img_idx += 1
        cv2.waitKey(1)

    def create_online_plots(self, num_subplots, xlabel, ylabels, legend, figsize=(4, 6), dpi=100):
        if self.ion:
            plt.ion()

        axes, lines = [], []
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=".0")
        for i in range(len(ylabels)):
            axes.append(fig.add_subplot(num_subplots, 1, i + 1))
            if i == 3:
                for l in legend:
                    lines.append(axes[i].plot([], [], label=l)[0])
                if len(legend) > 1:
                    axes[i].legend(loc=(0.65, -0.5), ncol=len(legend))
            else:
                for l in legend:
                    lines.append(axes[i].plot([], [])[0])
            axes[i].patch.set_facecolor("black")
            axes[i].tick_params(axis="both", colors="white")
            axes[i].yaxis.label.set_color("white")
            axes[i].xaxis.label.set_color("white")
            for side in axes[i].spines.keys():
                axes[i].spines[side].set_color("white")
            axes[i].set_ylabel(ylabels[i])
        axes[i].set_xlabel(xlabel)
        fig.tight_layout()
        fig.canvas.draw()

        return fig, axes, lines

    @staticmethod
    def update_online_plots(fig, axes, lines, xbuffer, ybuffer, data, yrange):
        if xbuffer.shape[0] == 0:
            xbuffer = data[:, 0:1]
            ybuffer = data[:, 1:]
        else:
            if data[:, 0:1] > xbuffer[-1, 0]:
                xbuffer = np.vstack((xbuffer, data[:, 0:1]))
                ybuffer = np.vstack((ybuffer, data[:, 1:]))
            else:
                valid = data[:, 0:1] >= xbuffer[:, 0]
                xbuffer = xbuffer[valid[0, :], :]
                ybuffer = ybuffer[valid[0, :], :]

        xbuffer = xbuffer[-500:, :]  # TODO: hardcoded
        ybuffer = ybuffer[-500:, :]  # TODO: hardcoded

        if yrange is None:
            yrange = [[1e6 for i in range(len(axes))], [-1e6 for i in range(len(axes))]]

        plots_per_axis = len(lines) // len(axes)
        for i in range(len(lines)):
            if ybuffer[:, i].min() < yrange[0][i // plots_per_axis]:
                yrange[0][i // plots_per_axis] = ybuffer[:, i].min() - 0.1
            if ybuffer[:, i].max() > yrange[1][i // plots_per_axis]:
                yrange[1][i // plots_per_axis] = ybuffer[:, i].max() + 0.1

        for i, line in enumerate(lines):
            line.set_data(xbuffer, ybuffer[:, i])
            if xbuffer.shape[0] > 1:
                axes[i // plots_per_axis].set_xlim(xbuffer[0], xbuffer[-1])
            else:
                axes[i // plots_per_axis].set_xlim(xbuffer[0], xbuffer[0] + 0.01)
            axes[i // plots_per_axis].set_ylim(yrange[0][i // plots_per_axis], yrange[1][i // plots_per_axis])

        fig.canvas.draw()

        return xbuffer, ybuffer, yrange

    @staticmethod
    def flow_to_image(flow_x, flow_y):
        """
        Use the optical flow color scheme from the supplementary materials of the paper 'Back to Event
        Basics: Self-Supervised Image Reconstruction for Event Cameras via Photometric Constancy',
        Paredes-Valles et al., CVPR'21.
        :param flow_x: [H x W x 1] horizontal optical flow component
        :param flow_y: [H x W x 1] vertical optical flow component
        :return flow_rgb: [H x W x 3] color-encoded optical flow
        """
        flows = np.stack((flow_x, flow_y), axis=2)
        mag = np.linalg.norm(flows, axis=2)
        min_mag = np.min(mag)
        mag_range = np.max(mag) - min_mag

        ang = np.arctan2(flow_y, flow_x) + np.pi
        ang *= 1.0 / np.pi / 2.0

        hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3])
        hsv[:, :, 0] = ang
        hsv[:, :, 1] = 1.0
        hsv[:, :, 2] = mag - min_mag
        if mag_range != 0.0:
            hsv[:, :, 2] /= mag_range

        flow_rgb = matplotlib.colors.hsv_to_rgb(hsv)
        return (255 * flow_rgb).astype(np.uint8)

    @staticmethod
    def minmax_norm(x):
        """
        Robust min-max normalization.
        :param x: [H x W x 1]
        :return x: [H x W x 1] normalized x
        """
        den = np.percentile(x, 99) - np.percentile(x, 1)
        if den != 0:
            x = (x - np.percentile(x, 1)) / den
        return np.clip(x, 0, 1)

    @staticmethod
    def events_to_image(event_cnt, color_scheme="green_red"):
        """
        Visualize the input events.
        :param event_cnt: [batch_size x 2 x H x W] per-pixel and per-polarity event count
        :param color_scheme: green_red/gray
        :return event_image: [H x W x 3] color-coded event image
        """
        pos = event_cnt[:, :, 0]
        neg = event_cnt[:, :, 1]
        pos_max = np.percentile(pos, 99)
        pos_min = np.percentile(pos, 1)
        neg_max = np.percentile(neg, 99)
        neg_min = np.percentile(neg, 1)
        max = pos_max if pos_max > neg_max else neg_max

        if pos_min != max:
            pos = (pos - pos_min) / (max - pos_min)
        if neg_min != max:
            neg = (neg - neg_min) / (max - neg_min)

        pos = np.clip(pos, 0, 1)
        neg = np.clip(neg, 0, 1)

        event_image = np.ones((event_cnt.shape[0], event_cnt.shape[1]))
        if color_scheme == "gray":
            event_image *= 0.5
            pos *= 0.5
            neg *= -0.5
            event_image += pos + neg

        elif color_scheme == "green_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            event_image *= 0
            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            event_image[:, :, 0][mask_pos] = 0
            event_image[:, :, 1][mask_pos] = pos[mask_pos]
            event_image[:, :, 2][mask_pos * mask_not_neg] = 0
            event_image[:, :, 2][mask_neg] = neg[mask_neg]
            event_image[:, :, 0][mask_neg] = 0
            event_image[:, :, 1][mask_neg * mask_not_pos] = 0

        elif color_scheme == "rpg":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            mask_pos = pos > 0
            mask_neg = neg > 0

            event_image[:, :, 0][mask_neg] = 1
            event_image[:, :, 1][mask_neg] = 0
            event_image[:, :, 2][mask_neg] = 0
            event_image[:, :, 0][mask_pos] = 0
            event_image[:, :, 1][mask_pos] = 0
            event_image[:, :, 2][mask_pos] = 1

        elif color_scheme == "prophesee":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            mask_pos = pos > 0
            mask_neg = neg > 0

            event_image[:, :, 0][mask_neg] = 0.24313725490196078
            event_image[:, :, 1][mask_neg] = 0.11764705882352941
            event_image[:, :, 2][mask_neg] = 0.047058823529411764
            event_image[:, :, 0][mask_pos] = 0.6352941176470588
            event_image[:, :, 1][mask_pos] = 0.4235294117647059
            event_image[:, :, 2][mask_pos] = 0.23529411764705882

        else:
            print("Visualization error: Unknown color scheme for event images.")
            raise AttributeError

        return event_image
