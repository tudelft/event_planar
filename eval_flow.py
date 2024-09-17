import argparse
import collections
import os

import mlflow
import torch
from torch.optim import *

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import EvalCriteria
from models.model import *
from utils.activity_packager import Packager
from utils.homography import HomographyWarping
from utils.iwe import compute_pol_iwe
from utils.utils import load_model, create_model_dir, save_state_dict
from utils.mlflow import log_config, log_results
from utils.visualization import Visualization


def test(args, config_parser):
    mlflow.set_tracking_uri(args.path_mlflow)

    run = mlflow.get_run(args.runid)
    config = config_parser.merge_configs(run.data.params)
    config = config_parser.combine_entries(config)

    # configs
    if config["loader"]["batch_size"] > 1:
        config["vis"]["activity"] = False
        config["vis"]["bars"] = False
        config["vis"]["enabled"] = False
        config["vis"]["ground_truth"] = False
        config["vis"]["store"] = False

    distortion = False
    if "distortion" in config.keys():
        if config["distortion"]["undistort"]:
            distortion = True

    # create directory for inference results
    path_results = create_model_dir(args.path_results, args.runid)

    # store validation settings
    eval_id = log_config(path_results, args.runid, config)

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs
    config["loader"]["device"] = device

    # visualization tool
    vis = Visualization(config, eval_id=eval_id, path_results=path_results)
    if config["vis"]["activity"]:
        activity_dir = path_results + "activity/" + str(eval_id) + "/"
        if not os.path.exists(activity_dir):
            os.makedirs(activity_dir)

    # data loader
    data = H5Loader(config)
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # activate axonal delays for loihi compatible networks
    if config["model"]["name"] in ["LoihiRec4ptNet", "SplitLoihiRec4ptNet"]:
        config["model"]["spiking_neuron"]["delay"] = True

    # model initialization and settings
    num_bins = 2 if config["data"]["voxel"] is None else config["data"]["voxel"]
    model = eval(config["model"]["name"])(config["model"].copy(), config["data"]["crop"], num_bins)
    model = model.to(device)
    model = load_model(args.runid, model, device)
    save_state_dict(args.runid, model)
    model.eval()
    if config["vis"]["activity"]:
        model.store_activity()

    # homogrpahy projection
    homography = HomographyWarping(config, flow_scaling=config["loss"]["flow_scaling"], K=data.cam_mtx)

    # validation metric
    criteria = EvalCriteria(config, device)

    # inference loop
    val_results = {}
    end_test = False
    prev_sequence = None
    axonal_delays = None

    gt_bf = {}
    ts_bf = None
    event_list_bf = None
    event_mask_bf = None
    event_list_pol_mask_bf = None

    with torch.no_grad():
        while True:
            for inputs in dataloader:
                sequence = data.files[data.batch_idx[0] % len(data.files)].split("/")[-1].split(".")[0]
                if config["vis"]["activity"]:
                    if not os.path.exists(activity_dir + sequence + ".h5"):
                        packager = Packager(activity_dir + sequence + ".h5")
                        sample_idx = 0

                if data.new_seq:
                    data.new_seq = False

                    if config["vis"]["ground_truth"] and prev_sequence is not None:
                        if prev_sequence not in val_results.keys():
                            val_results[prev_sequence] = {}
                        val_results[prev_sequence]["AEE"] = criteria.aee(criteria.gt).cpu().numpy()
                        vis.store_ground_truth(criteria.gt, prev_sequence)
                        homography.reset_pose_gt()

                    ts_bf.clear()
                    event_list_bf.clear()
                    event_mask_bf.clear()
                    event_list_pol_mask_bf.clear()
                    for key in gt_bf.keys():
                        gt_bf[key].clear()
                    vis.final_spike_rate = None

                    model.reset_states()
                    criteria.reset_gt()
                    criteria.reset_rsat()

                # finish inference loop
                if data.seq_num >= len(data.files):
                    end_test = True
                    break

                # forward pass
                x = model(inputs["net_input"].to(device))

                # axonal delays
                if axonal_delays is None:
                    axonal_delays = getattr(model.encoder_unet, "delays", 0)
                    ts_bf = collections.deque(maxlen=axonal_delays + 1)
                    event_list_bf = collections.deque(maxlen=axonal_delays + 1)
                    event_mask_bf = collections.deque(maxlen=axonal_delays + 1)
                    event_list_pol_mask_bf = collections.deque(maxlen=axonal_delays + 1)
                    for key in inputs.keys():
                        if key.split("_")[0] == "gt":
                            gt_bf[key] = collections.deque(maxlen=axonal_delays + 1)

                # input buffer
                ts_bf.append(data.last_proc_timestamp)
                event_list_bf.append(inputs["event_list"].to(device))
                event_mask_bf.append(inputs["event_mask"].to(device))
                event_list_pol_mask_bf.append(inputs["event_list_pol_mask"].to(device))
                for key in gt_bf.keys():
                    gt_bf[key].append(inputs[key].to(device))

                # homography projection
                flow_vectors = x["flow_vectors"].clone()
                if config["data"]["mode"] == "time":
                    flow_vectors *= config["data"]["window"] / 0.001  # (flow in px/ms -> flow in px/input_time)
                flow_list = homography.get_flow_map(flow_vectors)

                # optical flow scaling
                x["flow_vectors"] *= config["loss"]["flow_scaling"]

                # mask flow for visualization
                flow_vis = flow_list[-1].clone()
                flow_vis *= event_mask_bf[0]

                # image of warped events
                iwe = compute_pol_iwe(
                    flow_vis,
                    event_list_bf[0],
                    config["loader"]["resolution"],
                    event_list_pol_mask_bf[0][:, :, 0:1],
                    event_list_pol_mask_bf[0][:, :, 1:2],
                    round_idx=False,
                    distortion=distortion,
                )

                iwe_window_vis = None
                events_window_vis = None
                masked_window_flow_vis = None

                # update criteria
                if len(ts_bf) == axonal_delays + 1:
                    criteria.event_flow_association(
                        flow_list, event_list_bf[0], event_list_pol_mask_bf[0], event_mask_bf[0]
                    )
                    if config["vis"]["ground_truth"]:
                        gt_flow_list = homography.pose_to_4ptflow(gt_bf["gt_position_OT"][0], gt_bf["gt_Euler_imu"][0])
                        criteria.update_gt(ts_bf[0], x["flow_vectors"], gt_flow_list)

                # validation
                if criteria.num_passes >= config["data"]["passes_loss"]:
                    # compute metric
                    deblurring_metric = criteria.rsat()

                    # accumulate results
                    for batch in range(config["loader"]["batch_size"]):
                        if sequence not in val_results.keys():
                            val_results[sequence] = {}
                            val_results[sequence]["it"] = 0
                            val_results[sequence]["RSAT"] = 0
                            val_results[sequence]["Hamming"] = 0

                        if event_list_bf[0].shape[1] > 0:
                            val_results[sequence]["it"] += 1
                            val_results[sequence]["RSAT"] += deblurring_metric[batch].cpu().numpy()

                    # visualize
                    if (config["vis"]["enabled"] or config["vis"]["store"]) and criteria.num_passes > 1:
                        events_window_vis = criteria.compute_window_events()
                        iwe_window_vis = criteria.compute_window_iwe()
                        masked_window_flow_vis = criteria.compute_masked_window_flow()

                    # reset criteria
                    criteria.reset_rsat()
                    prev_sequence = sequence

                # visualize
                if config["vis"]["bars"]:
                    for bar in data.open_files_bar:
                        bar.next()
                if config["vis"]["enabled"]:
                    vis.update(
                        inputs,
                        flow_vis,
                        iwe,
                        events_window_vis,
                        masked_window_flow_vis,
                        iwe_window_vis,
                        flow_vectors=x["flow_vectors"],
                    )
                if config["vis"]["store"]:
                    vis.store(
                        inputs,
                        flow_vis,
                        sequence,
                        iwe,
                        events_window_vis,
                        masked_window_flow_vis,
                        iwe_window_vis,
                        flow_vectors=x["flow_vectors"],
                        vision_spikes=x["spikes"].clone(),
                        ts=data.last_proc_timestamp,
                    )
                if config["vis"]["activity"]:
                    if "activity" in x.keys() and x["activity"] is not None:
                        for key in x["activity"]:
                            packager.package_array(x["activity"][key], sample_idx, dir=key)
                    packager.package_array(x["flow_vectors"], sample_idx, dir="4pt")
                    sample_idx += 1

            if end_test:
                break

    if config["vis"]["bars"]:
        for bar in data.open_files_bar:
            bar.finish()

    # store validation config and results
    results = {}
    results["AEE"] = {}
    results["RSAT"] = {}
    results["Hamming"] = {}
    for key in val_results.keys():
        results["RSAT"][key] = str(val_results[key]["RSAT"] / val_results[key]["it"])
        if config["vis"]["ground_truth"]:
            results["AEE"][key] = str(val_results[key]["AEE"])

    log_results(args.runid, results, path_results, eval_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", help="mlflow model run")
    parser.add_argument(
        "--config",
        default="configs/eval_flow.yml",
        help="config file, overwrites mlflow settings",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument("--path_results", default="results_inference/")
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))
