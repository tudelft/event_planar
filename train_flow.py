import argparse

import mlflow
import torch
from torch.optim import *

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import EventWarping
from models.model import *
from utils.homography import HomographyWarping
from utils.gradients import get_grads
from utils.utils import load_model, save_csv, save_diff, save_model
from utils.visualization import Visualization


def train(args, config_parser):
    mlflow.set_tracking_uri(args.path_mlflow)

    # configs
    config = config_parser.config
    mlflow.set_experiment(config["experiment"])
    mlflow.start_run()
    mlflow.log_params(config)
    mlflow.log_param("prev_runid", args.prev_runid)
    config = config_parser.combine_entries(config)
    print("MLflow dir:", mlflow.active_run().info.artifact_uri[:-9])

    # log git diff
    save_diff("train_diff.txt")

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs
    config["loader"]["device"] = device
    assert config["data"]["passes_bptt"] % config["data"]["passes_loss"] == 0

    # visualization tool
    if config["vis"]["enabled"]:
        vis = Visualization(config)

    # data loader
    data = H5Loader(config, shuffle=True, path_cache=args.path_cache)
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # model initialization and settings
    num_bins = 2 if config["data"]["voxel"] is None else config["data"]["voxel"]
    model = eval(config["model"]["name"])(config["model"].copy(), config["data"]["crop"], num_bins)
    model = model.to(device)
    model = load_model(args.prev_runid, model, device)
    model.train()

    # loss functions
    loss_function = EventWarping(config, device)

    # homogrpahy projection
    homography = HomographyWarping(config, flow_scaling=config["loss"]["flow_scaling"])

    # optimizers
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
    optimizer.zero_grad()

    # simulation variables
    bw_loss = 0
    bptt_passes = 0
    train_loss = 0
    best_loss = 1.0e6
    end_train = False
    grads_w = []

    # training loop
    while True:
        for inputs in dataloader:
            if data.new_seq:
                data.new_seq = False

                if torch.is_tensor(bw_loss):
                    bw_loss.backward()

                bw_loss = 0
                bptt_passes = 0
                optimizer.zero_grad()
                model.reset_states()
                loss_function.reset()

            if data.seq_num >= len(data.files):
                mlflow.log_metric("loss", train_loss / (data.samples + 1), step=data.epoch)

                with torch.no_grad():
                    if train_loss / (data.samples + 1) < best_loss:
                        save_model(model)
                        best_loss = train_loss / (data.samples + 1)

                data.epoch += 1
                data.samples = 0
                train_loss = 0
                data.seq_num = data.seq_num % len(data.files)

                # save grads to file
                if config["vis"]["store_grads"]:
                    save_csv(grads_w, "grads_w.csv")
                    grads_w = []

                # finish training loop
                if data.epoch == config["loader"]["n_epochs"]:
                    end_train = True

            # forward pass
            x = model(inputs["net_input"].to(device))

            # time scaling (flow in px/ms -> flow in px/input_time)
            if config["data"]["mode"] == "time":
                x["flow_vectors"] = x["flow_vectors"] * config["data"]["window"] / 0.001

            # homography projection
            flow_list = homography.get_flow_map(x["flow_vectors"])

            # event flow association
            loss_function.update_flow_loss(
                x["flow_vectors"],
                flow_list,
                inputs["event_list"].to(device),
                inputs["event_list_pol_mask"].to(device),
            )

            # loss computation
            if loss_function.num_passes >= config["data"]["passes_loss"]:
                # update number of loss samples seen by the network
                data.samples += config["loader"]["batch_size"]

                # loss
                loss = loss_function()
                train_loss += loss.item()
                bw_loss += loss

                # mask flow for visualization
                if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                    flow_vis = flow_list[-1]
                    flow_vis *= inputs["event_mask"].to(device)

                # reset loss function
                loss_function.reset()

                # visualize
                with torch.no_grad():
                    if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                        vis.update(inputs, flow_vis, None)

                # print training info
                if config["vis"]["verbose"]:
                    print(
                        "Train Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)] Loss: {:.6f}".format(
                            data.epoch,
                            data.seq_num,
                            len(data.files),
                            int(100 * data.seq_num / len(data.files)),
                            train_loss / (data.samples + 1),
                        ),
                        end="\r",
                    )

            # backward pass
            bptt_passes += 1
            if bptt_passes >= config["data"]["passes_bptt"]:
                bw_loss.backward()

                # clip grads and optimize
                if config["loss"]["clip_grad"] is not None:
                    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
                if config["vis"]["store_grads"]:
                    grads_w.append(get_grads(model.named_parameters()))
                optimizer.step()
                optimizer.zero_grad()

                # reset loss info
                bw_loss = 0
                bptt_passes = 0
                model.detach_states()

        if end_train:
            break

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_flow.yml",
        help="training configuration",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument(
        "--path_cache",
        default="",
        help="location of the cache version of the formatted dataset",
    )
    parser.add_argument(
        "--prev_runid",
        default="",
        help="pre-trained model to use as starting point",
    )
    args = parser.parse_args()

    # launch training
    train(args, YAMLParser(args.config))
