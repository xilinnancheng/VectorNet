from argoverse.data_loading.argoverse_forecasting_loader import (
    ArgoverseForecastingLoader,
)
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from utils.config import color_dict
import torch
import numpy


def show_doubled_lane(polygon):
    """
    args: ndarray in shape of (n, 2)
    returns:
    """
    xs, ys = polygon[:, 0], polygon[:, 1]
    plt.plot(xs, ys, "--", color="grey")


def show_traj(traj, type_):
    """
    args: ndarray in shape of (n, 2)
    returns:
    """
    if type_ == "AGENT" or type_ == "AV":
        traj_color = color_dict[type_]
    else:
        traj_color = np.random.rand(
            3,
        )
    plt.plot(traj[:, 0], traj[:, 1], color=traj_color)


def reconstract_polyline(features, traj_mask, lane_mask, add_len):
    traj_ls, lane_ls = [], []
    for id_, mask in traj_mask.items():
        data = features[mask[0] : mask[1]]
        traj = np.vstack((data[:, 0:2], data[-1, 2:4]))
        traj_ls.append(traj)
    for id_, mask in lane_mask.items():
        data = features[mask[0] + add_len : mask[1] + add_len]
        # lane = np.vstack((data[:, 0:2], data[-1, 3:5]))
        # change lanes feature to (xs, ys, zs, xe, ye, ze, polyline_id)
        lane = np.vstack((data[:, 0:2], data[-1, 2:4]))
        lane_ls.append(lane)
    return traj_ls, lane_ls


def show_pred_and_gt(pred_y, y):
    min_x = min(numpy.min(pred_y[:, 0], axis=0), numpy.min(y[:, 0], axis=0))
    max_x = min(numpy.max(pred_y[:, 0], axis=0), numpy.max(y[:, 0], axis=0))

    min_y = min(numpy.min(pred_y[:, 1], axis=0), numpy.min(y[:, 1], axis=0))
    max_y = min(numpy.max(pred_y[:, 1], axis=0), numpy.max(y[:, 1], axis=0))

    plt.xlim(min_x - 5, max_x + 5)
    plt.ylim(min_y - 5, max_y + 5)
    plt.plot(y[:, 0], y[:, 1], marker="x", color="r")
    plt.plot(pred_y[:, 0], pred_y[:, 1], lw=0, marker="o", fillstyle="none")


def show_predict_result(data, pred_y: torch.Tensor, y, add_len, show_lane=True):
    features, _ = data["POLYLINE_FEATURES"].values[0], data["GT"].values[0].astype(
        np.float32
    )
    traj_mask, lane_mask = (
        data["TRAJ_ID_TO_MASK"].values[0],
        data["LANE_ID_TO_MASK"].values[0],
    )

    traj_ls, lane_ls = reconstract_polyline(features, traj_mask, lane_mask, add_len)

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    type_ = "AGENT"
    plt.figure(figsize=[10, 10])
    for traj in traj_ls:
        show_traj(traj, type_)
        plt.plot(traj[0, 0], traj[0, 1], "x", color="blue", markersize=4)
        plt.plot(traj[-1, 0], traj[-1, 1], "*", color="blue", markersize=4)
        min_x = min(numpy.min(traj[:, 0], axis=0), min_x)
        max_x = max(numpy.max(traj[:, 0], axis=0), max_x)

        min_y = min(numpy.min(traj[:, 1], axis=0), min_y)
        max_y = max(numpy.max(traj[:, 1], axis=0), max_y)
        type_ = "OTHERS"

    if show_lane:
        for lane in lane_ls:
            show_doubled_lane(lane)
            min_x = min(numpy.min(lane[:, 0], axis=0), min_x)
            max_x = max(numpy.max(lane[:, 0], axis=0), max_x)

            min_y = min(numpy.min(lane[:, 1], axis=0), min_y)
            max_y = max(numpy.max(lane[:, 1], axis=0), max_y)

    pred_y = pred_y.numpy().reshape((-1, 2)).cumsum(axis=0)
    y = y.numpy().reshape((-1, 2)).cumsum(axis=0)
    show_pred_and_gt(pred_y, y)
    if (
        min_x != float("inf")
        and max_x != float("-inf")
        and min_y != float("inf")
        and max_y != float("-inf")
    ):
        plt.xlim(min_x - 5, max_x + 5)
        plt.ylim(min_y - 5, max_y + 5)
