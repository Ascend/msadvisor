# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import matplotlib.pyplot as plt
import numpy as np

from utils.constant import Constant


BAR_WIDTH = 0.2
TRANSPORT_TYPE_DISPLAY_DICT = {0: "", 1: "HCCS", 2: "PCIE", 3: "RDMA"}


class HcclVisualization:
    @staticmethod
    def plot_data(ax, xvalue, yvalue, fig_type, fontsize):
        ax.bar(range(len(xvalue)), yvalue, BAR_WIDTH, tick_label=xvalue)
        for i in range(len(xvalue)):
            ax.text(i, yvalue[i], yvalue[i], color="red", fontsize=fontsize)
        ax.set_title("{} Packet Size Distribution".format(fig_type), fontdict={"fontsize": fontsize})
        ax.set_xlabel("Packet Size(Byte)", fontdict={"fontsize": fontsize})
        ax.set_ylabel("Packet Num", fontdict={"fontsize": fontsize})

    @staticmethod
    def draw_communication_time_distribution(op_time_info, save_path, fontsize=30):
        rank_list = op_time_info[0]
        elapse_time_list = op_time_info[1]
        transit_time_list = op_time_info[2]
        synchronization_time_list = op_time_info[3]
        wait_time_list = op_time_info[4]
        synchronization_ratio_list = op_time_info[5]
        wait_ratio_list = op_time_info[6]
        fig = plt.figure(figsize=(20, 20), dpi=200)
        plt.title("Hccl Op Time Analysis", fontsize=fontsize)
        bar_width = BAR_WIDTH
        x = np.arange(len(rank_list))
        plt.bar(x, elapse_time_list, bar_width, label="elapse_time")
        plt.bar(x + bar_width, transit_time_list, bar_width, label="transit_time")
        plt.bar(x + 2 * bar_width, synchronization_time_list, bar_width, label="synchronization_time")
        plt.bar(x + 3 * bar_width, wait_time_list, bar_width, label="wait_time")
        plt.xticks(x + 2 * bar_width, rank_list)
        plt.xlabel("RankID", fontdict={"fontsize": fontsize})
        plt.ylabel("Time(ms)", fontdict={"fontsize": fontsize})
        for x_value, y_value in zip(x + bar_width, transit_time_list):
            if y_value > 0:
                plt.text(x_value, y_value, y_value, ha="center", va="bottom", fontsize=15)
        plt.legend(loc="upper left", prop={"size": fontsize})
        ax2 = plt.twinx()
        plt.ylabel("Ratio", fontdict={"fontsize": fontsize})
        plt.plot(x + 2 * bar_width, synchronization_ratio_list, marker="d", c="black",
                 label="synthronization_time_ratio_before_transit")
        for x_value, y_value in zip(x + 2 * bar_width, synchronization_ratio_list):
            if y_value > 0:
                plt.text(x_value, y_value, y_value, ha="center", va="bottom", fontsize=fontsize)
        plt.plot(x + 2 * bar_width, wait_ratio_list, marker="o", c="r", label="wait_time_ratio")
        for x_value, y_value in zip(x + 2 * bar_width, wait_ratio_list):
            if y_value > 0:
                plt.text(x_value, y_value, y_value, ha="center", va="bottom", fontsize=fontsize)

        plt.legend(loc="upper right", prop={"size": fontsize})
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig("{}/{}.jpg".format(save_path, "Communication Time Analysis"))
        plt.close()

    @staticmethod
    def draw_communication_size_distribution(transit_size_info, save_path, fontsize=30):
        fig = plt.figure(figsize=(20, 20), dpi=200)
        fig_type_list = []
        for transit_type in Constant.TRANSIT_TYPE:
            if transit_type in transit_size_info.keys():
                fig_type_list.append(transit_type)
        for fig_index, fig_type in enumerate(fig_type_list):
            ax = fig.add_subplot(1, len(fig_type_list), fig_index + 1)
            x_value = list(transit_size_info.get(fig_type).keys())
            x_value.sort()
            y_value = [transit_size_info.get(fig_type)[val] for val in x_value]
            HcclVisualization.plot_data(ax, x_value, y_value, fig_type, fontsize)
        if len(fig_type_list) > 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig("{}/{}.jpg".format(save_path, "Packet Size Distribution"))
            plt.close()

    @staticmethod
    def draw_heatmap(input_data, heatmap_title, cluster_max_rank, save_path, fontsize=240):
        fontsize = fontsize // cluster_max_rank
        axis_label = np.arange(0, cluster_max_rank, 1)
        fig = plt.figure(figsize=(20, 20), dpi=200)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks(axis_label)
        ax.set_xticklabels(axis_label)
        ax.set_yticks(axis_label)
        ax.set_yticklabels(axis_label)
        ax.set_xlabel("dst_rank_id", fontdict={"size": 30})
        ax.set_ylabel("src_rank_id", fontdict={"size": 30})
        ax.set_title(heatmap_title, fontdict={"size": 30})
        im = ax.imshow(input_data, cmap="YlGn", origin="upper")
        plt.colorbar(im)
        for dst_rank_idx in range(cluster_max_rank):
            for src_rank_idx in range(cluster_max_rank):
                if input_data[dst_rank_idx][src_rank_idx] <= 0 or input_data[dst_rank_idx][src_rank_idx] == "nan":
                    continue
                if "Transport Type" in heatmap_title:
                    ax.text(src_rank_idx, dst_rank_idx,
                            TRANSPORT_TYPE_DISPLAY_DICT[int(input_data[dst_rank_idx][src_rank_idx])],
                            ha="center", va="center", color="black", fontsize=fontsize)
                else:
                    ax.text(src_rank_idx, dst_rank_idx, format(input_data[dst_rank_idx][src_rank_idx], ".2f"),
                            ha="center", va="center", color="black", fontsize=fontsize)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig("{}/{}.jpg".format(save_path, heatmap_title.split("(")[0]))
        plt.close()

