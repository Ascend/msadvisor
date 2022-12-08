import matplotlib.pylab as plt
import numpy as np

PIE_COLORS = ["cornflowerblue", "lightskyblue", "cyan", "springgreen", "greenyellow", "gold",
              "orange", "lightsalmon", "orangered", "pink", "mediumorchid", "olivedrab"]


class AscendTimeAnalysisVisualization:
    @staticmethod
    def task_type_visualization(task_data, save_path):
        task_type_list = []
        task_op_num_list = []
        task_op_time_list = []
        for k, v in task_data.items():
            if v["op_num"] > 0:
                task_type_list.append(k)
                task_op_num_list.append(v["op_num"])
                task_op_time_list.append(v["op_time"])
        fig = plt.figure(figsize=(10, 4), dpi=200)
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.axis("equal")
        ax1.pie(task_op_num_list, autopct="%1.2f%%", shadow=False, startangle=0, colors=PIE_COLORS)
        ax1.set_title("Task Num Distribution")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axis("equal")
        ax2.pie(task_op_time_list, autopct="%1.2f%%", shadow=False, startangle=0, colors=PIE_COLORS)
        ax2.set_title("Task Time Distribution")
        fig.legend(task_type_list, loc="lower center", ncol=2)
        plt.savefig("{}/{}.jpg".format(save_path, "Task Type Distribution"))
        plt.close()

    @staticmethod
    def hotspot_task_visualization(topk_op, task_type, save_path, topk=3):
        op_name_list = [""] * topk
        op_time_list = [0.0] * topk
        for idx, op in enumerate(topk_op):
            op_name_list[idx] = op["name"]
            op_time_list[idx] = float(op["dur"]) / 1000
        fig = plt.figure(figsize=(4, 10), dpi=200)
        plt.bar(np.arange(len(op_time_list)), op_time_list, color=["r", "g", "b"], width=0.5)
        for x in np.arange(len(op_name_list)):
            plt.text(x, op_time_list[x], format(op_time_list[x], ".2f"), ha="center")
        plt.xticks(np.arange(len(op_name_list)), op_name_list, rotation=90)
        plt.ylabel("op time(ms)")
        plt.title("{} hotspot task".format(task_type))
        plt.savefig("{}/{}_hotspot_task.jpg".format(save_path, task_type))
        plt.close()

    @staticmethod
    def execution_time_visualization(execution_time_data, save_path):
        execution_time_name = [i for i in execution_time_data.keys()]
        execution_time = [float(i) / 1000 for i in execution_time_data.values()]
        fig = plt.figure(figsize=(6, 10), dpi=200)
        plt.bar(np.arange(len(execution_time)), execution_time, color=["r", "g", "b", "peru"], width=0.5)
        for x in np.arange(len(execution_time)):
            plt.text(x, execution_time[x], format(execution_time[x], ".2f"), ha="center")
        plt.xticks(np.arange(len(execution_time_name)), execution_time_name, rotation=45)
        plt.ylabel("time(ms)")
        plt.title("Workload Execution Time Distribution")
        plt.savefig(f"{save_path}/workload_execution_time_distribution.jpg")
        plt.close()
