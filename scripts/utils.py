import os
import random
import resource

import dgl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


def setup_environment(config, cuda_visible_device="", seed=None):

    if cuda_visible_device:
        ngpus = len(cuda_visible_device.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_device

        torch.set_float32_matmul_precision("medium")  # For A6000 GPUs
    else:
        ngpus = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    num_cpu = config.get("torch_num_cpu", 4)  # Use 4 as a safe default
    if num_cpu is not None:
        torch.set_num_threads(num_cpu)
        torch.set_num_interop_threads(num_cpu)

    num_ray = config.get("training", {}).get("num_ray", 0)
    if num_ray > 0:
        import ray
        ray.init(num_cpus=num_ray, include_dashboard=False)

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

    if seed is None:
        random.seed(None)
        np.random.seed(None)
    else:
        random.seed(1234 + seed)
        np.random.seed(12345 + seed)
        torch.manual_seed(123456 + seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

    return ngpus


def get_latest_checkpoints(lightning_logs="workspace/hgpflow/lightning_logs"):
    checkpoints_dir = os.path.join(lightning_logs, "checkpoints")

    if not os.path.exists(checkpoints_dir):
        return None

    best_model_path_file = os.path.join(checkpoints_dir, "best_model_path")
    if os.path.exists(best_model_path_file):
        with open(best_model_path_file, "r") as f:
            best_model_path = f.read()

        if os.path.exists(best_model_path):
            return best_model_path
        elif os.path.exists(os.path.join(checkpoints_dir, os.path.basename(best_model_path))):
            print(
                f"best_model_path ({best_model_path}) not found, using the file in checkpoints directory."
            )
            return os.path.join(checkpoints_dir, os.path.basename(best_model_path))
        else:
            raise ValueError(f"best_model_path ({best_model_path}) does not exist.")

    files_with_ctime = [
        (d, os.path.getctime(os.path.join(checkpoints_dir, d)))
        for d in os.listdir(checkpoints_dir)
        if os.path.isfile(os.path.join(checkpoints_dir, d))
    ]

    if len(files_with_ctime) == 0:
        return None

    latest_checkpoint = max(files_with_ctime, key=lambda x: x[1])[0]
    return os.path.join(checkpoints_dir, latest_checkpoint)


def _color_text(text, color):
    if float(text) >= 0.1:
        return f"\033[31m{text}\033[0m"
    else:
        return text
    # r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
    # return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def _print_colored_value(value, min_value, max_value, end="\n", energy=False):
    cmap = plt.get_cmap("viridis")  # 'viridis', 'plasma', 'inferno', 'magma'
    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)
    color = cmap(norm(value))
    if energy:
        print(_color_text(f"{value:3.0f}", color), end=end)
    else:
        print(_color_text(f"{value:2.1f}", color), end=end)


def _print_incidence_matrix(matrix, shape=None):
    ncols, nrows = shape if shape else matrix.shape
    _matrix = matrix.reshape((ncols, nrows))

    for j in range(nrows):
        for i in range(ncols):
            _print_colored_value(_matrix[i][j], 0.0, 1.0, end=" ")
        print("")


def _print_indicator(indicator):
    for value in indicator.reshape((-1,)):
        _print_colored_value(value, 0.0, 1.0, end=" ")
    print("")


def print_incidence_matrix_from_g(g):
    g_unbatched = dgl.unbatch(g)
    for _g in g_unbatched:
        if "incidence_val_logit" in _g.edges["pflow_to_node"].data:
            incidence_val = torch.sigmoid(_g.edges["pflow_to_node"].data["incidence_val_logit"])
        elif "incidence_val" in _g.edges["pflow_to_node"].data:
            incidence_val = _g.edges["pflow_to_node"].data["incidence_val"]
        else:
            raise NotImplementedError()

        incidence_val = incidence_val.detach().cpu().numpy()
        _print_incidence_matrix(
            incidence_val, shape=(_g.num_nodes(ntype="pflows"), _g.num_nodes(ntype="nodes"))
        )


def print_indicator_from_g(g):
    g_unbatched = dgl.unbatch(g)
    for _g in g_unbatched:
        if "indicator_logit" in _g.nodes["pflows"].data:
            indicator = torch.sigmoid(_g.nodes["pflows"].data["indicator_logit"])
        elif "indicator" in _g.nodes["pflows"].data:
            indicator = _g.nodes["pflows"].data["indicator"]
        else:
            raise NotImplementedError()

        indicator = indicator.detach().cpu().numpy()
        _print_indicator(indicator)
