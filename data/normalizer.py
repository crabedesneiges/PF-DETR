import json
import os

import numpy as np
from sklearn import preprocessing


class Normalizer:
    def __init__(self, normalization_path):
        self.normalization_path = normalization_path
        self.normalization_params = None
        self.scaler = {}

    def fit_and_dump(self, targets_dict, outputname):
        normalization_params = {}

        for name, data in targets_dict.items():
            scaler = preprocessing.StandardScaler()
            if len(data.shape) == 1:
                scaler = scaler.fit(data[:, np.newaxis])
            else:
                scaler = scaler.fit(data)
            normalization_params[name] = {
                "mean_": scaler.mean_.tolist(),
                "scale_": scaler.scale_.tolist(),
            }

        os.makedirs(os.path.dirname(outputname), exist_ok=True)
        with open(outputname, "w") as f:
            json.dump(normalization_params, f, indent=2)

    def normalize(self, x, name, inverse=False):
        if self.normalization_path == "":
            return x

        if self.normalization_params is None:
            if not os.path.exists(self.normalization_path):
                raise ValueError(f"{self.normalization_path} does not exist.")

            with open(self.normalization_path) as f:
                self.normalization_params = json.load(f)

        if name not in self.scaler:
            self.scaler[name] = preprocessing.StandardScaler()
            self.scaler[name].mean_ = np.array(self.normalization_params[name]["mean_"])
            self.scaler[name].scale_ = np.array(self.normalization_params[name]["scale_"])

        dtype = x.dtype
        if np.issubdtype(dtype, np.integer):
            if name in ["cell_layer", "cell_topo_idx"]:
                dtype = np.float32()
            else:
                raise ValueError(f"{name} is integer. Please be carefull")

        if len(x.shape) == 1:
            x = x[:, np.newaxis]
            if inverse:
                x = self.scaler[name].inverse_transform(x)
            else:
                x = self.scaler[name].transform(x)
            x = x[:, 0]
        else:
            if inverse:
                x = self.scaler[name].inverse_transform(x)
            else:
                x = self.scaler[name].transform(x)

        x = x.astype(dtype)
        return x

    def denormalize(self, x, name):
        return self.normalize(x, name, inverse=True)
