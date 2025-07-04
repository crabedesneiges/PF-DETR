import numpy as np

"""
    Class Label
    | Particle       | Label in evaluation | Label in training |
    | charged hadron | 0 | 0 |
    | electron       | 1 | 1 |
    | muon           | 2 | 2 |
    | neutral hadron | 3 | 0 |
    | photon         | 4 | 1 |

    neutral hadron: n, pion0, K0, Xi0, lambda
    charged hadron: p+-, K+-, pion+-, Xi+, Omega, Sigma

    Charge Label
    | Charge | Label |
    | 0 | 0 |
    | + | 1 |
    | - | 2 |
"""

_class_labels = {
    0: [211, -211, 321, -321, 2212, -2212, -3112, 3112, 3222, -3222, 3312, -3312, -3334, 3334],
    1: [11, -11],
    2: [13, -13],
    3: [130, 310, 2112, -2112, 3122, -3122, 3322, -3322],
    4: [22],
}
_class_labels = {pdgid: label for label, pdgids in _class_labels.items() for pdgid in pdgids}

_charge_labels = {
    0: [22, 130, 310, 2112, -2112, 3122, -3122, 3322, -3322],  # 0
    1: [-11, -13, 211, 321, 2212, -3112, 3222, -3312, -3334],  # +
    2: [11, 13, -211, -321, -2212, 3112, -3222, 3312, 3334],  # -
}
_charge_labels = {pdgid: label for label, pdgids in _charge_labels.items() for pdgid in pdgids}


def class_label(pdgids, combined_index=True):
    label = np.array([_class_labels[x] for x in pdgids])
    if not combined_index:
        label = np.where(label >= 3, label - 3, label)

    return label


def charge_label(pdgids):
    label = np.array([_charge_labels[x] for x in pdgids])

    return label
