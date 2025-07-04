import os

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit, linear_sum_assignment


def _calc_track_pt_sigma(pt, ref_pt, bin_pt, fignames=None, verbose=0):

    def _sigma_from_q1q3(x):
        if len(x) > 0:
            return (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / (2 * 0.6745)
        else:
            return 0.0

    def _sigma_from_fit(x, fignames=None):
        def _gauss(x, N, mean, sigma):
            s2 = sigma**2
            return N / np.sqrt(2 * np.pi * s2) * np.exp(-((x - mean) ** 2) / (2 * s2))

        def _fit(x, xmin, xmax, init_params=None, nbins=20):
            hist, bins = np.histogram(x, bins=nbins, range=(xmin, xmax))
            center = (bins[:-1] + bins[1:]) / 2

            if init_params is None:
                init_params = [len(x), np.quantile(x, 0.5), 10.0]

            params, covariance = curve_fit(
                f=_gauss,
                xdata=center,
                ydata=hist,
                sigma=np.sqrt(hist),
                p0=init_params,
            )
            return params  # C, mean, sigma

        if len(x) == 0:
            return -1.0

        hist, bins = np.histogram(x, bins=20, range=(-50.0, 50.0))
        center = (bins[:-1] + bins[1:]) / 2
        mode = center[np.argmax(hist)]
        delta = min(20.0, (np.quantile(x, 0.90) - np.quantile(x, 0.10)) / 2)
        xmin1, xmax1 = mode - delta, mode + delta
        C1, mean1, sigma1 = _fit(x, xmin1, xmax1, nbins=20)
        xmin2, xmax2 = mean1 - 2 * sigma1, mean1 + 2 * sigma1
        C2, mean2, sigma2 = _fit(
            x,
            xmin2,
            xmax2,
            init_params=(C1 * (4 * sigma1) / (xmax1 - xmin1), mean1, sigma1),
            nbins=20,
        )

        if fignames:
            _ = plt.figure(figsize=(8, 6))
            xmin, xmax = -50.0, 50.0
            _ = plt.hist(x, bins=100, range=(xmin, xmax), label="data")

            x_data1 = np.linspace(xmin1, xmax1, 100)
            norm1 = C1 * ((xmax - xmin) / 100) / ((xmax1 - xmin1) / 20)
            plt.plot(
                x_data1,
                _gauss(x_data1, norm1, mean1, sigma1),
                color="red",
                label="1st fit",
            )
            x_data2 = np.linspace(xmin2, xmax2, 100)
            norm2 = C2 * ((xmax - xmin) / 100) / (4 * sigma1 / 20)
            plt.plot(
                x_data2,
                _gauss(x_data2, norm2, mean2, sigma2),
                color="orange",
                label="2nd fit",
            )
            plt.legend()
            plt.savefig(fignames, dpi=300)
            plt.close()

        return sigma2

    sigma_pt_q1q3 = np.empty(len(bin_pt))
    sigma_pt_std = np.empty(len(bin_pt))
    sigma_pt_fit = np.empty(len(bin_pt))
    sigma_qopt_q1q3 = np.empty(len(bin_pt))
    sigma_qopt_std = np.empty(len(bin_pt))

    if fignames:
        _ = plt.figure(figsize=(12, 24))

    for i, (low, high) in enumerate(bin_pt):
        idx = np.logical_and(ref_pt > low, ref_pt < high)
        d_pt = (ref_pt - pt)[idx]
        d_pt_qopt = ((1.0 / ref_pt - 1.0 / pt) * np.square(ref_pt))[idx]

        if fignames:
            ax = plt.subplot(4, 2, 2 * i + 1)
            _ = plt.hist(d_pt, bins=100, range=(-100, 100))
            plt.xlabel("truth pt - reco pt [GeV]")
            plt.text(0.05, 0.95, f"pT: [{low}, {high}] GeV", transform=ax.transAxes)

        sigma_pt_q1q3[i] = _sigma_from_q1q3(d_pt)
        sigma_qopt_q1q3[i] = _sigma_from_q1q3(d_pt_qopt)
        sigma_pt_std[i] = d_pt.std()
        sigma_qopt_std[i] = d_pt_qopt.std()
        sigma_pt_fit[i] = _sigma_from_fit(d_pt, fignames=fignames + f".{i}.png")

        if verbose > 0:
            print(f"pt = ({low} ~ {high})")
            print(f"  sigma (Q1/Q3) = {sigma_pt_q1q3[-1]}")
            print(f"  std = {sigma_pt_std[-1]}")
            print(f"  fit = {sigma_pt_fit[-1]}")
            print(f"  q/pt sigma (Q1/Q3) = {sigma_qopt_q1q3[-1]}")
            print(f"  q/pt std = {sigma_qopt_std[-1]}")

    if fignames:
        plt.savefig(fignames, dpi=300)
        plt.close()

    return {
        "q1q3": sigma_pt_q1q3,
        "std": sigma_pt_std,
        "fit": sigma_pt_fit,
        "qoverpt_q1q3": sigma_qopt_q1q3,
        "qoverpt_std": sigma_qopt_std,
    }


def plot_charged_resolution(preds_pt, truth_pt, mask=None, resolution_type="q1q3", savedir="figs"):
    if mask is None:
        mask = np.ones(len(truth_pt), dtype=bool)

    xticks = [15, 20, 30, 50, 200]
    bin_pt = [(xticks[i], xticks[i + 1]) for i in range(len(xticks) - 1)]
    x = [(high + low) / 2 for low, high in bin_pt]
    xerr = [(high - low) / 2 for low, high in bin_pt]

    for k, pt in preds_pt.items():
        sigma = _calc_track_pt_sigma(
            pt[mask],
            truth_pt[mask],
            bin_pt,
            fignames=os.path.join(savedir, f"charged_residual_{k}.png"),
        )

        plt.errorbar(
            x=x,
            y=sigma[resolution_type],
            xerr=xerr,
            yerr=None,
            linestyle="None",
            marker="o",
            label=k,
        )

    plt.xscale("log")
    plt.xlim(xticks[0], xticks[-1])
    plt.xlabel("truth pT [GeV]")
    plt.ylabel("sigma pT [GeV]")
    plt.legend()
    plt.grid()
    plt.xticks(xticks)
    plt.yticks([5, 15, 25, 35, 45, 55, 65])
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())

    plt.savefig(os.path.join(savedir, "charged_resolution.png"), dpi=300)
    plt.close()


def plot_neutral_efficiency(
    class_pred, indicator_pred, class_truth, indicator_truth, truth_pt, mask=None, savedir="figs"
):
    if mask is None:
        mask = np.ones(len(truth_pt)).astype("bool")

    xticks = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]
    bin_pt = [(xticks[i], xticks[i + 1]) for i in range(len(xticks) - 1)]
    x = [(high + low) / 2 for low, high in bin_pt]
    xerr = [(high - low) / 2 for low, high in bin_pt]

    _ = plt.figure(figsize=(16, 6))

    for i_class, label in {0: "photon", 1: "neutral hadron"}.items():
        ind_denominator = np.logical_and.reduce([mask, class_truth == i_class])
        ind_numerator = np.logical_and.reduce(
            [class_pred <= 1, indicator_pred > 0.5, ind_denominator]
        )

        _ = plt.subplot(1, 2, 1 + i_class)
        eff = np.empty(len(bin_pt))
        efferr = np.empty(len(bin_pt))
        for i, (low, high) in enumerate(bin_pt):
            ind_pt = np.logical_and(truth_pt > low, truth_pt < high)
            numerator = np.logical_and(ind_numerator, ind_pt).sum()
            denominator = np.logical_and(ind_denominator, ind_pt).sum()
            eff[i] = numerator / (denominator + 1e-15)
            efferr[i] = np.sqrt(eff[i] * (1 - eff[i]) / (denominator + 1e-15))

        plt.errorbar(x=x, y=eff, xerr=xerr, yerr=efferr, linestyle="None", marker="o")
        plt.xlim(bin_pt[0][0], bin_pt[-1][1])
        _ = plt.xlabel("truth pT [GeV]")
        _ = plt.ylabel(f"efficiency ({label})")
        _ = plt.grid()
        _ = plt.xticks([1, 5, 10, 20, 30, 50])
        _ = plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    plt.savefig(os.path.join(savedir, "neutral_efficiency.png"), dpi=300)
    plt.close()


def plot_neutral_fakerate(
    class_pred, indicator_pred, class_truth, indicator_truth, pflow_pt, mask=None, savedir="figs"
):
    if mask is None:
        mask = np.ones(len(pflow_pt)).astype("bool")

    xticks = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]
    bin_pt = [(xticks[i], xticks[i + 1]) for i in range(len(xticks) - 1)]
    x = [(high + low) / 2 for low, high in bin_pt]
    xerr = [(high - low) / 2 for low, high in bin_pt]

    _ = plt.figure(figsize=(16, 6))

    for i_class, label in {0: "photon", 1: "neutral hadron"}.items():
        ind_denominator = np.logical_and.reduce(
            [
                mask,
                class_pred == i_class,
                indicator_pred > 0.5,
            ]
        )
        ind_numerator = np.logical_and.reduce([class_truth != i_class, ind_denominator])

        _ = plt.subplot(1, 2, 1 + i_class)
        eff = np.empty(len(bin_pt))
        efferr = np.empty(len(bin_pt))
        for i, (low, high) in enumerate(bin_pt):
            ind_pt = np.logical_and(pflow_pt > low, pflow_pt < high)
            numerator = np.logical_and(ind_numerator, ind_pt).sum()
            denominator = np.logical_and(ind_denominator, ind_pt).sum()
            if numerator == 0 and denominator == 0:
                eff[i] = 0.0
                efferr[i] = 0.0
            else:
                eff[i] = numerator / denominator
                efferr[i] = np.sqrt(eff[i] * (1 - eff[i]) / denominator) + 1e-15

        plt.errorbar(x=x, y=eff, xerr=xerr, yerr=efferr, linestyle="None", marker="o")
        plt.xlim(bin_pt[0][0], bin_pt[-1][1])
        _ = plt.xlabel("reco pT [GeV]")
        _ = plt.ylabel(f"fake rate ({label})")
        _ = plt.grid()
        _ = plt.xticks([1, 5, 10, 20, 30, 50])
        _ = plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    plt.savefig(os.path.join(savedir, "neutral_fakerate.png"), dpi=300)
    plt.close()


def plot_neutral_probability(
    class_pred, indicator_pred, class_truth, indicator_truth, truth_pt, mask=None, savedir="figs"
):
    if mask is None:
        mask = np.ones(len(truth_pt)).astype("bool")

    xticks = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]
    bin_pt = [(xticks[i], xticks[i + 1]) for i in range(len(xticks) - 1)]
    x = [(high + low) / 2 for low, high in bin_pt]
    xerr = [(high - low) / 2 for low, high in bin_pt]

    _ = plt.figure(figsize=(16, 6))

    for i_class, label in {0: "photon", 1: "neutral hadron"}.items():
        ind_denominator = np.logical_and.reduce([mask, class_truth == i_class])
        ind_numerator = np.logical_and.reduce(
            [class_pred == i_class, indicator_pred > 0.5, ind_denominator]
        )

        _ = plt.subplot(1, 2, 1 + i_class)
        eff = np.empty(len(bin_pt))
        efferr = np.empty(len(bin_pt))
        for i, (low, high) in enumerate(bin_pt):
            ind_pt = np.logical_and(truth_pt > low, truth_pt < high)
            numerator = np.logical_and(ind_numerator, ind_pt).sum()
            denominator = np.logical_and(ind_denominator, ind_pt).sum()
            eff[i] = numerator / (denominator + 1e-15)
            efferr[i] = np.sqrt(eff[i] * (1 - eff[i]) / (denominator + 1e-15))

        plt.errorbar(x=x, y=eff, xerr=xerr, yerr=efferr, linestyle="None", marker="o")
        plt.xlim(bin_pt[0][0], bin_pt[-1][1])
        _ = plt.xlabel("truth pT [GeV]")
        _ = plt.ylabel(f"P(reco {label}| truth {label})")
        _ = plt.grid()
        _ = plt.xticks([1, 5, 10, 20, 30, 50])
        _ = plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    plt.savefig(os.path.join(savedir, "neutral_probability.png"), dpi=300)
    plt.close()


def plot_neutral_kinematics(
    pflow_pt,
    pflow_eta,
    pflow_phi,
    truth_pt,
    truth_eta,
    truth_phi,
    mask=None,
    savedir="figs",
):
    def _custom_hist(data, bins, range):
        c, edges = np.histogram(data, bins=bins, range=range)
        _ = plt.step(
            edges[:-1],
            c / c.sum(),
            where="mid",
        )

    if mask is None:
        mask = np.ones(len(truth_pt)).astype("bool")

    pflow_pt = pflow_pt[mask]
    pflow_eta = pflow_eta[mask]
    pflow_phi = pflow_phi[mask]
    truth_pt = truth_pt[mask]
    truth_eta = truth_eta[mask]
    truth_phi = truth_phi[mask]

    pflow_phi = np.where(pflow_phi > np.pi, pflow_phi - 2 * np.pi, pflow_phi)
    pflow_phi = np.where(pflow_phi < -np.pi, pflow_phi + 2 * np.pi, pflow_phi)

    pt_res = (truth_pt - pflow_pt) / truth_pt
    logpt_res = np.log(truth_pt) - np.log(pflow_pt)
    eta_res = truth_eta - pflow_eta
    phi_res = np.mod(truth_phi - pflow_phi + np.pi, 2 * np.pi) - np.pi

    _ = plt.figure(figsize=(8, 20))

    _ = plt.subplot(4, 1, 1)
    nbins = 41
    _custom_hist(pt_res, bins=nbins, range=(-1.0, 1.0))
    _ = plt.xlabel("rel. res. Neutral Particles pT")
    _ = plt.ylabel("arb. unit")
    _ = plt.grid()
    _ = plt.xticks([-1 + 0.5 * i for i in range(5)])

    _ = plt.subplot(4, 1, 2)
    _custom_hist(logpt_res, bins=nbins, range=(-1.0, 1.0))
    _ = plt.xlabel("res. Neutral Particles log(pT)")
    _ = plt.ylabel("arb. unit")
    _ = plt.grid()

    _ = plt.subplot(4, 1, 3)
    _custom_hist(eta_res, bins=nbins, range=(-0.4, 0.4))
    _ = plt.xlabel("Neutral Particles delta eta")
    _ = plt.ylabel("arb. unit")
    _ = plt.grid()
    _ = plt.xticks([-0.4 + 0.1 * i for i in range(9)])

    _ = plt.subplot(4, 1, 4)
    _custom_hist(phi_res, bins=nbins, range=(-0.4, 0.4))
    _ = plt.xlabel("Neutral Particles delta phi")
    _ = plt.ylabel("arb. unit")
    _ = plt.grid()
    _ = plt.xticks([-0.4 + 0.1 * i for i in range(9)])

    plt.savefig(os.path.join(savedir, "neutral_kinematic.png"), dpi=300)
    plt.close()

    _ = plt.figure(figsize=(8, 20))

    _ = plt.subplot(3, 1, 1)
    _ = plt.hist2d(
        np.log(pflow_pt),
        np.log(truth_pt),
        bins=[50, 50],
        range=[(-1.0, 5.0), (-1.0, 5.0)],
        density=True,
        norm=mpl.colors.LogNorm(),
    )
    _ = plt.xlabel("predicted log(pt)")
    _ = plt.ylabel("truth log(pt)")
    _ = plt.grid()

    _ = plt.subplot(3, 1, 2)
    _ = plt.hist2d(
        pflow_eta,
        truth_eta,
        bins=[50, 50],
        range=[(-4, 4), (-4, 4)],
        density=True,
        norm=mpl.colors.LogNorm(),
    )
    _ = plt.xlabel("predicted eta")
    _ = plt.ylabel("truth eta")
    _ = plt.grid()

    _ = plt.subplot(3, 1, 3)
    _ = plt.hist2d(
        pflow_phi,
        truth_phi,
        bins=[50, 50],
        range=[(-4, 4), (-4, 4)],
        density=True,
        norm=mpl.colors.LogNorm(),
    )
    _ = plt.xlabel("predicted phi")
    _ = plt.ylabel("truth phi")
    _ = plt.grid()

    plt.savefig(os.path.join(savedir, "neutral_kinematic_2d.png"), dpi=300)
    plt.close()


def plot_indicator(
    pred_indicator,
    truth_indicator,
    mask=None,
    savedir="figs",
):
    if mask is None:
        mask = np.ones(len(pred_indicator)).astype("bool")

    _ = plt.figure(figsize=(8, 28))

    hist_kwargs = dict(bins=100, range=(0.0, 1.0), density=True)

    _ = plt.subplot(4, 1, 1)
    _ = plt.hist(pred_indicator[mask], **hist_kwargs)
    _ = plt.xlabel("predicted indicator score")
    _ = plt.ylabel("arb. unit")
    _ = plt.yscale("log")
    _ = plt.grid()

    _ = plt.subplot(4, 1, 2)
    _ = plt.hist(truth_indicator[mask], **hist_kwargs)
    _ = plt.xlabel("truth indicator score")
    _ = plt.ylabel("arb. unit")
    _ = plt.grid()

    _ = plt.subplot(4, 1, 3)
    _ = plt.hist2d(
        truth_indicator[mask],
        pred_indicator[mask],
        bins=[50, 50],
        range=[(0.0, 1.0), (0.0, 1.0)],
        density=True,
        norm=mpl.colors.LogNorm(),
    )
    _ = plt.xlabel("truth indicator score")
    _ = plt.ylabel("pred indicator score")
    _ = plt.grid()

    _ = plt.subplot(4, 1, 4)
    mask_0 = np.logical_and(mask, truth_indicator < 0.5)
    mask_1 = np.logical_and(mask, truth_indicator > 0.5)
    _ = plt.hist(pred_indicator[mask_0], label="truth = 1", histtype="step", **hist_kwargs)
    _ = plt.hist(pred_indicator[mask_1], label="truth = 0", histtype="step", **hist_kwargs)
    _ = plt.xlabel("predicted indicator score")
    _ = plt.ylabel("arb. unit")
    _ = plt.yscale("log")
    _ = plt.legend()
    _ = plt.grid()

    plt.savefig(os.path.join(savedir, "indicator.png"), dpi=300)
    plt.close()


def plot_incidence_matrix(
    pred_incidence,
    truth_incidence,
    mask=None,
    savedir="figs",
):
    if mask is None:
        mask = np.ones(len(pred_incidence)).astype("bool")

    # _ = plt.figure(figsize=(8, 20))
    _ = plt.figure(figsize=(8, 28))

    hist_kwargs = dict(bins=100, range=(0.0, 1.0), density=True)

    _ = plt.subplot(4, 1, 1)
    _ = plt.hist(pred_incidence[mask], **hist_kwargs)
    _ = plt.xlabel("predicted incidence value")
    _ = plt.ylabel("arb. unit")
    _ = plt.yscale("log")
    _ = plt.grid()

    _ = plt.subplot(4, 1, 2)
    _ = plt.hist(truth_incidence[mask], **hist_kwargs)
    _ = plt.xlabel("truth incidence value")
    _ = plt.ylabel("arb. unit")
    _ = plt.yscale("log")
    _ = plt.grid()

    _ = plt.subplot(4, 1, 3)
    _ = plt.hist2d(
        truth_incidence[mask],
        pred_incidence[mask],
        bins=[50, 50],
        range=[(0.0, 1.0), (0.0, 1.0)],
        density=True,
        norm=mpl.colors.LogNorm(),
    )
    _ = plt.xlabel("truth incidence value")
    _ = plt.ylabel("pred incidence value")
    _ = plt.grid()

    _ = plt.subplot(4, 1, 4)
    mask_0 = np.logical_and.reduce([mask, truth_incidence < 0.05])
    mask_1 = np.logical_and.reduce([mask, truth_incidence >= 0.05, truth_incidence < 0.2])
    mask_2 = np.logical_and.reduce([mask, truth_incidence >= 0.2, truth_incidence < 0.8])
    mask_3 = np.logical_and.reduce([mask, truth_incidence >= 0.8, truth_incidence < 0.95])
    mask_4 = np.logical_and.reduce([mask, truth_incidence >= 0.95])
    _ = plt.hist(pred_incidence[mask_0], label="truth:(0, 0.05)", histtype="step", **hist_kwargs)
    _ = plt.hist(pred_incidence[mask_1], label="truth:(0.05, 0.2)", histtype="step", **hist_kwargs)
    _ = plt.hist(pred_incidence[mask_2], label="truth:(0.2, 0.8)", histtype="step", **hist_kwargs)
    _ = plt.hist(pred_incidence[mask_3], label="truth:(0.8, 0.95)", histtype="step", **hist_kwargs)
    _ = plt.hist(pred_incidence[mask_4], label="truth:(0.95, 1.0)", histtype="step", **hist_kwargs)
    _ = plt.xlabel("predicted incidence value")
    _ = plt.ylabel("arb. unit")
    _ = plt.yscale("log")
    _ = plt.legend()
    _ = plt.grid()

    plt.savefig(os.path.join(savedir, "incidence.png"), dpi=300)
    plt.close()


class DataReader:
    def __init__(self, filename, matching="original"):
        # ================= #
        # === Load Data === #
        # ================= #
        data = np.load(filename)

        # ================= #
        # === Variables === #
        # ================= #
        self.truth_class = data["truth_class"].flatten()
        self.pflow_class = data["pflow_class"].flatten()

        self.is_not_dummy = self.truth_class != -999.0

        self.is_charged = (data["truth_has_track"] == 1).flatten()
        self.is_neutral = np.logical_not(self.is_charged)

        track_pt = (
            data["node_pt"][:, np.newaxis, :]
            * data["node_is_track"][:, np.newaxis, :]
            * data["truth_inc"]
        ).sum(axis=2)
        self.track_pt = track_pt.flatten() / 1000

        self.truth_pt = data["truth_pt"].flatten() / 1000
        self.pflow_pt = data["pflow_pt"].flatten() / 1000
        self.pflow_eta = data["pflow_eta"].flatten()
        self.truth_eta = data["truth_eta"].flatten()
        self.pflow_phi = data["pflow_phi"].flatten()
        self.truth_phi = data["truth_phi"].flatten()

        if matching == "custom":
            indices_sort = self.custom_matching(data, self.is_not_dummy)

            self.pflow_class = self.pflow_class[indices_sort]
            self.track_pt = self.track_pt[indices_sort]
            self.pflow_pt = self.pflow_pt[indices_sort]
            self.pflow_eta = self.pflow_eta[indices_sort]
            self.pflow_phi = self.pflow_phi[indices_sort]
        elif matching == "original":
            pass
        else:
            raise NotImplementedError()

        self.pflow_phi = np.where(
            self.pflow_phi > np.pi, self.pflow_phi - 2 * np.pi, self.pflow_phi
        )
        self.pflow_phi = np.where(
            self.pflow_phi < -np.pi, self.pflow_phi + 2 * np.pi, self.pflow_phi
        )

        # ==================================== #
        # === Incidence matrix / Indicator === #
        # ==================================== #
        def _extract_incidence_matrix(incidence_matrix, num_nodes):
            return [x[:, :n] for x, n in zip(incidence_matrix, num_nodes)]

        pred_indicator = data["pred_ind"]
        truth_indicator = data["truth_ind"]
        self.pred_indicator = pred_indicator.flatten()
        self.truth_indicator = truth_indicator.flatten()

        pred_incidence = _extract_incidence_matrix(data["pred_inc"], data["num_nodes"])
        truth_incidence = _extract_incidence_matrix(data["truth_inc"], data["num_nodes"])
        pred_incidence = np.concatenate([v.flatten() for v in pred_incidence], axis=-1)
        truth_incidence = np.concatenate([v.flatten() for v in truth_incidence], axis=-1)
        self.pred_incidence = pred_incidence.flatten()
        self.truth_incidence = truth_incidence.flatten()

        is_true_particle = self.is_truth_objects().reshape((-1, 30))
        is_true_particle = np.repeat(is_true_particle, data["num_nodes"], axis=0).flatten()
        self.is_true_particle = is_true_particle

    def custom_matching(selfm, data, is_not_dummy):
        indices_sort = np.empty(is_not_dummy.shape, dtype=np.int32)

        for ie, (pt_t, pt_r, eta_t, eta_r, phi_t, phi_r, ind_t, ind_r) in enumerate(
            zip(
                data["truth_pt"],
                data["pflow_pt"],
                data["truth_eta"],
                data["pflow_eta"],
                data["truth_phi"],
                data["pflow_phi"],
                data["truth_ind"],
                data["pred_ind"],
            )
        ):
            pt_r = np.tile(pt_r, (30, 1))
            pt_t = np.tile(pt_t.reshape(-1, 1), (1, 30))
            eta_r = np.tile(eta_r, (30, 1))
            eta_t = np.tile(eta_t.reshape(-1, 1), (1, 30))
            phi_r = np.tile(phi_r, (30, 1))
            phi_t = np.tile(phi_t.reshape(-1, 1), (1, 30))
            delta_pt = (pt_t - pt_r) / pt_t
            delta_eta = eta_t - eta_r
            delta_phi = np.mod(phi_t - phi_r + np.pi, 2 * np.pi) - np.pi
            d2 = delta_pt**2 + 25 * (delta_eta**2 + delta_phi**2)

            d2 = np.where(np.tile(ind_r.flatten(), (30, 1)) > 0.5, d2, 1e10)
            d2 = np.where(np.tile(ind_t.reshape(-1, 1), (1, 30)) > 0.5, d2, 1e10)

            indices = linear_sum_assignment(d2)[1]
            shift = 30 * ie
            indices_sort[shift : shift + 30] = indices + shift

        return indices_sort

    def mask_is_charged(self):
        mask = np.logical_and(self.is_charged, self.is_not_dummy)

        # Remove the events s.t. track pt = 0
        mask = np.logical_and(mask, self.track_pt != 0.0)

        return mask

    def is_truth_objects(self, classid=None):
        if classid is None:
            return np.logical_and(self.is_not_dummy, self.truth_indicator > 0.5)
        else:
            return np.logical_and(self.is_truth_objects(), self.truth_class == classid)

    def is_pflow_objects(self, classid=None):
        if classid is None:
            return np.logical_and(self.is_not_dummy, self.pred_indicator > 0.5)
        else:
            return np.logical_and(self.is_pflow_objects(), self.pflow_class == classid)


@click.command()
@click.option("--inputfile", type=str, default="./workspace/eval/v1004.npz")
@click.option("--outputdir", type=str, default="./workspace/figs/visualize")
@click.option("--original_matching", is_flag=True)
def main(**args):
    os.makedirs(args["outputdir"], exist_ok=True)

    if args["original_matching"]:
        data = DataReader(args["inputfile"], matching="original")
    else:
        data = DataReader(args["inputfile"], matching="custom")

    plot_indicator(
        data.pred_indicator,
        data.truth_indicator,
        mask=None,
        savedir=args["outputdir"],
    )

    plot_incidence_matrix(
        data.pred_incidence,
        data.truth_incidence,
        mask=data.is_true_particle,
        savedir=args["outputdir"],
    )

    # ==================================== #
    # === Charged particles resolution === #
    # ==================================== #
    plot_charged_resolution(
        preds_pt={"pflow": data.pflow_pt, "track": data.track_pt},
        truth_pt=data.truth_pt,
        mask=data.mask_is_charged(),
        resolution_type="fit",
        savedir=args["outputdir"],
    )

    # =============== #
    # === Neutral === #
    # =============== #

    print("Neutral particle statistics")
    print("Truth photon        : ", data.is_truth_objects(classid=0).sum())
    print("Truth neutral hadron: ", data.is_truth_objects(classid=1).sum())
    print("Pred photon         : ", data.is_pflow_objects(classid=0).sum())
    print("Pred neutral hadron : ", data.is_pflow_objects(classid=1).sum())
    print(
        "TP: ",
        np.logical_and(data.is_truth_objects(classid=0), data.is_pflow_objects(classid=0)).sum(),
    )
    print(
        "FP: ",
        np.logical_and(data.is_truth_objects(classid=0), data.is_pflow_objects(classid=1)).sum(),
    )
    print(
        "FN: ",
        np.logical_and(data.is_truth_objects(classid=1), data.is_pflow_objects(classid=0)).sum(),
    )
    print(
        "TN: ",
        np.logical_and(data.is_truth_objects(classid=1), data.is_pflow_objects(classid=1)).sum(),
    )

    # ================== #
    # === Efficiency === #
    # ================== #
    plot_neutral_efficiency(
        data.pflow_class,
        data.pred_indicator,
        data.truth_class,
        data.truth_indicator,
        data.truth_pt,
        mask=data.is_truth_objects(),
        savedir=args["outputdir"],
    )

    # ============ #
    # === Fake === #
    # ============ #
    plot_neutral_fakerate(
        data.pflow_class,
        data.pred_indicator,
        data.truth_class,
        data.truth_indicator,
        data.pflow_pt,
        mask=data.is_pflow_objects(),
        savedir=args["outputdir"],
    )

    # =================== #
    # === Probability === #
    # =================== #
    plot_neutral_probability(
        data.pflow_class,
        data.pred_indicator,
        data.truth_class,
        data.truth_indicator,
        data.truth_pt,
        mask=data.is_truth_objects(),
        savedir=args["outputdir"],
    )

    # ================== #
    # === Kinematics === #
    # ================== #

    ind_selection = np.logical_and.reduce(
        [data.is_neutral, data.is_truth_objects(), data.is_pflow_objects()]
    )

    plot_neutral_kinematics(
        data.pflow_pt,
        data.pflow_eta,
        data.pflow_phi,
        data.truth_pt,
        data.truth_eta,
        data.truth_phi,
        mask=ind_selection,
        savedir=args["outputdir"],
    )


if __name__ == "__main__":
    main()
