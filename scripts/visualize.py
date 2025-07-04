import os
import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
from sklearn.metrics import precision_recall_curve, average_precision_score

def _calc_track_pt_sigma(pt, ref_pt, bin_pt, fignames=None, verbose=0):
    """
    Calcule la résolution de pT d'une trace en différents bins de vérité.
    Retourne un dict avec les sigmas calculées selon Q1/Q3, std et fit Gaussienne.
    """
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
                sigma=np.sqrt(hist + 1e-15),
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
            plt.figure(figsize=(8, 6))
            xmin, xmax = -50.0, 50.0
            plt.hist(x, bins=100, range=(xmin, xmax), label="data")
            x_data1 = np.linspace(xmin1, xmax1, 100)
            norm1 = C1 * ((xmax - xmin) / 100) / ((xmax1 - xmin1) / 20)
            plt.plot(x_data1, _gauss(x_data1, norm1, mean1, sigma1), label="1st fit")
            x_data2 = np.linspace(xmin2, xmax2, 100)
            norm2 = C2 * ((xmax - xmin) / 100) / (4 * sigma1 / 20)
            plt.plot(x_data2, _gauss(x_data2, norm2, mean2, sigma2), label="2nd fit")
            plt.legend()
            plt.savefig(fignames, dpi=300)
            plt.close()

        return sigma2

    # Initialisation des tableaux de sigma
    sigma_pt_q1q3 = np.empty(len(bin_pt))
    sigma_pt_std = np.empty(len(bin_pt))
    sigma_pt_fit = np.empty(len(bin_pt))

    if fignames:
        plt.figure(figsize=(12, 24))

    for i, (low, high) in enumerate(bin_pt):
        idx = np.logical_and(ref_pt > low, ref_pt < high)
        d_pt = (ref_pt - pt)[idx]
        # Dessin de l'histogramme résiduel
        if fignames:
            ax = plt.subplot(4, 2, 2 * i + 1)
            plt.hist(d_pt, bins=100, range=(-100, 100))
            plt.xlabel("truth pt - reco pt [GeV]")
            plt.text(0.05, 0.95, f"pT: [{low}, {high}] GeV", transform=ax.transAxes)

        sigma_pt_q1q3[i] = _sigma_from_q1q3(d_pt)
        sigma_pt_std[i] = d_pt.std()
        sigma_pt_fit[i] = _sigma_from_fit(d_pt, fignames=fignames + f".bin{i}.png")

        if verbose > 0:
            print(f"pt = ({low} ~ {high})")
            print(f"  sigma (Q1/Q3) = {sigma_pt_q1q3[-1]}")
            print(f"  std = {sigma_pt_std[-1]}")
            print(f"  fit = {sigma_pt_fit[-1]}")

    if fignames:
        plt.savefig(fignames, dpi=300)
        plt.close()

    return {
        "q1q3": sigma_pt_q1q3,
        "std": sigma_pt_std,
        "fit": sigma_pt_fit,
    }


class DataReader:
    def __init__(self, filename):
        data = np.load(filename)
        self.truth_labels = data['truth_labels']  # (n_events, n_truth)
        self.truth_boxes = data['truth_boxes']    # (..., 3) (pt, eta, phi)
        
        self.pred_logits = data['pred_logits']  # (n_events, n_queries, n_classes)
        self.pred_boxes = data['pred_boxes']    # (..., 3)
        self.pred_classes = data['pred_class']  # (n_events, n_queries)
        # Check if the boxes have already been denormalized during evaluation
        # If not, we need to load the normalization parameters and denormalize here
        is_denormalized = data.get('is_denormalized', False)
        if is_denormalized:
            print("Data has already been denormalized during evaluation, skipping denormalization.")
        else:
            print("[ERROR]Data needs to be denormalized...")

        self.pred_classes_flat = self.pred_classes.flatten()
        self.truth_labels_flat = self.truth_labels.flatten()

        self.truth_pt = self.truth_boxes[...,0].flatten() / 1000  # GeV
        self.pred_pt = self.pred_boxes[...,0].flatten() / 1000  # GeV

        self.truth_eta = self.truth_boxes[...,1].flatten()
        self.pred_eta = self.pred_boxes[...,1].flatten()
        self.truth_phi = self.truth_boxes[...,2].flatten()
        self.pred_phi = self.pred_boxes[...,2].flatten()
        print('pred_pt:', self.pred_pt.shape, 'NaN:', np.isnan(self.pred_pt).sum(), 'Inf:', np.isinf(self.pred_pt).sum())

def plot_pr_curves(truth_labels_flat,
                   pred_logits,
                   class_map,
                   savedir='figs',
                   n_thresh=5):
    """
    Plot Precision-Recall curves with threshold (epsilon) annotations,
    and compute AP for each class and the mean AP.
    
    Parameters:
    - truth_labels_flat: array-like of shape (n_samples,)
        Ground truth integer labels.
    - pred_logits: array-like of shape (n_samples, n_classes)
        Raw model logits for each class.
    - class_map: dict
        Mapping from class names to list of label integers, e.g. {"charged":[0,1], ...}.
    - savedir: str
        Directory in which to save the figure.
    - n_thresh: int
        How many thresholds (evenly spaced) to annotate on each curve.
    """
    # Compute probabilities via softmax
    logits = np.array(pred_logits)
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    
    ap_scores = {}
    plt.figure(figsize=(8, 6))
    
    for cls_name, labels in class_map.items():
        # Build binary truth mask for this class
        truth_binary = np.isin(truth_labels_flat, labels).astype(int)
        # Sum probabilities for multi-label classes
        class_prob = probs[:, labels].sum(axis=1)
        
        # Compute Precision-Recall and thresholds
        precision, recall, thresholds = precision_recall_curve(truth_binary, class_prob)
        ap = average_precision_score(truth_binary, class_prob)
        ap_scores[cls_name] = ap
        
        # Plot the PR curve
        plt.step(recall, precision, where='post', label=f"{cls_name} (AP={ap:.3f})")
        
        # Annotate thresholds: skip the first entry (no threshold) and align
        prec = precision[1:]
        rec  = recall[1:]
        thr  = thresholds
        N = len(thr)
        # pick evenly spaced indices
        idxs = np.linspace(0, N-1, min(n_thresh, N), dtype=int)
        for i in idxs:
            x = rec[i]
            y = prec[i]
            label = f"{thr[i]:.2f}"
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, -5),
                         textcoords='offset points',
                         fontsize=7,
                         arrowprops=dict(arrowstyle='->', lw=0.5))
    
    # Mean AP
    mean_ap = np.mean(list(ap_scores.values()))
    ap_scores['mean'] = mean_ap
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves by Class with ε Annotations")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(savedir, exist_ok=True)
    plt.savefig(os.path.join(savedir, "pr_curves_with_eps.png"))
    plt.close()
    
    # Print AP scores
    print("Average Precision (AP) scores:")
    for cls_name, ap in ap_scores.items():
        print(f"  {cls_name:15s}: {ap:.4f}")

def plot_neutral_efficiency(class_pred, class_truth, truth_pt, mask=None, savedir='figs'):
    """
    Taux de détection pour classes neutres (photon=4, hadron=3),
    avec exactement la même logique de binning et d'erreurs que le code 1.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    if mask is None:
        mask = np.ones(len(truth_pt), dtype=bool)

    # bornes de pT et calcul des x, xerr
    xticks = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]
    bin_pt = [(xticks[i], xticks[i+1]) for i in range(len(xticks)-1)]
    x = [(low + high)/2 for low, high in bin_pt]
    xerr = [(high - low)/2 for low, high in bin_pt]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    class_labels = {4: 'photon', 3: 'neutral hadron'}

    for idx, (i_class, label) in enumerate(class_labels.items()):
        # masque global des véritables i_class
        ind_den = np.logical_and(mask, class_truth == i_class)
        # masque des prédictions correctes pour i_class
        ind_num = np.logical_and(class_pred == i_class, ind_den)

        # tableaux d'efficacité et d'erreur
        eff = np.empty(len(bin_pt))
        efferr = np.empty(len(bin_pt))

        for j, (low, high) in enumerate(bin_pt):
            ind_bin = np.logical_and(truth_pt > low, truth_pt < high)

            n_den = np.count_nonzero(ind_den & ind_bin)
            n_num = np.count_nonzero(ind_num & ind_bin)

            # on ajoute un epsilon pour éviter la division par zéro
            eff[j] = n_num / (n_den + 1e-15)
            efferr[j] = np.sqrt(eff[j] * (1 - eff[j]) / (n_den + 1e-15))

        ax = axes[idx]
        ax.errorbar(x, eff, xerr=xerr, yerr=efferr,
                    marker='o', linestyle='None', label=label)
        ax.set_title(label)
        ax.set_xlabel('truth pT [GeV]')
        ax.set_xticks(xticks)
        ax.set_xlim(xticks[0], xticks[-1])
        ax.grid()
        ax.legend()

    axes[0].set_ylabel('efficiency')
    os.makedirs(savedir, exist_ok=True)
    plt.savefig(os.path.join(savedir, 'neutral_efficiency.png'), dpi=300)
    plt.close()

def plot_ncharged_efficiency(class_pred, class_truth, truth_pt, mask=None, savedir='figs'):
    """
    Taux de détection pour particules chargées (charged hadron=0, electron=1).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if mask is None:
        mask = np.ones(len(truth_pt), dtype=bool)

    xticks = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]
    bin_pt = [(xticks[i], xticks[i+1]) for i in range(len(xticks)-1)]
    x = [(low + high)/2 for low, high in bin_pt]
    xerr = [(high - low)/2 for low, high in bin_pt]

    plt.figure(figsize=(10,5))
    for i_class, label in {0:'charged hadron', 1:'electron'}.items():
        denom = np.logical_and(mask, class_truth==i_class)
        num = np.logical_and(denom, class_pred==i_class)
        eff = np.array([ num[np.logical_and(truth_pt>low, truth_pt<high)].sum()/\
                         max(1, denom[np.logical_and(truth_pt>low, truth_pt<high)].sum())
                         for low, high in bin_pt ])
        err = np.sqrt(eff*(1-eff)/np.maximum(1, denom.sum()))
        plt.errorbar(x, eff, xerr=xerr, yerr=err, marker='o', linestyle='None', label=label)

    plt.xscale('log')
    plt.xlabel('truth pT [GeV]')
    plt.ylabel('efficiency')
    plt.xticks(xticks)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(savedir, 'charged_efficiency.png'), dpi=300)
    plt.close()

def plot_neutral_fakerate(class_pred, class_truth, pflow_pt, mask=None, savedir='figs'):
    """
    Fake rate pour classes neutres (photon=4, hadron=3),
    exactement comme dans le code 1 (sans indicator_pred),
    en binning sur pflow_pt et erreurs bin-par-bin.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    if mask is None:
        mask = np.ones(len(pflow_pt), dtype=bool)

    # bornes de pT reconstruit et calcul des x, xerr
    xticks = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]
    bin_pt = [(xticks[i], xticks[i+1]) for i in range(len(xticks)-1)]
    x = [(low + high)/2 for low, high in bin_pt]
    xerr = [(high - low)/2 for low, high in bin_pt]

    # pré-création du dossier de sortie
    os.makedirs(savedir, exist_ok=True)

    plt.figure(figsize=(16, 6))
    class_labels = {4: 'photon', 3: 'neutral hadron'}

    for i_class, label in class_labels.items():
        # dénominateur : toutes les prédictions de la classe i_class
        ind_den = np.logical_and(mask, class_pred == i_class)
        # numérateur : parmi ces prédictions, celles qui sont en fait d'une autre classe
        ind_num = np.logical_and(ind_den, class_truth != i_class)

        fakerate = np.empty(len(bin_pt))
        fakerate_err = np.empty(len(bin_pt))

        for i, (low, high) in enumerate(bin_pt):
            ind_pt = np.logical_and(pflow_pt > low, pflow_pt < high)
            n_den = np.count_nonzero(ind_den & ind_pt)
            n_num = np.count_nonzero(ind_num & ind_pt)

            if n_den == 0 and n_num == 0:
                fakerate[i] = 0.0
                fakerate_err[i] = 0.0
            else:
                fakerate[i] = n_num / n_den
                # erreur bin par bin + petit epsilon comme dans le code 1
                fakerate_err[i] = np.sqrt(fakerate[i] * (1 - fakerate[i]) / n_den) + 1e-15

        # tracé
        ax = plt.subplot(1, 2, 1 + (0 if i_class == 4 else 1))
        ax.errorbar(
            x, fakerate, xerr=xerr, yerr=fakerate_err,
            linestyle='None', marker='o'
        )
        ax.set_xlim(bin_pt[0][0], bin_pt[-1][1])
        ax.set_xlabel('reco pT [GeV]')
        ax.set_ylabel(f'fake rate ({label})')
        ax.grid()
        ax.set_xticks([1, 5, 10, 20, 30, 50])
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    plt.savefig(os.path.join(savedir, 'neutral_fakerate.png'), dpi=300)
    plt.close()

def plot_ncharged_fakerate(class_pred, class_truth, pred_pt, mask=None, savedir='figs'):
    """
    Fake rate pour particules chargées (charged hadron=0, electron=1).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if mask is None:
        mask = np.ones(len(pred_pt), dtype=bool)

    xticks = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]
    bin_pt = [(xticks[i], xticks[i+1]) for i in range(len(xticks)-1)]
    x = [(low + high)/2 for low, high in bin_pt]
    xerr = [(high - low)/2 for low, high in bin_pt]

    plt.figure(figsize=(10,5))
    for i_class, label in {0:'charged hadron', 1:'electron'}.items():
        denom = np.logical_and(mask, class_pred==i_class)
        num = np.logical_and(denom, class_truth!=i_class)
        fakerate = np.array([ num[np.logical_and(pred_pt>low, pred_pt<high)].sum()/\
                         max(1, denom[np.logical_and(pred_pt>low, pred_pt<high)].sum())
                         for low, high in bin_pt ])
        err = np.sqrt(fakerate*(1-fakerate)/np.maximum(1, denom.sum()))
        plt.errorbar(x, fakerate, xerr=xerr, yerr=err, marker='o', linestyle='None', label=label)

    plt.xscale('log')
    plt.xlabel('predicted pT [GeV]')
    plt.ylabel('fake rate')
    plt.xticks(xticks)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(savedir, 'charged_fakerate.png'), dpi=300)
    plt.close()

def plot_neutral_probability(
    class_pred, class_truth, truth_pt, mask=None, savedir="figs"
):
    """
    P(reco i | truth i) pour i = photon (0) ou neutral hadron (1),
    sans utiliser indicator_pred.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    if mask is None:
        mask = np.ones(len(truth_pt), dtype=bool)

    # bornes de pT et calcul des points x, xerr
    xticks = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]
    bin_pt = [(xticks[i], xticks[i + 1]) for i in range(len(xticks) - 1)]
    x = [(low + high) / 2 for low, high in bin_pt]
    xerr = [(high - low) / 2 for low, high in bin_pt]

    # création du dossier de sortie si nécessaire
    os.makedirs(savedir, exist_ok=True)

    plt.figure(figsize=(16, 6))
    class_labels = {0: "photon", 1: "neutral hadron"}

    for i_class, label in class_labels.items():
        # dénominateur : tous les vrais i_class
        ind_den = np.logical_and(mask, class_truth == i_class)
        # numérateur : parmi ceux-ci, ceux prédits i_class
        ind_num = np.logical_and(ind_den, class_pred == i_class)

        eff = np.empty(len(bin_pt))
        efferr = np.empty(len(bin_pt))

        for j, (low, high) in enumerate(bin_pt):
            ind_bin = np.logical_and(truth_pt > low, truth_pt < high)
            n_den = np.count_nonzero(ind_den & ind_bin)
            n_num = np.count_nonzero(ind_num & ind_bin)

            # on évite la division par zéro
            eff[j] = n_num / (n_den + 1e-15)
            efferr[j] = np.sqrt(eff[j] * (1 - eff[j]) / (n_den + 1e-15))

        ax = plt.subplot(1, 2, 1 + i_class)
        ax.errorbar(x, eff, xerr=xerr, yerr=efferr,
                    linestyle="None", marker="o")
        ax.set_xlim(bin_pt[0][0], bin_pt[-1][1])
        ax.set_xlabel("truth pT [GeV]")
        ax.set_ylabel(f"P(reco {label} | truth {label})")
        ax.grid()
        ax.set_xticks([1, 5, 10, 20, 30, 50])
        ax.set_yticks(np.linspace(0, 1, 11))

    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "neutral_probability_no_indicator.png"), dpi=300)
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

def plot_charged_resolution(preds_pt, truth_pt, mask=None, resolution_type="q1q3", savedir="figs"):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
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

def plot_max_softmax_confidence(pred_logits, savedir="figs"):
    """
    Plot histogram of the maximum softmax probability (confidence) for all predictions.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    logits_flat = pred_logits.reshape(-1, pred_logits.shape[-1])
    exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    max_probs = np.max(probs, axis=-1)
    plt.figure()
    plt.hist(max_probs, bins=50, range=(0, 1), alpha=0.7, color='C0')
    plt.xlabel('Max softmax probability (confidence)')
    plt.ylabel('Count')
    plt.title('Histogram of DETR confidence (max softmax probability)')
    os.makedirs(savedir, exist_ok=True)
    plt.savefig(os.path.join(savedir, 'max_softmax_confidence.png'), dpi=300)
    plt.close()

@click.command()
@click.option('--inputfile', type=str, required=True)
@click.option('--outputdir', type=str, required=True)
@click.option('--conf-threshold', type=float, default=0.5, show_default=True, help='Confidence threshold: predictions below are set to class 5 (non-object)')
def main(inputfile, outputdir, conf_threshold):
    os.makedirs(outputdir, exist_ok=True)

    # --- Lecture des données et flattening
    data = DataReader(inputfile)
    truth = data.truth_labels_flat
    pred  = data.pred_classes_flat.copy()  # copie à écraser avec le seuil
    print(pred[:30])
    print(truth[:30])
    pt_truth = data.truth_pt
    pt_pred   = data.pred_pt
    print(pt_truth[:30])
    print(pt_pred[:30])
    # --- Scatter plot of per-event sum pT (pred vs truth)
    # Use per-event arrays from truth_boxes and pred_boxes
    
    pt_truth_per_event = data.truth_boxes[..., 0] / 1000  # shape: (n_events, n_truth_particles)
    pt_pred_per_event = data.pred_boxes[..., 0] / 1000    # shape: (n_events, n_pred_particles)

    # Mask class 5 (fake) particles
    truth_label_per_event = data.truth_labels  # shape: (n_events, n_truth_particles)
    pred_label_per_event = data.pred_classes   # shape: (n_events, n_pred_particles)
    valid_truth_mask = (truth_label_per_event != 5)
    valid_pred_mask = (pred_label_per_event != 5)

    # Set pT to zero for masked out particles, then sum
    sum_pt_truth = np.sum(pt_truth_per_event * valid_truth_mask, axis=1)
    sum_pt_pred  = np.sum(pt_pred_per_event * valid_pred_mask, axis=1)

    plt.figure(figsize=(7,7))
    rel_diff = (sum_pt_pred - sum_pt_truth) / sum_pt_truth
    plt.scatter(sum_pt_truth, rel_diff, alpha=0.3, s=10, label='All Events')
    plt.axhline(0, color='k', linestyle='--', alpha=0.7, label='y=0 (Perfect Agreement)')

    # Identify and print events with large relative error
    outlier_threshold = 2.0
    # Avoid division by zero or very small sum_pt_truth for stable outlier detection
    # We consider rel_diff directly as sum_pt_truth in denominator can be an issue if it's near zero
    # Let's ensure sum_pt_truth is not zero for the outliers we report to avoid misleading relative errors.
    outlier_indices = np.where((np.abs(rel_diff) > outlier_threshold) & (sum_pt_truth > 1e-3))[0] # Ensure sum_pt_truth is not effectively zero

    if len(outlier_indices) > 0:
        print(f"\n--- Events with |Relative pT Sum Error| > {outlier_threshold*100:.0f}% ---")
        num_outliers_to_print = min(len(outlier_indices), 5)
        for i in range(num_outliers_to_print):
            idx = outlier_indices[i]
            print(f"  Event {idx}: Truth Sum pT: {sum_pt_truth[idx]:.2f} GeV, Pred Sum pT: {sum_pt_pred[idx]:.2f} GeV, Rel. Error: {rel_diff[idx]:.2f}")

            # Get pT and labels for this specific event, applying masks
            event_pt_truth = pt_truth_per_event[idx][valid_truth_mask[idx]]
            event_labels_truth = truth_label_per_event[idx][valid_truth_mask[idx]]
            truth_particle_details = list(zip(np.round(event_pt_truth, 2), event_labels_truth.astype(int)))
            print(f"    Truth Particles (pT [GeV], class): {truth_particle_details if truth_particle_details else 'None'}")

            event_pt_pred = pt_pred_per_event[idx][valid_pred_mask[idx]]
            event_labels_pred = pred_label_per_event[idx][valid_pred_mask[idx]]
            pred_particle_details = list(zip(np.round(event_pt_pred, 2), event_labels_pred.astype(int)))
            print(f"    Pred. Particles (pT [GeV], class): {pred_particle_details if pred_particle_details else 'None'}\n")
        
        # Highlight outliers on the plot
        plt.scatter(sum_pt_truth[outlier_indices], rel_diff[outlier_indices], 
                    color='red', marker='x', s=50, label=f'|Rel. Error| > {outlier_threshold*100:.0f}%', zorder=5)
    else:
        print(f"\nNo events found with |Relative pT Sum Error| > {outlier_threshold*100:.0f}% (with Sum pT Truth > 0.001 GeV).")

    plt.xlabel('Sum pT Truth [GeV]')
    plt.ylabel('(Sum pT Pred - Sum pT Truth) / Sum pT Truth')
    plt.title('Per-event Relative Bias in Sum pT (Class 5 Excluded)')
    plt.legend()
    plt.ylim(max(-5, np.nanmin(rel_diff[np.isfinite(rel_diff)]) if np.any(np.isfinite(rel_diff)) else -5), 
             min(5, np.nanmax(rel_diff[np.isfinite(rel_diff)]) if np.any(np.isfinite(rel_diff)) else 5)) # Limit y-axis for better readability
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, 'sum_pt_scatter.png'))
    plt.close()

    # --- Plot pT distributions (truth vs predicted)
    bins = np.linspace(20, 300, 50)
    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    # If you have calibrated predictions, load as pt_pred_calibrated; else comment out
    # pt_pred_calibrated = ...
    axes = np.array([[axes]])  # for compatibility with axes[0,0] syntax
    axes[0,0].hist(pt_truth, bins=bins, alpha=0.7, label='Truth', density=True)
    axes[0,0].hist(pt_pred, bins=bins, alpha=0.7, label='Predicted (Original)', density=True)
    # Uncomment if you have calibrated predictions
    # axes[0,0].hist(pt_pred_calibrated, bins=bins, alpha=0.7, label='Predicted (Calibrated)', density=True)
    axes[0,0].set_xlabel('pT [GeV]')
    axes[0,0].set_ylabel('Normalized Count')
    axes[0,0].set_title('pT Distributions')
    axes[0,0].legend()
    axes[0,0].set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, 'pt_distributions.png'))
    plt.close()

    # --- Application du seuil de confiance
    # calcul des softmax + mask “non confiants” → classe 5
    logits_flat = data.pred_logits.reshape(-1, data.pred_logits.shape[-1])
    exp_l = np.exp(logits_flat - logits_flat.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    maxp = probs.max(axis=1)
    pred[maxp < conf_threshold] = 5

    # --- Masques « généraux »
    mask_truth_valid = truth != 5      # tout ce qui n’est pas “fake” au truth
    mask_pred_valid  = pred  != 5      # tout ce qui n’est pas “fake” en prédiction

    # --- Dictionnaire des types de particules
    # on reprend vos codes : 0/1 → chargés, 3 → neutral hadron, 4 → photon, 5 → fake
    classes = {
        "charged":        [0, 1],
        "muon":           [2],
        "photon":         [4],
        "neutral_hadron": [3]
    }

    # --- Histogramme des classes
    plt.hist(truth, bins=range(data.pred_logits.shape[-1] + 2),
             alpha=0.5, label="Truth")
    plt.hist(pred,  bins=range(data.pred_logits.shape[-1] + 2),
             alpha=0.5, label=f"Pred (thr={conf_threshold})")
    plt.legend(), plt.xlabel("Class"), plt.ylabel("Count")
    plt.title("Class distribution")
    plt.savefig(os.path.join(outputdir, f"class_distribution_thr{conf_threshold}.png"))
    plt.close()

    
    # --- Statistiques globales
    print("=== Particle counts ===")
    total_pred = 0
    total_truth = 0
    for name, codes in classes.items():
        tcount = np.isin(truth, codes).sum()
        pcount = np.isin(pred,  codes).sum()
        total_pred += pcount
        total_truth += tcount
        print(f"{name:15s} | truth = {tcount:6d}  pred = {pcount:6d}")
    print(f"{'fake':15s} | truth = {(truth==5).sum():6d}  pred = {(pred==5).sum():6d}")
    total_pred += (pred==5).sum()
    total_truth += (truth==5).sum()
    print()
    print("===Total===")
    print(f"truth = {total_truth:6d}  pred = {total_pred:6d}")
    print()
    #PR curve and AP score
    curves_classes = {
        "charged_1":        [0],
        "charged_2":        [1],
        "photon":         [4],
        "neutral_hadron": [3],
        "None particle": [5]
    }
    plot_pr_curves(
    truth,
    data.pred_logits.reshape(-1, data.pred_logits.shape[-1]),
    curves_classes,
    outputdir
    )

    # --- Appels aux fonctions de plot
    # resolution chargée
    mask_charged_truth = np.isin(truth, classes["charged"])
    plot_charged_resolution(
        {"pflow": pt_pred}, pt_truth,
        mask=mask_charged_truth,
        savedir=outputdir
    )

    # efficacité neutre
    plot_neutral_efficiency(
        pred, truth, pt_truth,
        mask=mask_truth_valid,
        savedir=outputdir
    )

    # fake-rate neutre
    plot_neutral_fakerate(
        pred, truth, pt_truth,
        mask=mask_pred_valid,
        savedir=outputdir
    )

    # probabilité neutre
    plot_neutral_probability(
        pred, truth, pt_truth,
        mask=mask_truth_valid,
        savedir=outputdir
    )

    # cinématiques neutres (on restreint aux neutres bien identifiés)
    mask_neutral_both = np.logical_and(
        np.isin(truth, classes["photon"] + classes["neutral_hadron"]),
        np.isin(pred,  classes["photon"] + classes["neutral_hadron"])
    )
    plot_neutral_kinematics(
        pt_pred, data.pred_eta, data.pred_phi,
        pt_truth, data.truth_eta, data.truth_phi,
        mask=mask_neutral_both,
        savedir=outputdir
    )

if __name__ == "__main__":
    main()
