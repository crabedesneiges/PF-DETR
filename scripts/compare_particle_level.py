import os
import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
from sklearn.metrics import precision_recall_curve, average_precision_score

# Assuming visualize.py and visualize_hgp.py are in the same directory or accessible in PYTHONPATH
# We will rename the DataReader classes to avoid naming conflicts
from visualize import DataReader as DetrDataReader
from visualize_hgp import DataReader as HGPDataReader

# DETR class mapping (from visualize.py, assuming this is standard)
DETR_CLASS_MAP_VIS = {
    "charged_hadron": [0],
    "electron": [1],
    "muon": [2],
    "neutral_hadron": [3],
    "photon": [4],
    "padding": [5] # or the actual padding index if different
}

# HGP class mapping (needs to be defined based on HGPDataReader output)
# Example: 0: photon, 1: neutral hadron, 2: charged hadron, 3: electron, 4: muon
HGP_TO_DETR_LABEL_MAP = {
    0: 4,  # HGP photon -> DETR photon
    1: 3,  # HGP neutral hadron -> DETR neutral hadron
    2: 0,  # HGP charged hadron -> DETR charged hadron
    3: 1,  # HGP electron -> DETR electron
    4: 2   # HGP muon -> DETR muon
    # Add mappings for any other HGP classes if they exist
}

CLASS_NAMES = ["charged_hadron", "electron", "muon", "neutral_hadron", "photon"]

def adapt_detr_data(data, conf_threshold=0.1):
    """Adapts DETR data to a common format."""
    truth = data.truth_labels_flat
    pred  = data.pred_classes_flat.copy()  # copie à écraser avec le seuil
    pt_truth = data.truth_pt
    pt_pred   = data.pred_pt
    print("=======================")
    print("pt_truth", pt_truth[:30])
    print("pt_pred", pt_pred[:30])
    print("trut", truth[:30])
    print("pred", pred[:30])
    print("=======================")
    # --- Scatter plot of per-event sum pT (pred vs truth)
    # Use per-event arrays from truth_boxes and pred_boxes
    
    #pt_truth_per_event = data.truth_boxes[..., 0] / 1000  # shape: (n_events, n_truth_particles)
    #pt_pred_per_event = data.pred_boxes[..., 0] / 1000    # shape: (n_events, n_pred_particles)

    # Mask class 5 (fake) particles
    truth_label_per_event = data.truth_labels  # shape: (n_events, n_truth_particles)
    pred_label_per_event = data.pred_classes   # shape: (n_events, n_pred_particles)
    #valid_truth_mask = (truth_label_per_event != 5)
    #valid_pred_mask = (pred_label_per_event != 5)    
    # --- Application du seuil de confiance
    # calcul des softmax + mask "non confiants" → classe 5
    logits_flat = data.pred_logits.reshape(-1, data.pred_logits.shape[-1])
    exp_l = np.exp(logits_flat - logits_flat.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    maxp = probs.max(axis=1)
    argmax_class = np.argmax(probs, axis=1)
    mask = ((argmax_class >= 3) & (argmax_class <= 5) & (maxp < conf_threshold))
    pred[mask] = 5

    # --- Masques « généraux »
    mask_truth = truth != 5      # tout ce qui n'est pas "fake" au truth
    mask_pred  = pred  != 5      # tout ce qui n'est pas "fake" en prédiction
    mask_truth_and_pred = np.logical_and(mask_truth, mask_pred)
    
    mask_neutral_pred = pred > 2 
    mask_truth_pred_neutral = np.logical_and(mask_truth_and_pred, mask_neutral_pred)
    mask_truth_pred_charged = np.logical_and(mask_truth_and_pred, np.logical_not(mask_neutral_pred))

    eta_truth = data.truth_eta
    eta_pred  = data.pred_eta
    phi_truth = data.truth_phi
    phi_pred  = data.pred_phi
    print("===================================================================")
    print("number hadron truth:", np.sum(mask_truth & (truth == 3)))
    print("number hadron pred:", np.sum(mask_pred & (pred == 3)))
    print("===================================================================")
    return truth, pred, pt_truth, pt_pred, eta_truth, eta_pred, phi_truth, phi_pred, mask_truth_and_pred, mask_pred, mask_truth, mask_truth_pred_neutral, mask_truth_pred_charged

def adapt_hgp_data(data):
    """Adapts HGP data to a common format."""
    hgp_truth_labels = data.truth_class
    hgp_pred_labels = data.pflow_class

    # Map HGP labels to DETR labels
    # Use the padding index from DETR_CLASS_MAP_VIS for unmapped labels
    # (defined globally as DETR_CLASS_MAP_VIS = {..., "padding": [5]})
    padding_idx = DETR_CLASS_MAP_VIS.get("padding", [5])[0]
    
    # Create a vectorized mapping function. This ensures that HGP_TO_DETR_LABEL_MAP
    # is applied element-wise to the hgp_truth_labels and hgp_pred_labels arrays.
    # If a label from HGP data is not in HGP_TO_DETR_LABEL_MAP, it defaults to padding_idx.
    map_func = np.vectorize(lambda x: HGP_TO_DETR_LABEL_MAP.get(x, padding_idx))
    
    truth = map_func(hgp_truth_labels)
    pred = map_func(hgp_pred_labels)
    pt_truth = data.truth_pt
    pt_pred = data.pflow_pt
    eta_truth = data.truth_eta
    eta_pred = data.pflow_eta
    phi_truth = data.truth_phi
    phi_pred = data.pflow_phi

    pred_indicator = data.pred_indicator
    truth_indicator = data.truth_indicator

    print("truth:", truth[:10])
    print("truth_indicator:", truth_indicator[:10])
    print("pred:", pred[:10])
    print("pred_indicator:", pred_indicator[:10])

    mask_pred = pred_indicator > 0.5
    mask_truth = truth_indicator > 0.5
    mask_truth_and_pred = np.logical_and(mask_truth, mask_pred)
    
    mask_neutral_pred = pred > 2 
    mask_truth_pred_neutral = np.logical_and(mask_truth_and_pred, mask_neutral_pred)
    mask_truth_pred_charged = np.logical_and(mask_truth_and_pred, np.logical_not(mask_neutral_pred))

    return truth, pred, pt_truth, pt_pred, eta_truth, eta_pred, phi_truth, phi_pred, pred_indicator, truth_indicator, mask_truth_and_pred, mask_pred, mask_truth, mask_truth_pred_neutral, mask_truth_pred_charged

def _calc_track_pt_sigma(pt, ref_pt, bin_pt, fignames=None, verbose=0):
    """Calculate pT sigma resolution using different methods."""

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
        # Safeguard against zero or negative width which would cause np.histogram to raise an error
        if delta <= 0:
            # Fallback to simple standard deviation if distribution is too narrow
            return np.std(x) if len(x) > 1 else 0.0
        xmin1, xmax1 = mode - delta, mode + delta
        if xmax1 <= xmin1:
            return np.std(x) if len(x) > 1 else 0.0
        C1, mean1, sigma1 = _fit(x, xmin1, xmax1, nbins=20)
        # Guard against pathological sigma values
        if sigma1 <= 0:
            return np.std(x) if len(x) > 1 else 0.0
        xmin2, xmax2 = mean1 - 2 * sigma1, mean1 + 2 * sigma1
        if xmax2 <= xmin2:
            return sigma1  # best we can do
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
        sigma_pt_fit[i] = _sigma_from_fit(d_pt, fignames=fignames + f".{i}.png" if fignames else None)

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


def calculate_efficiency_by_pt_bin(truth, pred, pt_truth, mask_truth, mask_pred, pt_bins):
    """Calculate efficiency for each class in pT bins."""
    efficiency_by_class = {}
    
    for class_id in range(5):  # 0-4 for the 5 particle types
        efficiency_bins = []
        
        for low, high in pt_bins:
            # Mask for particles in this pT bin
            pt_mask = (pt_truth >= low) & (pt_truth < high)
            
            # Combined mask: truth class, pT bin, and validity masks
            class_pt_mask = (truth == class_id) & pt_mask & mask_truth
            
            # True positives: correctly predicted particles of this class in this pT bin
            tp = np.sum(class_pt_mask & (pred == class_id) & mask_pred)
            
            # Total truth particles of this class in this pT bin
            total_truth = np.sum(class_pt_mask)
            
            # Efficiency = TP / Total_Truth
            efficiency = tp / total_truth if total_truth > 0 else 0
            efficiency_bins.append(efficiency)
        
        efficiency_by_class[class_id] = np.array(efficiency_bins)
    
    return efficiency_by_class

def plot_efficiency_by_pt_bin(truth_1, pred_1, pt_truth_1, mask_truth_1, mask_pred_1, name_1,
                             truth_2, pred_2, pt_truth_2, mask_truth_2, mask_pred_2, name_2,
                             outputdir):
    """Plot efficiency by pT bin for each particle class."""
    from matplotlib.ticker import ScalarFormatter
    
    # Define pT bins (same as used in sigma resolution)
    xticks = [1, 5, 10, 20, 30, 50]
    bin_pt = [(xticks[i], xticks[i + 1]) for i in range(len(xticks) - 1)]
    x = [(high + low) / 2 for low, high in bin_pt]
    xerr = [(high - low) / 2 for low, high in bin_pt]
    
    # Calculate efficiency for both models
    eff_1 = calculate_efficiency_by_pt_bin(truth_1, pred_1, pt_truth_1, mask_truth_1, mask_pred_1, bin_pt)
    eff_2 = calculate_efficiency_by_pt_bin(truth_2, pred_2, pt_truth_2, mask_truth_2, mask_pred_2, bin_pt)
    
    # Create subplots for each class
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for class_id in range(5):  # 0-4 for the 5 particle types
        ax = axes[class_id]
        
        # Plot efficiency curves for both models
        ax.errorbar(
            x=x, y=eff_1[class_id], xerr=xerr, yerr=None,
            linestyle="none", marker="o", label=name_1,
            markersize=8, markerfacecolor='blue', markeredgecolor='darkblue', 
            markeredgewidth=1, color='blue', linewidth=2
        )
        ax.errorbar(
            x=x, y=eff_2[class_id], xerr=xerr, yerr=None,
            linestyle="none", marker="s", label=name_2,
            markersize=6, markerfacecolor='red', markeredgecolor='darkred', 
            markeredgewidth=1, color='red', linewidth=2
        )
        
        # Styling
        ax.set_xlabel('p_T^truth [GeV]', fontsize=12)
        ax.set_ylabel('efficiency', fontsize=12)
        ax.set_title(f'{CLASS_NAMES[class_id].replace("_", " ").title()} Efficiency', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='lower right')
        
        # Set axis limits and scale
        ax.set_xlim(xticks[0], xticks[-1])
        ax.set_ylim(0, 1.05)  # 0 to 105% for efficiency
        ax.set_xscale('log')
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        
        # Add percentage ticks on y-axis
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        
        # Add minor grid for better readability
        ax.grid(True, which='minor', alpha=0.2)
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
    
    # Hide the last subplot if we have 6 subplots but only 5 classes
    axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, 'efficiency_by_pt_bin_all_classes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create individual plots for each class (similar to your reference image style)
    for class_id in range(5):
        if CLASS_NAMES[class_id] != 'muon':
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot efficiency curves for both models
            ax.errorbar(
                x=x, y=eff_1[class_id], xerr=xerr, yerr=None,
                linestyle="none", marker="o", label=name_1,
                markersize=10, markerfacecolor='blue', markeredgecolor='darkblue', 
                markeredgewidth=2, color='blue', linewidth=2
            )
            ax.errorbar(
                x=x, y=eff_2[class_id], xerr=xerr, yerr=None,
                linestyle="none", marker="^", label=name_2,
                markersize=8, markerfacecolor='red', markeredgecolor='darkred', 
                markeredgewidth=2, color='red', linewidth=2
            )
            
            # Styling to match your reference image
            ax.set_xlabel('p_T^truth [GeV]', fontsize=14)
            ax.set_ylabel('efficiency', fontsize=14)
            ax.set_title(f'{CLASS_NAMES[class_id].replace("_", " ").title()} Classification Efficiency', fontsize=16)
            ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.5)
            ax.legend(fontsize=12, loc='lower right', frameon=True, fancybox=True, shadow=True)
            
            # Set axis limits and scale
            ax.set_xlim(xticks[0], xticks[-1])
            ax.set_ylim(0, 1.05)  # 0 to 105% for efficiency
            ax.set_xscale('log')
            ax.set_xticks(xticks)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            
            # Add percentage ticks on y-axis
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
            
            # Add minor grid for better readability
            ax.grid(True, which='minor', alpha=0.3, linestyle=':', linewidth=0.5)
            
            # Set tick parameters
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            
            # Add border around the plot
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(outputdir, f'efficiency_by_pt_bin_{CLASS_NAMES[class_id]}.png'), 
                    dpi=300, bbox_inches='tight')
            plt.close()

def plot_pt_relative_residual(truth_1, pred_1, pt_truth_1, pt_pred_1, mask_1, name_1,
                             truth_2, pred_2, pt_truth_2, pt_pred_2, mask_2, name_2,
                             outputdir, particle_type):
    """Plot relative pT residuals comparison."""
    plt.figure(figsize=(12, 5))
    
    # Calculate relative residuals
    rel_residual_1 = (pt_pred_1[mask_1] - pt_truth_1[mask_1]) / pt_truth_1[mask_1]
    rel_residual_2 = (pt_pred_2[mask_2] - pt_truth_2[mask_2]) / pt_truth_2[mask_2]

    # Overall comparison
    plt.subplot(1, 1, 1)
    _custom_hist(rel_residual_1, name_1, bins=41, range=(-1, 1))
    _custom_hist(rel_residual_2, name_2, bins=41, range=(-1, 1))
    plt.xlabel('Relative pT Residual (pred - truth) / truth')
    plt.ylabel('Arbitrary unit')
    plt.title(f'Relative pT Residual Comparison ({particle_type})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, f'pt_relative_residual_comparison_{particle_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def _custom_hist(data1, name1, bins, range):
    c1, edges = np.histogram(data1, bins=bins, range=range)

    _ = plt.step(
        edges[:-1],
        c1 / c1.sum(),
        where="mid",
        label=name1,
    )

    plt.legend()
    
def plot_eta_residual(truth_1, pred_1, eta_truth_1, eta_pred_1, mask_1, name_1,
                      truth_2, pred_2, eta_truth_2, eta_pred_2, mask_2, name_2,
                      outputdir, particle_type):
    """Plot eta residuals comparison."""
    plt.figure(figsize=(12, 5))
    
    # Calculate residuals
    eta_residual_1 = eta_pred_1[mask_1] - eta_truth_1[mask_1]
    eta_residual_2 = eta_pred_2[mask_2] - eta_truth_2[mask_2]
    
    # Overall comparison
    plt.subplot(1, 1, 1)
    _custom_hist(eta_residual_1, name_1, bins=41, range=(-0.4, 0.4))
    _custom_hist(eta_residual_2, name_2, bins=41, range=(-0.4, 0.4))
    plt.xlabel('η Residual (pred - truth)')
    plt.ylabel('Density')
    plt.title(f'η Residual Comparison ({particle_type})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, f'eta_residual_comparison_{particle_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_phi_residual(truth_1, pred_1, phi_truth_1, phi_pred_1, mask_1, name_1,
                      truth_2, pred_2, phi_truth_2, phi_pred_2, mask_2, name_2,
                      outputdir, particle_type):
    """Plot phi residuals comparison."""
    plt.figure(figsize=(12, 5))
    
    # Calculate residuals (handle wraparound at ±π)
    phi_residual_1 = phi_pred_1[mask_1] - phi_truth_1[mask_1]
    phi_residual_1 = np.where(phi_residual_1 > np.pi, phi_residual_1 - 2*np.pi, phi_residual_1)
    phi_residual_1 = np.where(phi_residual_1 < -np.pi, phi_residual_1 + 2*np.pi, phi_residual_1)
    
    phi_residual_2 = phi_pred_2[mask_2] - phi_truth_2[mask_2]
    phi_residual_2 = np.where(phi_residual_2 > np.pi, phi_residual_2 - 2*np.pi, phi_residual_2)
    phi_residual_2 = np.where(phi_residual_2 < -np.pi, phi_residual_2 + 2*np.pi, phi_residual_2)
    
    # Overall comparison
    plt.subplot(1, 1, 1)
    _custom_hist(phi_residual_1, name_1, bins=41, range=(-0.4, 0.4))
    _custom_hist(phi_residual_2, name_2, bins=41, range=(-0.4, 0.4))
    plt.xlabel('φ Residual (pred - truth)')
    plt.ylabel('Density')
    plt.title(f'φ Residual Comparison ({particle_type})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, f'phi_residual_comparison_{particle_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_efficiency_fake_rate(truth, pred, mask_truth, mask_pred):
    """Calculate efficiency and fake rate for each class."""
    efficiency = {}
    fake_rate = {}
    
    for class_id in range(5):  # 0-4 for the 5 particle types
        # True positives: correctly predicted particles of this class
        tp = np.sum((truth == class_id) & (pred == class_id) & mask_truth & mask_pred)
        
        # Total truth particles of this class
        total_truth = np.sum((truth == class_id) & mask_truth)
        
        # Total predicted particles of this class
        total_pred = np.sum((pred == class_id) & mask_pred)
        
        # False positives: predicted as this class but actually another class
        fp = total_pred - tp
        
        # Efficiency = TP / (TP + FN) = TP / Total_Truth
        efficiency[class_id] = tp / total_truth if total_truth > 0 else 0
        
        # Fake rate = FP / (TP + FP) = FP / Total_Pred
        fake_rate[class_id] = fp / total_pred if total_pred > 0 else 0
    
    return efficiency, fake_rate

def plot_efficiency_fake_rate(truth_1, pred_1, mask_truth_1, mask_pred_1, name_1,
                             truth_2, pred_2, mask_truth_2, mask_pred_2, name_2,
                             outputdir):
    """Plot efficiency and fake rate comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate metrics
    eff_1, fake_1 = calculate_efficiency_fake_rate(truth_1, pred_1, mask_truth_1, mask_pred_1)
    eff_2, fake_2 = calculate_efficiency_fake_rate(truth_2, pred_2, mask_truth_2, mask_pred_2)
    
    classes = list(range(5))
    x_pos = np.arange(len(classes))
    width = 0.35
    
    # Efficiency plot
    eff_vals_1 = [eff_1[c] for c in classes]
    eff_vals_2 = [eff_2[c] for c in classes]
    
    ax1.bar(x_pos - width/2, eff_vals_1, width, label=name_1, alpha=0.8)
    ax1.bar(x_pos + width/2, eff_vals_2, width, label=name_2, alpha=0.8)
    ax1.set_xlabel('Particle Class')
    ax1.set_ylabel('Efficiency')
    ax1.set_title('Classification Efficiency by Class')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([CLASS_NAMES[i] for i in classes], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Fake rate plot
    fake_vals_1 = [fake_1[c] for c in classes]
    fake_vals_2 = [fake_2[c] for c in classes]
    
    ax2.bar(x_pos - width/2, fake_vals_1, width, label=name_1, alpha=0.8)
    ax2.bar(x_pos + width/2, fake_vals_2, width, label=name_2, alpha=0.8)
    ax2.set_xlabel('Particle Class')
    ax2.set_ylabel('Fake Rate')
    ax2.set_title('Classification Fake Rate by Class')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([CLASS_NAMES[i] for i in classes], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, 'efficiency_fake_rate_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pt_sigma_resolution(truth_1, pred_1, pt_truth_1, pt_pred_1, mask_1, name_1,
                            truth_2, pred_2, pt_truth_2, pt_pred_2, mask_2, name_2,
                            outputdir, particle_type="neutral", resolution_type="fit"):
    """Plot pT sigma resolution comparison for two models."""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.ticker import ScalarFormatter
    
    # Define pT bins
    xticks = [10,15, 20, 30, 50, 200]
    bin_pt = [(xticks[i], xticks[i + 1]) for i in range(len(xticks) - 1)]
    x = [(high + low) / 2 for low, high in bin_pt]
    xerr = [(high - low) / 2 for low, high in bin_pt]
    
    # Calculate sigma for both models
    sigma_results_1 = _calc_track_pt_sigma(
        pt_pred_1[mask_1], pt_truth_1[mask_1], bin_pt,
        fignames=None, verbose=1
    )
    
    sigma_results_2 = _calc_track_pt_sigma(
        pt_pred_2[mask_2], pt_truth_2[mask_2], bin_pt,
        fignames=None, verbose=1
    )
    
    # Create single plot for comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Use specified resolution method
    sigma_1 = sigma_results_1[resolution_type]
    sigma_2 = sigma_results_2[resolution_type]
    
    # Plot both curves with errorbar style (matching second function)
    ax.errorbar(
        x=x, y=sigma_1, xerr=xerr, yerr=None,
        linestyle="None", marker="o", label=name_1,
        markersize=10, markerfacecolor='blue', markeredgecolor='darkblue', 
        markeredgewidth=1, color='blue'
    )
    ax.errorbar(
        x=x, y=sigma_2, xerr=xerr, yerr=None,
        linestyle="None", marker="s", label=name_2,
        markersize=8, markerfacecolor='red', markeredgecolor='darkred', 
        markeredgewidth=1, color='red'
    )
    
    # Styling to match both reference styles
    ax.set_xlabel('truth pT [GeV]', fontsize=14)
    ax.set_ylabel('sigma pT [GeV]', fontsize=14)
    ax.grid(True, alpha=0.5)
    ax.legend(fontsize=12, loc='upper left')
    
    # Set axis limits and scale
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xscale('log')
    ax.set_xticks(xticks)
    ax.set_yticks([5, 15, 25, 35, 45, 55, 65])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    
    # Add minor grid for better readability
    ax.grid(True, which='minor', alpha=0.3)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, f'pt_sigma_resolution_comparison_{particle_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create detailed subplots for all methods (optional additional plot)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = ['q1q3', 'std', 'fit', 'qoverpt_q1q3']
    titles = [f'σ(pT) from Q1/Q3 {particle_type}', f'σ(pT) from Std Dev {particle_type}', f'σ(pT) from Gaussian Fit {particle_type}', f'σ(q/pT) from Q1/Q3 {particle_type}']
    
    for i, (method, title) in enumerate(zip(methods, titles)):
        ax = axes[i//2, i%2]
        
        sigma_1 = sigma_results_1[method]
        sigma_2 = sigma_results_2[method]
        
        ax.errorbar(
            x=x, y=sigma_1, xerr=xerr, yerr=None,
            linestyle="None", marker="o", label=name_1
        )
        ax.errorbar(
            x=x, y=sigma_2, xerr=xerr, yerr=None,
            linestyle="None", marker="s", label=name_2
        )
        
        ax.set_xlabel('truth pT [GeV]')
        ax.set_ylabel('σ')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xticks(xticks)
        ax.set_yticks([5, 15, 25, 35, 45, 55, 65])
        ax.xaxis.set_major_formatter(ScalarFormatter())

    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, 'pt_sigma_resolution_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_true_prediction_counts(truth_1, pred_1, mask_truth_1, mask_pred_1, name_1,
                                truth_2, pred_2, mask_truth_2, mask_pred_2, name_2,
                                outputdir):
    """Bar plot of true particle counts and correctly predicted counts per class for two models."""
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    classes = np.arange(5)

    # Ground truth counts (use masks to filter valid truth particles)
    truth_counts = np.array([np.sum((truth_1 == c) & mask_truth_1) for c in classes])

    # True positive counts for each model
    pred_counts_1 = np.array([np.sum((pred_1 == c) & mask_truth_1 & mask_pred_1) for c in classes])
    pred_counts_2 = np.array([np.sum((pred_2 == c) & mask_truth_2 & mask_pred_2) for c in classes])

    width = 0.25
    x = np.arange(len(classes))

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, truth_counts, width=width, label='Truth', color='gray', alpha=0.7)
    plt.bar(x, pred_counts_1, width=width, label=f'{name_1} TP', color='blue', alpha=0.8)
    plt.bar(x + width, pred_counts_2, width=width, label=f'{name_2} TP', color='red', alpha=0.8)

    plt.xticks(x, [CLASS_NAMES[i] for i in classes], rotation=45)
    plt.ylabel('Count')
    plt.title('Number of Predicted Particles per Class')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    os.makedirs(outputdir, exist_ok=True)
    plt.savefig(os.path.join(outputdir, 'true_prediction_counts.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_cardinality_error(truth_1, pred_1, pt_truth_1, mask_truth_1, mask_pred_1, name_1,
                           truth_2, pred_2, pt_truth_2, mask_truth_2, mask_pred_2, name_2,
                           outputdir):
    """Plot N_pred - N_truth per class (overall) and as a function of pT bin for each model."""
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    classes = np.arange(5)

    # Overall counts
    truth_counts_1 = np.array([np.sum((truth_1 == c) & mask_truth_1) for c in classes])
    pred_counts_1  = np.array([np.sum((pred_1 == c) & mask_pred_1) for c in classes])
    with np.errstate(divide='ignore', invalid='ignore'):
        diff_1 = (pred_counts_1 - truth_counts_1) / truth_counts_1.astype(float)
    
    truth_counts_2 = np.array([np.sum((truth_2 == c) & mask_truth_2) for c in classes])
    pred_counts_2  = np.array([np.sum((pred_2 == c) & mask_pred_2) for c in classes])
    with np.errstate(divide='ignore', invalid='ignore'):
        diff_2 = (pred_counts_2 - truth_counts_2) / truth_counts_2.astype(float)
    
    width = 0.3
    x = np.arange(len(classes))

    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, diff_1, width=width, label=name_1, color='blue', alpha=0.8)
    plt.bar(x + width/2, diff_2, width=width, label=name_2, color='red', alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xticks(x, [CLASS_NAMES[i] for i in classes], rotation=45)
    plt.ylabel('Relative Cardinality Error')
    plt.title('Relative Cardinality Error per Class (Overall)')
    plt.legend()
    plt.tight_layout()
    os.makedirs(outputdir, exist_ok=True)
    plt.savefig(os.path.join(outputdir, 'cardinality_error_overall.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- pT-bin dependent ---
    xticks = [1, 5, 10, 20, 30, 50]
    bin_pt = [(xticks[i], xticks[i + 1]) for i in range(len(xticks) - 1)]
    x_mid = [(low + high)/2 for low, high in bin_pt]
    xerr = [(high - low)/2 for low, high in bin_pt]

    fig, axes = plt.subplots(2,3, figsize=(18, 12))
    axes = axes.flatten()

    for class_id in classes:
        ax = axes[class_id]
        diff_bins_1 = []
        diff_bins_2 = []
        for low, high in bin_pt:
            mask_bin_1 = (pt_truth_1 >= low) & (pt_truth_1 < high)
            mask_bin_2 = (pt_truth_2 >= low) & (pt_truth_2 < high)
            n_truth_1_bin = np.sum((truth_1 == class_id) & mask_bin_1 & mask_truth_1)
            n_pred_1_bin = np.sum((pred_1 == class_id) & mask_bin_1 & mask_pred_1)
            if n_truth_1_bin > 0:
                diff_bins_1.append((n_pred_1_bin - n_truth_1_bin) / n_truth_1_bin)
            else:
                diff_bins_1.append(np.nan)
            n_truth_2_bin = np.sum((truth_2 == class_id) & mask_bin_2 & mask_truth_2)
            n_pred_2_bin = np.sum((pred_2 == class_id) & mask_bin_2 & mask_pred_2)
            if n_truth_2_bin > 0:
                diff_bins_2.append((n_pred_2_bin - n_truth_2_bin) / n_truth_2_bin)
            else:
                diff_bins_2.append(np.nan)
        ax.errorbar(x_mid, diff_bins_1, xerr=xerr, fmt='o', label=name_1, color='blue')
        ax.errorbar(x_mid, diff_bins_2, xerr=xerr, fmt='s', label=name_2, color='red')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(xticks)
        ax.set_xscale('log')
        ax.set_xlim(1, 200)
        ax.set_xlabel('p_T^truth [GeV]')
        ax.set_ylabel('Relative Cardinality Error')
        ax.set_title(CLASS_NAMES[class_id])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[5].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, 'cardinality_error_by_pt_bin.png'), dpi=300, bbox_inches='tight')
    plt.close()

# --- Main CLI --- 
@click.command()
@click.option('--model1_file', required=True, type=click.Path(exists=True), help='Path to the first .npz file.')
@click.option('--model1_type', required=True, type=click.Choice(['detr', 'hgp'], case_sensitive=False), help='Type of the first model.')
@click.option('--model1_name', required=True, type=str, help='Name for the first model (for legends).')
@click.option('--model2_file', required=True, type=click.Path(exists=True), help='Path to the second .npz file.')
@click.option('--model2_type', required=True, type=click.Choice(['detr', 'hgp'], case_sensitive=False), help='Type of the second model.')
@click.option('--model2_name', required=True, type=str, help='Name for the second model (for legends).')
@click.option('--outputdir', default='comparison_plots_particle_level', help='Directory to save plots.')
@click.option('--detr_conf_threshold', default=0.1, type=float, help='Confidence threshold for DETR predictions.')
def main(model1_file, model1_type, model1_name, 
         model2_file, model2_type, model2_name, 
         outputdir, detr_conf_threshold):
    
    os.makedirs(outputdir, exist_ok=True)
    mpl.use('Agg') # Use Agg backend for non-interactive plotting

    # Load data
    print(f"Loading Model 1 ({model1_name}, type: {model1_type}) from {model1_file}")
    if model1_type == 'detr':
        reader1 = DetrDataReader(model1_file)
        truth_1, pred_1, pt_truth_1, pt_pred_1, eta_truth_1, eta_pred_1, phi_truth_1, phi_pred_1, mask_truth_and_pred_1, mask_pred_1, mask_truth_1, mask_truth_pred_neutral_1, mask_truth_pred_charged_1 = adapt_detr_data(reader1, detr_conf_threshold)
    else: # hgp
        reader1 = HGPDataReader(model1_file) # Assuming HGPDataReader takes filename
        truth_1, pred_1, pt_truth_1, pt_pred_1, eta_truth_1, eta_pred_1, phi_truth_1, phi_pred_1, pred_indicator_1, truth_indicator_1, mask_truth_and_pred_1, mask_pred_1, mask_truth_1, mask_truth_pred_neutral_1, mask_truth_pred_charged_1 = adapt_hgp_data(reader1)

    print(f"Loading Model 2 ({model2_name}, type: {model2_type}) from {model2_file}")
    if model2_type == 'detr':
        reader2 = DetrDataReader(model2_file)
        truth_2, pred_2, pt_truth_2, pt_pred_2, eta_truth_2, eta_pred_2, phi_truth_2, phi_pred_2, mask_truth_and_pred_2, mask_pred_2, mask_truth_2, mask_truth_pred_neutral_2, mask_truth_pred_charged_2 = adapt_detr_data(reader2, detr_conf_threshold)
    else: # hgp
        reader2 = HGPDataReader(model2_file)
        truth_2, pred_2, pt_truth_2, pt_pred_2, eta_truth_2, eta_pred_2, phi_truth_2, phi_pred_2, pred_indicator_2, truth_indicator_2, mask_truth_and_pred_2, mask_pred_2, mask_truth_2, mask_truth_pred_neutral_2, mask_truth_pred_charged_2 = adapt_hgp_data(reader2)
    
    print("Data loaded and adapted. Starting plotting...")

    pt_pred_1 = pt_pred_1.clip(0.1, 200)  # Clip pT predictions to avoid extreme values
    pt_pred_2 = pt_pred_2.clip(0.1, 200)
    # --- Generate Plots ---
    
    print("1. Plotting relative pT residuals...")
    plot_pt_relative_residual(
        truth_1, pred_1, pt_truth_1, pt_pred_1, mask_truth_pred_neutral_1, model1_name,
        truth_2, pred_2, pt_truth_2, pt_pred_2, mask_truth_pred_neutral_2, model2_name,
        outputdir, 'neutral'
    )
    
    plot_pt_relative_residual(
        truth_1, pred_1, pt_truth_1, pt_pred_1, mask_truth_pred_charged_1, model1_name,
        truth_2, pred_2, pt_truth_2, pt_pred_2, mask_truth_pred_charged_2, model2_name,
        outputdir, 'charged'
    )
    
    print("2. Plotting eta residuals...")
    plot_eta_residual(
        truth_1, pred_1, eta_truth_1, eta_pred_1, mask_truth_pred_neutral_1, model1_name,
        truth_2, pred_2, eta_truth_2, eta_pred_2, mask_truth_pred_neutral_2, model2_name,
        outputdir, 'neutral'
    )
    
    plot_eta_residual(
        truth_1, pred_1, eta_truth_1, eta_pred_1, mask_truth_pred_charged_1, model1_name,
        truth_2, pred_2, eta_truth_2, eta_pred_2, mask_truth_pred_charged_2, model2_name,
        outputdir, 'charged'
    )
    
    print("3. Plotting phi residuals...")
    plot_phi_residual(
        truth_1, pred_1, phi_truth_1, phi_pred_1, mask_truth_pred_neutral_1, model1_name,
        truth_2, pred_2, phi_truth_2, phi_pred_2, mask_truth_pred_neutral_2, model2_name,
        outputdir, 'neutral'
    )
    
    plot_phi_residual(
        truth_1, pred_1, phi_truth_1, phi_pred_1, mask_truth_pred_charged_1, model1_name,
        truth_2, pred_2, phi_truth_2, phi_pred_2, mask_truth_pred_charged_2, model2_name,
        outputdir, 'charged'
    )
    
    print("4. Plotting efficiency and fake rates...")
    plot_efficiency_fake_rate(
        truth_1, pred_1, mask_truth_1, mask_pred_1, model1_name,
        truth_2, pred_2, mask_truth_2, mask_pred_2, model2_name,
        outputdir
    )
    
    print("5. Plotting pT sigma resolution...")
    plot_pt_sigma_resolution(
        truth_1, pred_1, pt_truth_1, pt_pred_1, mask_truth_pred_neutral_1, model1_name,
        truth_2, pred_2, pt_truth_2, pt_pred_2, mask_truth_pred_neutral_2, model2_name,
        outputdir, 'neutral'
    )

    plot_pt_sigma_resolution(
        truth_1, pred_1, pt_truth_1, pt_pred_1, mask_truth_pred_charged_1, model1_name,
        truth_2, pred_2, pt_truth_2, pt_pred_2, mask_truth_pred_charged_2, model2_name,
        outputdir, 'charged'
    )

    print("6. Plotting efficiency by pT bin for each class...")
    plot_efficiency_by_pt_bin(
        truth_1, pred_1, pt_truth_1, mask_truth_1, mask_pred_1, model1_name,
        truth_2, pred_2, pt_truth_2, mask_truth_2, mask_pred_2, model2_name,
        outputdir
    )

    print("7. Plotting true prediction counts per class vs ground truth...")
    plot_true_prediction_counts(
        truth_1, pred_1, mask_truth_1, mask_pred_1, model1_name,
        truth_2, pred_2, mask_truth_2, mask_pred_2, model2_name,
        outputdir
    )
    
    print("8. Plotting cardinality error (overall and by pT bin)...")
    plot_cardinality_error(
        truth_1, pred_1, pt_truth_1, mask_truth_1, mask_pred_1, model1_name,
        truth_2, pred_2, pt_truth_2, mask_truth_2, mask_pred_2, model2_name,
        outputdir
    )
    print("Max pr truth 1",np.max(pt_truth_1[mask_truth_1]))
    print("Max pr pred 1",np.max(pt_pred_1[mask_pred_1]))
    print("Max pr truth 2",np.max(pt_truth_2[mask_truth_2]))
    print("Max pr pred 2",np.max(pt_pred_2[mask_pred_2]))
    print(f"All plots saved in {outputdir}")

if __name__ == '__main__':
    main()