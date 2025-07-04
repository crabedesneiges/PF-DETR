# plotting.py

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from cycler import cycler
from scipy.optimize import curve_fit

# Ces fonctions utilitaires sont reprises de votre script et ne nécessitent pas ou peu de modifications.
# Elles seront appelées par les nouvelles fonctions de plotting "multi".

CLASS_NAMES = ["charged_hadron", "electron", "muon", "neutral_hadron", "photon"]

# Définir un cycle de styles avec la référence en rouge et des couleurs bien séparées
def get_style_cycler(model_names):
    """Crée un cycle de styles avec la référence en rouge et des couleurs bien distinctes."""
    # Couleurs bien séparées dans l'espace colorimétrique
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*']
    
    # Si le premier modèle contient "ref", "reference", "hgp" ou similaire, on le met en rouge
    first_model = list(model_names)[0].lower()
    if any(keyword in first_model for keyword in ['ref', 'hgp', 'baseline']):
        # Premier modèle (référence) en rouge, les autres avec des couleurs distinctes
        style_colors = ['red'] + colors[1:len(model_names)]
    else:
        # Utiliser les couleurs dans l'ordre
        style_colors = colors[:len(model_names)]
    
    return cycler(marker=markers[:len(model_names)], color=style_colors)

def _calc_track_pt_sigma(pt, ref_pt, bin_pt, fignames=None, verbose=0):
    """(Fonction utilitaire inchangée de votre script) Calculate pT sigma resolution."""
    # ... (Copiez/collez le code complet de la fonction _calc_track_pt_sigma de votre script ici)
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
            if np.sum(hist) == 0: # Si l'histogramme est vide, on ne peut pas fitter
                return [0, 0, 0]
            
            if init_params is None:
                init_params = [len(x), np.quantile(x, 0.5), 10.0]

            try:
                params, covariance = curve_fit(
                    f=_gauss, xdata=center, ydata=hist, sigma=np.sqrt(hist), p0=init_params)
            except RuntimeError: # Le fit n'a pas convergé
                params = init_params
            return params

        if len(x) == 0:
            return -1.0

        hist, bins = np.histogram(x, bins=20, range=(-50.0, 50.0))
        center = (bins[:-1] + bins[1:]) / 2
        mode = center[np.argmax(hist)]
        delta = min(20.0, (np.quantile(x, 0.90) - np.quantile(x, 0.10)) / 2)
        if delta <= 0:
            return np.std(x) if len(x) > 1 else 0.0
        
        xmin1, xmax1 = mode - delta, mode + delta
        if xmax1 <= xmin1:
            return np.std(x) if len(x) > 1 else 0.0
            
        C1, mean1, sigma1 = _fit(x, xmin1, xmax1, nbins=20)
        if sigma1 <= 0:
            return np.std(x) if len(x) > 1 else 0.0

        xmin2, xmax2 = mean1 - 2 * sigma1, mean1 + 2 * sigma1
        if xmax2 <= xmin2:
            return sigma1
        
        C2, mean2, sigma2 = _fit(x, xmin2, xmax2, init_params=(C1, mean1, sigma1), nbins=20)
        return sigma2

    sigma_pt_q1q3, sigma_pt_std, sigma_pt_fit = [np.empty(len(bin_pt)) for _ in range(3)]
    sigma_qopt_q1q3, sigma_qopt_std = [np.empty(len(bin_pt)) for _ in range(2)]

    for i, (low, high) in enumerate(bin_pt):
        idx = np.logical_and(ref_pt > low, ref_pt < high)
        d_pt = (ref_pt - pt)[idx]
        d_pt_qopt = ((1.0 / ref_pt - 1.0 / pt) * np.square(ref_pt))[idx]
        sigma_pt_q1q3[i] = _sigma_from_q1q3(d_pt)
        sigma_qopt_q1q3[i] = _sigma_from_q1q3(d_pt_qopt)
        sigma_pt_std[i] = d_pt.std() if len(d_pt) > 0 else 0
        sigma_qopt_std[i] = d_pt_qopt.std() if len(d_pt_qopt) > 0 else 0
        sigma_pt_fit[i] = _sigma_from_fit(d_pt)

    return {"q1q3": sigma_pt_q1q3, "std": sigma_pt_std, "fit": sigma_pt_fit,
            "qoverpt_q1q3": sigma_qopt_q1q3, "qoverpt_std": sigma_qopt_std}

def calculate_efficiency_by_pt_bin(data, pt_bins):
    """(Fonction utilitaire inchangée) Calculate efficiency for each class in pT bins."""
    efficiency_by_class = {}
    truth, pred, pt_truth, mask_truth, mask_pred = data['truth'], data['pred'], data['pt_truth'], data['mask_truth'], data['mask_pred']
    for class_id in range(5):
        efficiency_bins = []
        for low, high in pt_bins:
            pt_mask = (pt_truth >= low) & (pt_truth < high)
            class_pt_mask = (truth == class_id) & pt_mask & mask_truth
            tp = np.sum(class_pt_mask & (pred == class_id) & mask_pred)
            total_truth = np.sum(class_pt_mask)
            efficiency = tp / total_truth if total_truth > 0 else 0
            efficiency_bins.append(efficiency)
        efficiency_by_class[class_id] = np.array(efficiency_bins)
    return efficiency_by_class

def calculate_efficiency_fake_rate(data):
    """(Fonction utilitaire inchangée) Calculate efficiency and fake rate for each class."""
    efficiency, fake_rate = {}, {}
    truth, pred, mask_truth, mask_pred = data['truth'], data['pred'], data['mask_truth'], data['mask_pred']
    for class_id in range(5):
        tp = np.sum((truth == class_id) & (pred == class_id) & mask_truth & mask_pred)
        total_truth = np.sum((truth == class_id) & mask_truth)
        total_pred = np.sum((pred == class_id) & mask_pred)
        fp = total_pred - tp
        efficiency[class_id] = tp / total_truth if total_truth > 0 else 0
        fake_rate[class_id] = fp / total_pred if total_pred > 0 else 0
    return efficiency, fake_rate

def _custom_hist_multi(ax, data, label, bins, range, color=None):
    """Helper to plot a styled step histogram on given axes."""
    counts, edges = np.histogram(data, bins=bins, range=range)
    # Normalise pour que l'aire sous l'histogramme soit 1
    total = counts.sum()
    if total > 0:
        ax.step(edges[:-1], counts / total, where="post", label=label, 
                linewidth=2.0, color=color)

# =========================================================================================
# ======================== NOUVELLES FONCTIONS DE PLOTTING "MULTI" ========================
# =========================================================================================

def plot_efficiency_by_pt_bin_multi(models_data, **kwargs):
    """Plot efficiency by pT bin for multiple models, creating one plot per class."""
    xticks = [1, 5, 10, 20, 30, 50]
    bin_pt = [(xticks[i], xticks[i + 1]) for i in range(len(xticks) - 1)]
    x = [(high + low) / 2 for low, high in bin_pt]
    xerr = [(high - low) / 2 for low, high in bin_pt]
    
    style_cycler = get_style_cycler(models_data.keys())
    
    figs = {}
    for class_id in range(5):
        if CLASS_NAMES[class_id] == 'muon': continue

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_prop_cycle(style_cycler)
        
        for name, data in models_data.items():
            eff = calculate_efficiency_by_pt_bin(data, bin_pt)
            ax.errorbar(x=x, y=eff[class_id], xerr=xerr, yerr=None, linestyle="none", 
                       label=name, markersize=8, linewidth=2)
        
        ax.set_xlabel('$p_T^{truth}$ [GeV]', fontsize=14)
        ax.set_ylabel('Efficiency', fontsize=14)
        ax.set_title(f'{CLASS_NAMES[class_id].replace("_", " ").title()} Classification Efficiency', fontsize=16)
        ax.grid(True, which='both', alpha=0.4)
        ax.legend(fontsize=12, loc='lower right')
        ax.set_xlim(xticks[0], xticks[-1])
        ax.set_ylim(0, 1.05)
        ax.set_xscale('log')
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        
        plt.tight_layout()
        filename = f'efficiency_by_pt_bin_{CLASS_NAMES[class_id]}.png'
        figs[filename] = fig

    return figs

def plot_residual_multi(models_data, particle_type, var_type, **kwargs):
    """Generic function to plot phi, eta, or pT residuals for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    var_config = {
        'pt': {'range': (-1, 1), 'label': 'Relative $p_T$ Residual ($(pred - truth) / truth)$'},
        'eta': {'range': (-0.4, 0.4), 'label': '$\eta$ Residual $(pred - truth)$'},
        'phi': {'range': (-0.4, 0.4), 'label': '$\phi$ Residual $(pred - truth)$'}
    }
    config = var_config[var_type]
    
    # Obtenir les couleurs personnalisées
    style_cycler = get_style_cycler(models_data.keys())
    colors = [prop['color'] for prop in style_cycler]
    
    for i, (name, data) in enumerate(models_data.items()):
        mask = data['mask_truth_pred_neutral'] if particle_type == 'neutral' else data['mask_truth_pred_charged']
        
        truth_val = data[f'{var_type}_truth'][mask]
        pred_val = data[f'{var_type}_pred'][mask]

        if len(truth_val) == 0: continue

        if var_type == 'pt':
            residual = (pred_val - truth_val) / truth_val
        elif var_type == 'phi':
            residual = pred_val - truth_val
            residual = np.where(residual > np.pi, residual - 2 * np.pi, residual)
            residual = np.where(residual < -np.pi, residual + 2 * np.pi, residual)
        else: # eta
            residual = pred_val - truth_val
            
        _custom_hist_multi(ax, residual, name, bins=51, range=config['range'], 
                          color=colors[i] if i < len(colors) else None)

    ax.set_xlabel(config['label'], fontsize=12)
    ax.set_ylabel('Normalized Density', fontsize=12)
    ax.set_title(f'{var_type.capitalize()} Residual Comparison ({particle_type.capitalize()})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'{var_type}_residual_comparison_{particle_type}.png'
    return {filename: fig}

def plot_efficiency_fake_rate_multi(models_data, **kwargs):
    """Plot efficiency and fake rate comparison for multiple models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    model_names = list(models_data.keys())
    num_models = len(model_names)
    
    # Obtenir les couleurs personnalisées
    style_cycler = get_style_cycler(models_data.keys())
    colors = [prop['color'] for prop in style_cycler]
    
    classes = list(range(5))
    x_pos = np.arange(len(classes))
    width = 0.8 / num_models # Adjust bar width based on number of models
    
    for i, (name, data) in enumerate(models_data.items()):
        eff, fake = calculate_efficiency_fake_rate(data)
        
        offset = width * (i - num_models / 2 + 0.5)
        
        eff_vals = [eff[c] for c in classes]
        fake_vals = [fake[c] for c in classes]
        
        color = colors[i] if i < len(colors) else None
        ax1.bar(x_pos + offset, eff_vals, width, label=name, alpha=0.8, color=color)
        ax2.bar(x_pos + offset, fake_vals, width, label=name, alpha=0.8, color=color)

    ax1.set_ylabel('Efficiency', fontsize=12)
    ax1.set_title('Classification Efficiency by Class', fontsize=14)
    ax1.set_ylim(0, 1.05)
    
    ax2.set_ylabel('Fake Rate', fontsize=12)
    ax2.set_title('Classification Fake Rate by Class', fontsize=14)

    for ax in [ax1, ax2]:
        ax.set_xlabel('Particle Class', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([CLASS_NAMES[c] for c in classes], rotation=45, ha="right")
        ax.legend()
        ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    return {'efficiency_fake_rate_comparison.png': fig}

def plot_pt_sigma_resolution_multi(models_data, particle_type, resolution_type="fit", **kwargs):
    """Plot pT sigma resolution comparison for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    style_cycler = get_style_cycler(models_data.keys())
    ax.set_prop_cycle(style_cycler)
    
    xticks = [10, 15, 20, 30, 50, 200]
    bin_pt = [(xticks[i], xticks[i + 1]) for i in range(len(xticks) - 1)]
    x = [(high + low) / 2 for low, high in bin_pt]
    xerr = [(high - low) / 2 for low, high in bin_pt]

    for name, data in models_data.items():
        mask = data['mask_truth_pred_neutral'] if particle_type == 'neutral' else data['mask_truth_pred_charged']
        
        sigma_results = _calc_track_pt_sigma(
            data['pt_pred'][mask], data['pt_truth'][mask], bin_pt
        )
        sigma_values = sigma_results[resolution_type]
        
        ax.errorbar(x=x, y=sigma_values, xerr=xerr, yerr=None, linestyle="None", 
                   label=name, markersize=8, linewidth=2)

    ax.set_xlabel('Truth $p_T$ [GeV]', fontsize=14)
    ax.set_ylabel(f'Sigma $p_T$ [GeV] (from {resolution_type})', fontsize=14)
    ax.set_title(f'$p_T$ Resolution ({particle_type.capitalize()})', fontsize=16)
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_xscale('log')
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    filename = f'pt_sigma_resolution_comparison_{particle_type}.png'
    return {filename: fig}

def plot_true_prediction_counts_multi(models_data, **kwargs):
    """Bar plot of true vs predicted counts per class for multiple models."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    model_names = list(models_data.keys())
    num_models = len(model_names)
    classes = np.arange(5)
    
    # Obtenir les couleurs personnalisées
    style_cycler = get_style_cycler(models_data.keys())
    colors = [prop['color'] for prop in style_cycler]
    
    # Utiliser HGP (premier modèle dans le dict) comme référence pour le 'Truth'
    ref_data = list(models_data.values())[0]
    truth_counts = np.array([np.sum((ref_data['truth'] == c) & ref_data['mask_truth']) for c in classes])
    
    width = 0.8 / (num_models + 1) # +1 pour la barre 'Truth'
    x = np.arange(len(classes))
    
    # Plot 'Truth' bars
    ax.bar(x - width * num_models / 2, truth_counts, width=width, label='Truth', color='gray', alpha=0.7)
    
    for i, (name, data) in enumerate(models_data.items()):
        offset = width * (i - num_models / 2 + 1)
        # TP = où la classe prédite et la classe truth sont les mêmes
        pred_counts = np.array([np.sum((data['pred'] == c) & (data['truth'] == c) & data['mask_truth_and_pred']) for c in classes])
        color = colors[i] if i < len(colors) else None
        ax.bar(x + offset, pred_counts, width=width, label=f'{name} (TP)', alpha=0.8, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_NAMES[c] for c in classes], rotation=45, ha="right")
    ax.set_ylabel('Particle Count', fontsize=12)
    ax.set_title('True Positives vs. Ground Truth Counts per Class', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    
    return {'true_prediction_counts.png': fig}

def plot_cardinality_error_multi(models_data, **kwargs):
    """
    Plot overall cardinality error and per-pT-bin cardinality error.
    """
    figs = {}
    classes = np.arange(5)
    
    # Obtenir les couleurs personnalisées
    style_cycler = get_style_cycler(models_data.keys())
    colors = [prop['color'] for prop in style_cycler]
    
    # --- Plot 1: Overall Cardinality Error ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    model_names = list(models_data.keys())
    num_models = len(model_names)
    x = np.arange(len(classes))
    width = 0.8 / num_models

    for i, (name, data) in enumerate(models_data.items()):
        truth_counts = np.array([np.sum((data['truth'] == c) & data['mask_truth']) for c in classes])
        pred_counts = np.array([np.sum((data['pred'] == c) & data['mask_pred']) for c in classes])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # Remplacer inf par 0 ou une grande valeur si nécessaire
            diff = (pred_counts - truth_counts) / truth_counts.astype(float)
            diff[np.isinf(diff)] = 0 # Cas où truth_counts est 0
        
        offset = width * (i - num_models / 2 + 0.5)
        color = colors[i] if i < len(colors) else None
        ax1.bar(x + offset, diff, width=width, label=name, alpha=0.8, color=color)

    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([CLASS_NAMES[c] for c in classes], rotation=45, ha="right")
    ax1.set_ylabel('Relative Cardinality Error $(N_{pred} - N_{truth}) / N_{truth}$', fontsize=12)
    ax1.set_title('Overall Relative Cardinality Error per Class', fontsize=14)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    figs['cardinality_error_overall.png'] = fig1
    
    # --- Plot 2: Per-pT bin Cardinality Error ---
    xticks = [1, 5, 10, 20, 30, 50]
    bin_pt = [(xticks[i], xticks[i + 1]) for i in range(len(xticks) - 1)]
    x_mid = [(low + high) / 2 for low, high in bin_pt]
    
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for class_id in classes:
        ax = axes[class_id]
        
        for i, (name, data) in enumerate(models_data.items()):
            diff_bins = []
            for low, high in bin_pt:
                mask_bin = (data['pt_truth'] >= low) & (data['pt_truth'] < high)
                n_truth_bin = np.sum((data['truth'] == class_id) & mask_bin & data['mask_truth'])
                n_pred_bin = np.sum((data['pred'] == class_id) & mask_bin & data['mask_pred'])
                
                if n_truth_bin > 0:
                    diff_bins.append((n_pred_bin - n_truth_bin) / n_truth_bin)
                else:
                    diff_bins.append(np.nan) # Pas de données pour tracer
            
            color = colors[i] if i < len(colors) else None
            ax.plot(x_mid, diff_bins, label=name, color=color, linewidth=2, 
                   marker=style_cycler[i]['marker'] if i < len(style_cycler) else 'o', markersize=6)

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xscale('log')
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('$p_T^{truth}$ [GeV]')
        ax.set_ylabel('Relative Card. Error')
        ax.set_title(CLASS_NAMES[class_id])
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=9)
    
    if len(axes) > len(classes):
        axes[5].set_visible(False) # Cacher le dernier subplot vide
        
    plt.tight_layout()
    figs['cardinality_error_by_pt_bin.png'] = fig2

    return figs

def plot_jet_residual_multi(models_jet_results, var_type):
    """
    Generic function to plot jet phi, eta, or pT residuals for multiple models,
    including the Median (M), Interquartile Range (IQR), and fraction in range (f)
    in the legend.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    var_config = {
        'pt': {'range': (-0.4, 0.4), 'label': r'Jet Relative $p_T$ Residual ($(p_{T}^{reco} - p_{T}^{true}) / p_{T}^{true})$', 'bins': 51},
        'eta': {'range': (-0.1, 0.1), 'label': r'Jet $\eta$ Residual $(\eta^{reco} - \eta^{true})$', 'bins': 51},
        'phi': {'range': (-0.1, 0.1), 'label': r'Jet $\phi$ Residual $(\phi^{reco} - \phi^{true})$', 'bins': 51}
    }
    config = var_config[var_type]
    
    style_cycler = get_style_cycler(models_jet_results.keys())
    colors = [prop['color'] for prop in style_cycler]

    for i, (name, results) in enumerate(models_jet_results.items()):
        if 'residuals' not in results or var_type not in results['residuals']:
            print(f"Warning: No residuals of type '{var_type}' found for model '{name}'. Skipping.")
            continue
        
        residuals = np.array(results['residuals'][var_type]) # Ensure it's a numpy array
        
        if residuals.size == 0:
            print(f"Warning: No matched jets found for model '{name}' to plot {var_type} residuals.")
            continue
        
        # --- KEY CHANGE: Calculate new stats (Median, IQR, Fraction) and create the label ---
        median_val = np.median(residuals)
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        
        # Calculate the fraction of events within the plotting range
        plot_range = config['range']
        fraction_in_range = np.mean((residuals >= plot_range[0]) & (residuals <= plot_range[1]))

        # Create the new label with M, IQR, and f
        label = f'{name}\nM={median_val:.3f}, IQR={iqr:.3f}, f={fraction_in_range:.3f}'
        
        ax.hist(
            residuals,
            bins=config['bins'],
            range=config['range'],
            density=True,
            histtype='step',
            linewidth=2,
            label=label, # Use the new detailed label
            color=colors[i] if i < len(colors) else None
        )

    ax.set_xlabel(config['label'], fontsize=12)
    ax.set_ylabel('Normalized Density', fontsize=12)
    ax.set_title(f'Jet {var_type.capitalize()} Residual Comparison', fontsize=14)
    # Adjust legend properties for better readability if needed
    ax.legend(fontsize=10, title="Model & Stats") 
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    
    filename = f'jet_{var_type}_residual_comparison.png'
    return {filename: fig}

def plot_jet_resolution_vs_pt_multi(models_jet_results):
    """
    Plots jet pT resolution (sigma), its statistical error, and bias (mean), 
    with its statistical error, vs. true jet pT for multiple models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    pt_bins = np.array([25, 50, 75, 100, 125, 150, 175, 200, 225, 250])
    pt_bin_centers = (pt_bins[:-1] + pt_bins[1:]) / 2

    for model_name, results in models_jet_results.items():
        truth_pt = np.array(results['truth_props']['pt'])
        pt_residuals = np.array(results['residuals']['pt'])
        
        if len(truth_pt) == 0:
            continue
            
        resolutions = []
        biases = []
        resolution_errors = []
        bias_errors = []
        
        for i in range(len(pt_bins) - 1):
            low, high = pt_bins[i], pt_bins[i+1]
            mask = (truth_pt >= low) & (truth_pt < high)
            
            # Check if there are enough events in the bin for meaningful stats
            if np.sum(mask) > 1:
                residuals_in_bin = pt_residuals[mask]
                n = len(residuals_in_bin)
                std_dev = np.std(residuals_in_bin)
                
                # Append resolution (std dev) and its error
                resolutions.append(std_dev)
                resolution_errors.append(std_dev / np.sqrt(2 * (n - 1)))
                
                # Append bias (mean) and its error (standard error of the mean)
                biases.append(np.mean(residuals_in_bin))
                bias_errors.append(std_dev / np.sqrt(n))
            else:
                # Append NaN if there are not enough data points in the bin
                resolutions.append(np.nan)
                resolution_errors.append(np.nan)
                biases.append(np.nan)
                bias_errors.append(np.nan)
        
        # Use ax.errorbar to plot with error bars
        ax1.errorbar(pt_bin_centers, resolutions, yerr=resolution_errors, 
                     marker='o', linestyle='-', label=model_name, capsize=3)
        ax2.errorbar(pt_bin_centers, biases, yerr=bias_errors, 
                     marker='s', linestyle='--', label=model_name, capsize=3)

    # Formatting for Resolution Plot
    ax1.set_xlabel('True Jet $p_T$ [GeV]', fontsize=12)
    ax1.set_ylabel('pT Resolution ($\sigma_{pT}$)', fontsize=12)
    ax1.set_title('Jet pT Resolution vs. True pT', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.4)
    ax1.set_ylim(bottom=0)

    # Formatting for Bias Plot
    ax2.set_xlabel('True Jet $p_T$ [GeV]', fontsize=12)
    ax2.set_ylabel('pT Bias ($\mu_{pT}$)', fontsize=12)
    ax2.set_title('Jet pT Bias vs. True pT', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    ax2.axhline(0, color='black', lw=1, linestyle='--')

    plt.tight_layout()
    
    return {'jet_pt_resolution_bias_comparison.png': fig}