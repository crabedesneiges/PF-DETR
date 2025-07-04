import streamlit as st
import os
import glob
from zipfile import ZipFile
from io import BytesIO

# Importez vos modules personnalisés
from data_adapter import (
    DetrDataReader, HGPDataReader, adapt_detr_data, adapt_hgp_data, process_detr, process_hgp, match_truth_to_reco_jets
)
from plotting import (
    plot_efficiency_by_pt_bin_multi,
    plot_residual_multi,
    plot_efficiency_fake_rate_multi,
    plot_pt_sigma_resolution_multi,
    plot_jet_residual_multi,
    plot_jet_resolution_vs_pt_multi,
    # ... autres fonctions multi
)
# Place this function in your app.py
@st.cache_data
def get_jet_matching_results(model_name, file_path, conf_threshold):
    """
    Cached function to run clustering and matching for a single model.
    This avoids re-calculating for every plot.
    """
    st.write(f"Processing jets for {model_name}...")
    
    # Use the processing function to get clustered jets
    # This assumes process_detr/process_hgp are defined as in the previous answer
    def detect_and_process(model_name, inputfile, conf_thresh):
        if 'detr' in model_name.lower():
            return process_detr(inputfile, conf_thresh)
        elif 'hgp' in model_name.lower():
            return process_hgp(inputfile)
        else:
            st.warning(f"Model type for '{model_name}' not auto-detected, attempting DETR.")
            return process_detr(inputfile, conf_thresh)

    # The processing functions return: data, jets_truth, jets_pred, cluster_list_pred, cluster_list_truth
    _, jets_truth, jets_pred, _, _ = detect_and_process(model_name, file_path, None if 'original_refiner' in model_name.lower() else conf_threshold)
    
    # Match jets to get residuals and efficiency
    matched_residuals, truth_matched_props, pred_pts, matching_efficiency = match_truth_to_reco_jets(jets_truth, jets_pred, deltaR_max=0.1)
    
    return {
        'residuals': matched_residuals,
        'truth_props': truth_matched_props,
        'pred_pts': pred_pts,
        'efficiency': matching_efficiency
    }


# --- Configuration ---
st.set_page_config(layout="wide")
st.title("Comparaison de Modèles de Reconstruction de Particules")

# --- Chargement et adaptation des données ---
@st.cache_data
def load_data(file_path, model_type, conf_threshold=0.1):
    """Charge et adapte les données pour un modèle donné. Cache le résultat."""
    try:
        if model_type == 'detr':
            reader = DetrDataReader(file_path)
            return adapt_detr_data(reader, conf_threshold)
        else:
            reader = HGPDataReader(file_path)
            return adapt_hgp_data(reader)
    except Exception as e:
        st.error(f"Erreur lors du chargement de {os.path.basename(file_path)}: {e}")
        return None

# --- Barre latérale ---
with st.sidebar:
    st.header("Configuration")
    NPZ_DIR = st.text_input("Dossier des fichiers .npz", "./web_plot/npz_files")
    HGP_REF_FILE = st.text_input("Fichier de référence HGP", os.path.join(NPZ_DIR, "hgp/original_refiner.npz"))

    if not os.path.exists(HGP_REF_FILE):
        st.error(f"Fichier de référence HGP non trouvé: {HGP_REF_FILE}")
        st.stop()

    detr_files = glob.glob(os.path.join(NPZ_DIR, "*.npz"))
    detr_filenames = [os.path.basename(f) for f in detr_files]

    selected_detr_files = st.multiselect(
        "Choisissez les modèles DETR à comparer",
        options=detr_filenames,
        default=detr_filenames[:1]
    )

    conf_threshold = st.slider(
        "Seuil de confiance DETR",
        min_value=0.0, max_value=1.0, value=0.1, step=0.01
    )

    run_comparison = st.button("Générer les Graphiques", type="primary")
# --- Logique de comparaison et affichage ---
if run_comparison:
    if not selected_detr_files:
        st.warning("Veuillez sélectionner au moins un modèle DETR.")
        st.stop()

    # Chargement des données
    all_models_data = {}
    model_paths = {} # This will now be populated correctly

    with st.spinner("Chargement du modèle de référence HGP..."):
        hgp_data = load_data(HGP_REF_FILE, 'hgp')
        if hgp_data:
            model_name_hgp = "HGP (Ref)"
            all_models_data[model_name_hgp] = hgp_data
            # MODIFIED: Store the path for the HGP model
            model_paths[model_name_hgp] = HGP_REF_FILE

    with st.spinner(f"Chargement de {len(selected_detr_files)} modèle(s) DETR..."):
        for fname in selected_detr_files:
            # Construct the full path for loading and storing
            full_path = os.path.join(NPZ_DIR, fname)
            name = os.path.splitext(fname)[0].replace('_', ' ').title()
            
            data = load_data(full_path, 'detr', conf_threshold)
            if data:
                all_models_data[name] = data
                # MODIFIED: Store the path for the DETR model
                model_paths[name] = full_path

    # --- NEW: JET-LEVEL DATA PROCESSING ---
    # This block will now work because model_paths is populated.
    all_models_jet_results = {}
    with st.spinner("Processing jet-level data for all models..."):
        for name, path in model_paths.items():
            # This calls the new cached helper function
            jet_results = get_jet_matching_results(name, path, conf_threshold)
            if jet_results:
                all_models_jet_results[name] = jet_results

    # Check if enough models were loaded successfully for comparison
    if len(all_models_data) < 1: # Changed to 1, as even a single model can be analyzed
        st.warning("Aucun modèle n'a pu être chargé. Veuillez vérifier les fichiers.")
        st.stop()

    st.success("Données chargées. Génération des graphiques...")

    # Création des figures
    figs_eff = plot_efficiency_by_pt_bin_multi(all_models_data)

    figs_neutral_res_pt = plot_residual_multi(all_models_data, 'neutral', 'pt')
    figs_neutral_res_eta = plot_residual_multi(all_models_data, 'neutral', 'eta')
    figs_neutral_res_phi = plot_residual_multi(all_models_data, 'neutral', 'phi')

    figs_charged_res_pt = plot_residual_multi(all_models_data, 'charged', 'pt')
    figs_charged_res_eta = plot_residual_multi(all_models_data, 'charged', 'eta')
    figs_charged_res_phi = plot_residual_multi(all_models_data, 'charged', 'phi')
    figs_sigma = plot_pt_sigma_resolution_multi(all_models_data, 'neutral')
    figs_sigma_charged = plot_pt_sigma_resolution_multi(all_models_data, 'charged')  # Si nécessaire
    # ... autres figures multi

    # Organisation en onglets
    tabs = st.tabs(["Efficacité", "Résidus", "Résolution par pT"] + ["Efficiency/Fake Rate", "Analyse de Jets"])

    # Onglet Efficacité
    with tabs[0]:
        st.header("Efficacité par bin de pT")
        for title, fig in figs_eff.items():
            st.subheader(title.replace('_', ' ').title())
            st.pyplot(fig)

    # Onglet Résidus pT
    with tabs[1]:
        st.header("Neutral Résidus de pT")
        for title, fig in figs_neutral_res_pt.items():
            st.subheader(title.replace('_', ' ').title())
            st.pyplot(fig)
        st.header("Neutral Résidus de eta")
        for title, fig in figs_neutral_res_eta.items():
            st.subheader(title.replace('_', ' ').title())
            st.pyplot(fig)
        st.header("Neutral Résidus de phi")
        for title, fig in figs_neutral_res_phi.items():
            st.subheader(title.replace('_', ' ').title())
            st.pyplot(fig)
        
        st.header("Charged Résidus de pT")
        for title, fig in figs_charged_res_pt.items():
            st.subheader(title.replace('_', ' ').title())
            st.pyplot(fig)
        st.header("Charged Résidus de eta")
        for title, fig in figs_charged_res_eta.items():
            st.subheader(title.replace('_', ' ').title())
            st.pyplot(fig)
        st.header("Charged Résidus de phi")
        for title, fig in figs_charged_res_phi.items():
            st.subheader(title.replace('_', ' ').title())
            st.pyplot(fig)

    # Onglet Résolution pT
    with tabs[2]:
        st.header("Résolution de pT (σ) Neutral")
        for title, fig in figs_sigma.items():
            st.subheader(title.replace('_', ' ').title())
            st.pyplot(fig)
        st.header("Résolution de pT (σ) charged")
        for title, fig in figs_sigma_charged.items():
            st.subheader(title.replace('_', ' ').title())
            st.pyplot(fig)
    # Onglet Fake Rate (exemple)
    with tabs[3]:
        st.header("Taux de Faux Positifs et d'Efficacité par classe")
        figs_fake = plot_efficiency_fake_rate_multi(all_models_data)
        for title, fig in figs_fake.items():
            st.subheader(title.replace('_', ' ').title())
            st.pyplot(fig)
    with tabs[4]:
        st.header("Analyse Comparative de Jets (Multi-Modèle)")

        if not all_models_jet_results:
            st.warning("Aucune donnée de jet n'a pu être traitée.")
        else:
            


            st.subheader("Distribution des Résidus des Jets")
            fig_pt = plot_jet_residual_multi(all_models_jet_results, 'pt')
            st.pyplot(fig_pt[list(fig_pt.keys())[0]])
            
            fig_eta = plot_jet_residual_multi(all_models_jet_results, 'eta')
            st.pyplot(fig_eta[list(fig_eta.keys())[0]])
            
            fig_phi = plot_jet_residual_multi(all_models_jet_results, 'phi')
            st.pyplot(fig_phi[list(fig_phi.keys())[0]])

            st.subheader("Résolution et Biais du $p_T$ des Jets")
            fig_res_bias = plot_jet_resolution_vs_pt_multi(all_models_jet_results)
            st.pyplot(fig_res_bias[list(fig_res_bias.keys())[0]])
    # Téléchargement ZIP
    zip_buf = BytesIO()
    with ZipFile(zip_buf, 'w') as zf:
        for group in (figs_eff, figs_neutral_res_pt, figs_charged_res_pt, figs_charged_res_phi, figs_charged_res_eta, figs_neutral_res_phi, figs_neutral_res_eta,figs_sigma, figs_fake,figs_sigma_charged):
            for fname, fig in group.items():
                img = BytesIO()
                fig.savefig(img, format='png', dpi=300, bbox_inches='tight')
                zf.writestr(f"{group}_{fname}.png", img.getvalue())

    st.sidebar.download_button(
        label="📥 Télécharger tous les graphiques (.zip)",
        data=zip_buf.getvalue(),
        file_name=f"comparison_plots_conf_{conf_threshold}.zip",
        mime="application/zip"
    )
