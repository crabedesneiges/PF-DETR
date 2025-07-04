import torch
import torch.nn as nn
import torch.nn.functional as F

# Assurez-vous d'avoir installé torch_geometric
try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data, Batch
except ImportError:
    print("PyTorch Geometric n'est pas installé. Veuillez l'installer avec : pip install torch_geometric")
    GATv2Conv = None # Pour éviter une erreur d'importation si la lib est absente

from .mlp import MLP
from .positional_encoding import build_pos_embed_from_eta_phi


class CalorimeterGNNEmbedder(nn.Module):
    """
    Crée des embeddings enrichis pour les cellules et topoclusters en modélisant
    leur relation hiérarchique avec un GNN.
    """
    def __init__(self, num_cell_features, num_topo_features, hidden_dim, gnn_layers=2, gnn_heads=4):
        super().__init__()
        if GATv2Conv is None:
            raise ImportError("PyTorch Geometric est requis pour ce module.")
            
        self.hidden_dim = hidden_dim

        # 1. Projections initiales pour amener les features brutes dans l'espace latent
        self.cell_init_proj = MLP(num_cell_features, hidden_dim, hidden_dim, num_layers=2)
        self.topo_init_proj = MLP(num_topo_features, hidden_dim, hidden_dim, num_layers=2)
        
        # 2. Couches de GNN pour le message passing
        self.gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            # GATv2Conv est une couche d'attention sur graphe robuste
            conv = GATv2Conv(hidden_dim, hidden_dim, heads=gnn_heads, concat=False, dropout=0.1)
            self.gnn_layers.append(conv)
            
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(gnn_layers)])

    def forward(self, cell_feats, topo_feats, cell_mask=None, topo_mask=None):
        """
        Args:
            cell_feats (Tensor): Features des cellules. Shape [bs, num_cell_features, N_cell]
            topo_feats (Tensor): Features des topoclusters. Shape [bs, num_topo_features, N_topo]
            cell_mask (Tensor): Masque des cellules valides. Shape [bs, N_cell]
            topo_mask (Tensor): Masque des topoclusters valides. Shape [bs, N_topo]

        Returns:
            Tuple: (cell_embeddings, topo_embeddings, pos_c, pos_t)
        """
        device = cell_feats.device
        # Transposition pour manipulation facile: [bs, N, C]
        cell_feats = cell_feats.transpose(1, 2)
        topo_feats = topo_feats.transpose(1, 2)
        
        bs, n_cell, _ = cell_feats.shape
        _, n_topo, _ = topo_feats.shape
        
        # Extraire les infos de position pour les positional encodings
        # Assumant que eta/phi sont aux indices 4 et 5 pour les cellules
        # et 1 et 2 pour les topos (à ajuster selon vos données)
        pos_c = build_pos_embed_from_eta_phi(cell_feats[:, :, 4], cell_feats[:, :, 5], self.hidden_dim // 4)
        pos_t = build_pos_embed_from_eta_phi(topo_feats[:, :, 4], topo_feats[:, :, 5], self.hidden_dim // 4)

        # 1. Projections initiales
        cell_embeddings_init = self.cell_init_proj(cell_feats)
        topo_embeddings_init = self.topo_init_proj(topo_feats)

        # 2. Construction du graphe par batch
        # Assumant que la dernière feature de cellule est `cell_topo_idx`
        cell_topo_indices = cell_feats[..., -1].long()
        
        batch_graphs = []
        for i in range(bs):

            # Isoler les éléments valides de l'échantillon i
            c_mask_i = cell_mask[i]
            t_mask_i = topo_mask[i]
            
            valid_cells = cell_embeddings_init[i, c_mask_i]
            valid_topos = topo_embeddings_init[i, t_mask_i]
            
            num_valid_cells = valid_cells.shape[0]
            num_valid_topos = valid_topos.shape[0]

            if num_valid_cells == 0 or num_valid_topos == 0:
                continue # Gérer les cas vides

            # Concaténer les features des noeuds (cellules puis topos)
            node_features = torch.cat([valid_cells, valid_topos], dim=0)

            # Construire les arêtes (edges)
            # L'indice du topo pour chaque cellule
            source_nodes = torch.arange(num_valid_cells, device=device)
            target_nodes = cell_topo_indices[i, c_mask_i] + num_valid_cells # Offset par le nombre de cellules
            
            # VÉRIFICATION EXPLICITE QUI EMPÊCHERA LE CRASH CUDA

            #if target_nodes.numel() > 0:
            #    # Cette assertion va échouer et être attrapée par le 'except'
            #    # si un indice est hors limites.
            #    print(f"Checking befor mask nodes: {cell_topo_indices[i, :].max()} < {topo_embeddings_init[i, :].shape[0]}")
            #    if cell_topo_indices[i, :].max() >= topo_embeddings_init[i, :].shape[0]:
            #        print(f"WARNING: cell_topo_indices[i, :].max() = {cell_topo_indices[i, :].max()} >= topo_embeddings_init[i, :].shape[0] = {topo_embeddings_init[i, :].shape[0]}")
            #    print(f"Checking target nodes: {cell_topo_indices[i, c_mask_i].max()} < {num_valid_topos}")
            #    assert cell_topo_indices[i, c_mask_i].max() < num_valid_topos, "Indice de noeud cible HORS LIMITES !"
            # Créer des arêtes dans les deux sens pour un message passing bidirectionnel
            edge_index_c_to_t = torch.stack([source_nodes, target_nodes])
            edge_index_t_to_c = torch.stack([target_nodes, source_nodes])
            edge_index = torch.cat([edge_index_c_to_t, edge_index_t_to_c], dim=1)
            
            batch_graphs.append(Data(x=node_features, edge_index=edge_index))
            
        if not batch_graphs: # Si le batch est vide
            return cell_embeddings_init, topo_embeddings_init, pos_c, pos_t
        
        # Créer un unique grand graphe déconnecté pour le batch
        graph_batch = Batch.from_data_list(batch_graphs)
        x, edge_index = graph_batch.x, graph_batch.edge_index

        # 3. Message passing avec les couches GNN
        for i, (conv, norm) in enumerate(zip(self.gnn_layers, self.norm_layers)):
            # On ajoute une connexion résiduelle (skip connection)
            if torch.isnan(x).any():
                print(f"WARNING: NaN detected in 'x' before GNN layer {i}!")
            if torch.isinf(x).any():
                print(f"WARNING: Inf detected in 'x' before GNN layer {i}!")
            if torch.isnan(edge_index).any(): # Should ideally not happen for edge_index
                print(f"WARNING: NaN detected in 'edge_index' before GNN layer {i}!")
            try:
                conv_output = conv(x, edge_index)
                # Check output of conv
                
                if torch.isnan(conv_output).any():
                    print(f"WARNING: NaN detected in 'conv_output' after GNN layer {i}!")
                if torch.isinf(conv_output).any():
                    print(f"WARNING: Inf detected in 'conv_output' after GNN layer {i}!")

                norm_output = norm(conv_output)
                if torch.isnan(norm_output).any():
                    print(f"WARNING: NaN detected in 'norm_output' after GNN layer {i}!")
                if torch.isinf(norm_output).any():
                    print(f"WARNING: Inf detected in 'norm_output' after GNN layer {i}!")

                x = x + F.relu(norm_output) # This is your original line

            except Exception as e:
                print(f"ERROR: Exception caught in GNN Layer {i} during conv/norm operation:")
                print(e)
                raise # Re-raise the exception to stop execution and see full stack trace
        
        # 4. Récupérer les embeddings enrichis et les remettre dans le tenseur original
        final_cell_embeddings = cell_embeddings_init.clone()
        final_topo_embeddings = topo_embeddings_init.clone()
        
        # Le `graph_batch.batch` indique à quel échantillon du batch chaque noeud appartient
        # Le `graph_batch.ptr` aide à retrouver les indices
        ptr = graph_batch.ptr
        for i in range(bs):
            # Indices des noeuds pour l'échantillon i
            sample_nodes = x[ptr[i]:ptr[i+1]]
            
            # On doit savoir combien de cellules valides on avait pour cet échantillon
            num_valid_cells_i = cell_mask[i].sum()
            
            # Séparer les noeuds en cellules et topos
            enriched_cells = sample_nodes[:num_valid_cells_i]
            enriched_topos = sample_nodes[num_valid_cells_i:]
            
            # Placer les embeddings enrichis aux bons endroits dans les tenseurs de sortie
            final_cell_embeddings[i, cell_mask[i]] = enriched_cells.to(final_cell_embeddings.dtype)
            final_topo_embeddings[i, topo_mask[i]] = enriched_topos.to(final_topo_embeddings.dtype)
            
        return final_cell_embeddings, final_topo_embeddings, pos_c, pos_t