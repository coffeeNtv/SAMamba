import numpy as np

import torch
import torch.nn as nn

from .util import initialize_weights
from .util import BilinearFusion
from .util import SNN_Block
from .util import MultiheadAttention
from mamba_ssm import Mamba
import torch.nn.functional as F


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim=256):
        super(SelfAttentionPooling, self).__init__()
        self.query = nn.Parameter(torch.randn(1, input_dim))
        self.fc = nn.Linear(input_dim, input_dim)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        x_proj = self.fc(x)  # [batch_size, seq_length, input_dim]
        attn_scores = torch.matmul(x_proj, self.query.transpose(-2, -1))  # [batch_size, seq_length, 1]
        attn_scores = attn_scores.squeeze(-1)  # [batch_size, seq_length]
        attn_weights = F.softmax(attn_scores, dim=-1) 
        
        x_pool = torch.matmul(attn_weights.unsqueeze(1), x).squeeze(1)  # [batch_size, input_dim]

        return x_pool


class ClusteringLayer(nn.Module):
    def __init__(self, num_features, num_clusters):
        super(ClusteringLayer, self).__init__()
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, num_features))

    def forward(self, x):
        """
        Forward pass of the clustering layer.
        
        Args:
        x : torch.Tensor
            Input tensor of shape (1, n, num_features)
        
        Returns:
        torch.Tensor
            Output tensor of shape (1, num_clusters, num_features)
        """
        # Ensure the input is correctly shaped
        assert x.shape[1] > self.num_clusters and x.shape[2] == self.num_features
        
        # Calculate the distance from each input feature vector to each cluster center
        # x shape: (1, n, num_features)
        # cluster_centers shape: (num_clusters, num_features)
        # Expanded x shape: (1, n, 1, num_features)
        # Expanded cluster_centers shape: (1, 1, num_clusters, num_features)
        x_expanded = x.unsqueeze(2)
        centers_expanded = self.cluster_centers.unsqueeze(0).unsqueeze(0)
        
        # Compute distances
        distances = torch.norm(x_expanded - centers_expanded, dim=3)  # shape: (1, n, num_clusters)
        
        # Find the closest input features to each cluster center
        # We use argmin to find the index of the minimum distance
        _, indices = torch.min(distances, dim=1)  # Closest input feature index for each cluster
        
        # Gather the closest features
        selected_features = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, self.num_features))
        
        return selected_features

class SAMamba(nn.Module):
    def __init__(self,num_pathway,omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, fusion="concat", model_size="small"):
        super(SAMamba, self).__init__()
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        self.num_pathway = num_pathway

        ###
        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)
        # Self-attention pooling for feature
        self.path_att_pooling = SelfAttentionPooling()
        self.gene_att_pooling = SelfAttentionPooling()

        self.clustering = ClusteringLayer(num_features=256, num_clusters=256)

        # Genomic Embedding Network
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        # Pathomics Transformer Decoder
        # Encoder
        self.genomics_encoder = nn.ModuleList([
                            Mamba(d_model=256, # Model dimension d_model
                                d_state=16,  # SSM state expansion factor
                                d_conv=4,    # Local convolution width
                                expand=2,    # Block expansion factor
                                )
                    for i in range(1)])
        # Decoder
        self.genomics_decoder = nn.ModuleList([
                            Mamba(d_model=256, # Model dimension d_model
                                d_state=16,  # SSM state expansion factor
                                d_conv=4,    # Local convolution width
                                expand=2,    # Block expansion factor
                                )
                    for i in range(1)])


        # Pathomics Transformer
        # Encoder
        self.pathomics_encoder = nn.ModuleList([
                            Mamba(d_model=256, # Model dimension d_model
                                d_state=16,  # SSM state expansion factor
                                d_conv=4,    # Local convolution width
                                expand=2,    # Block expansion factor
                                )
                    for i in range(1)])
        # Decoder
        self.pathomics_decoder = nn.ModuleList([
                            Mamba(d_model=256, # Model dimension d_model
                                d_state=16,  # SSM state expansion factor
                                d_conv=4,    # Local convolution width
                                expand=2,    # Block expansion factor
                                )
                    for i in range(1)])

        # P->G Attention
        self.P_in_G_Att = MultiheadAttention(embed_dim=256, num_heads=1)
        self.G_in_P_Att = MultiheadAttention(embed_dim=256, num_heads=1)

        # Classification Layer
        if self.fusion == "concat":
            self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1] * 2, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], hidden[-1]), nn.ReLU()]
            )
        elif self.fusion == "bilinear":
            self.mm = BilinearFusion(dim1=hidden[-1], dim2=hidden[-1], scale_dim1=8, scale_dim2=8, mmhid=hidden[-1])
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        self.classifier = nn.Linear(hidden[-1], self.n_classes)

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        # meta genomics and pathomics features
        x_path = kwargs["x_path"]
        x_omic = [kwargs["x_omics"][i] for i in range(self.num_pathway)] 

        
        #---------------This segment could be remarked/modified for better performance---------------#
        # To save memory, you can also choose a subset of all patches from WSIs
        # as some cases have more than one WSI, the large number of patches will result in OOM
        max_features  = 200000 # can be adjusted by your GPU Memory
        num_features = x_path.size()[0]
        if num_features  > max_features:
            indices = np.random.choice(num_features,size = max_features,replace=False)
            x_path = x_path[indices]


        #------------------------------- embedding  -------------------------------#
        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)
        pathomics_features = self.clustering(pathomics_features) 

        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0)

        #------------------------------- encoder  -------------------------------#
        for g_mamba in self.genomics_encoder:
            genomics_features = g_mamba(genomics_features)

        for p_mamba in self.pathomics_encoder:
            pathomics_features = p_mamba(pathomics_features)

    
        #------------------------------- cross-omics attention  -------------------------------#
        pathomics_in_genomics, path_att_score = self.P_in_G_Att(
            pathomics_features.transpose(0,1),
            genomics_features.transpose(0,1),
            genomics_features.transpose(0,1),
        )  
        genomics_in_pathomics, gene_att_score = self.G_in_P_Att(
            genomics_features.transpose(0,1),
            pathomics_features.transpose(0,1),
            pathomics_features.transpose(0,1),
        ) 

        #------------------------------- decoder  -------------------------------#
        for p_mamba in self.pathomics_decoder:
            pathomics_in_genomics = p_mamba(pathomics_in_genomics.transpose(0,1))

        for g_mamba in self.genomics_decoder:
            genomics_in_pathomics = g_mamba(genomics_in_pathomics.transpose(0,1))


        #------------------------------- fusion  -------------------------------#
        path_fusion = self.path_att_pooling(pathomics_in_genomics)
        gene_fusion = self.gene_att_pooling(genomics_in_pathomics)


        if self.fusion == "concat":
            fusion = self.mm(torch.concat((path_fusion,gene_fusion),dim=1))  # take cls token to make prediction
        elif self.fusion == "bilinear":
            fusion = self.mm(gene_fusion, gene_fusion)  # take cls token to make prediction
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        # predict
        logits = self.classifier(fusion)  # [1, n_classes]

        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S

