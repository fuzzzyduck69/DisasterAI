import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. VISION BACKBONES (Custom CNN Classifier & Attention U-Net)
# Strict Constraint: All weights randomly initialized; ZERO pretrained weights.
# ============================================================================

class ConvBlock(nn.Module):
    """Dual 3x3 Conv + BatchNorm + ReLU block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.block(x)


class CustomCNNClassifier(nn.Module):
    """
    Custom 4-stage convolutional neural network trained from scratch.
    Processes dual-band SAR imagery (VV/VH) for disaster severity classification.
    Includes backward gradient hooks for Grad-CAM explainability.
    """
    def __init__(self, in_ch=2, num_classes=3, feat_dim=64):
        super().__init__()
        self.in_ch = in_ch
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        self.conv1 = ConvBlock(in_ch, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(64, feat_dim)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_embed = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(feat_dim, num_classes)
        
        # State holders for Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward_features(self, x):
        h = self.conv1(x)
        h = self.pool1(h)
        h = self.conv2(h)
        h = self.pool2(h)
        h = self.conv3(h)
        h = self.pool3(h)
        h = self.conv4(h)
        
        if h.requires_grad:
            h.register_hook(self.activations_hook)
        self.activations = h
        
        pooled = self.global_pool(h).flatten(1)
        embed = self.fc_embed(pooled)
        return embed, h

    def forward(self, x):
        embed, _ = self.forward_features(x)
        logits = self.classifier(embed)
        return logits, embed


class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net skip connections.
    Uses coarser gating features from the decoder to selectively weight
    spatial skip features from the encoder, suppressing radar speckle noise.
    """
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inter_ch, 1, 1), nn.Sigmoid())

    def forward(self, gate, skip):
        gate_proj = self.gate(gate)
        skip_proj = self.skip(skip)
        if gate_proj.shape[-2:] != skip_proj.shape[-2:]:
            gate_proj = F.interpolate(gate_proj, size=skip_proj.shape[-2:], mode="bilinear", align_corners=False)
        alpha = self.psi(gate_proj + skip_proj)
        return skip * alpha


class ProposedAttentionUNet(nn.Module):
    """
    Attention-gated U-Net for pixel-level flood inundation segmentation.
    All weights randomly initialized; zero pretrained components.
    """
    def __init__(self, in_ch=2, out_ch=1, base=16):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        
        self.ag3 = AttentionGate(base * 8, base * 4, base * 2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        
        self.ag2 = AttentionGate(base * 4, base * 2, base)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        
        self.ag1 = AttentionGate(base * 2, base, max(4, base // 2))
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        z = self.bottleneck(self.pool(c))
        
        u3 = self.up3(z)
        c_att = self.ag3(z, c)
        d = self.d3(torch.cat([u3, c_att], dim=1))
        
        u2 = self.up2(d)
        b_att = self.ag2(d, b)
        d = self.d2(torch.cat([u2, b_att], dim=1))
        
        u1 = self.up1(d)
        a_att = self.ag1(d, a)
        d = self.d1(torch.cat([u1, a_att], dim=1))
        
        return self.out(d)


# ============================================================================
# 2. TEMPORAL BACKBONE (Environmental Hydrology LSTM Sequence Forecaster)
# ============================================================================

class EnvironmentalLSTM(nn.Module):
    """
    2-Layer LSTM network tracking 24-step hydrological sequences.
    Inputs: [stage_norm, flow_norm, velocity, raw_stage, raw_flow].
    Outputs: forward river crest trend prediction and temporal embedding vector.
    """
    def __init__(self, in_features=5, hidden_dim=64, num_layers=2, out_steps=1, feat_dim=64):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.feat_dim = feat_dim
        
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.fc_embed = nn.Sequential(
            nn.Linear(hidden_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.trend_head = nn.Linear(feat_dim, out_steps)
        self.risk_classifier = nn.Linear(feat_dim, 3)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1] # (batch, hidden_dim)
        embed = self.fc_embed(last_hidden)
        trend_pred = self.trend_head(embed)
        risk_logits = self.risk_classifier(embed)
        return trend_pred, risk_logits, embed, out


# ============================================================================
# 3. TEXT BACKBONE (Small Scratch Transformer Encoder)
# Strict Constraint: NO BERT, NO LLM, NO Pretrained Word Embeddings.
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderBlockWithAttn(nn.Module):
    """Transformer encoder block capturing multi-head attention weights for XAI."""
    def __init__(self, d_model=64, num_heads=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, average_attn_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


class SmallDisasterTransformer(nn.Module):
    """
    Compact 2-layer multi-head self-attention Transformer encoder.
    Trained strictly from scratch on domain crisis dispatches.
    """
    def __init__(self, vocab_size=800, d_model=64, num_layers=2, num_heads=4, num_classes=3, max_len=32):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(0.1)
        
        self.layers = nn.ModuleList([
            TransformerEncoderBlockWithAttn(d_model=d_model, num_heads=num_heads, dim_feedforward=d_model * 2, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.fc_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, pad_mask=None):
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, key_padding_mask=pad_mask)
            attentions.append(attn)
            
        # Masked pooling over non-padded tokens
        if pad_mask is not None:
            mask_expanded = (~pad_mask).unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
            
        embed = self.fc_embed(pooled)
        logits = self.classifier(embed)
        return logits, embed, attentions


# ============================================================================
# 4. MULTIMODAL FUSION & UNCERTAINTY HEAD
# ============================================================================

class MultimodalFusionModel(nn.Module):
    """
    Intermediate representation fusion network.
    Projects Vision, Hydrology, and Text embeddings into an aligned latent space (dim 96),
    applies dynamic modality gating, and estimates uncertainty via Monte Carlo Dropout.
    """
    def __init__(self, visual_dim=64, temporal_dim=64, text_dim=64, fusion_dim=96, num_classes=3):
        super().__init__()
        self.visual_dim = visual_dim
        self.temporal_dim = temporal_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        
        # Alignment projections
        self.vis_proj = nn.Sequential(nn.Linear(visual_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU())
        self.temp_proj = nn.Sequential(nn.Linear(temporal_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU())
        self.text_proj = nn.Sequential(nn.Linear(text_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU())
        
        # Dynamic modality gating network
        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim * 3, 3),
            nn.Softmax(dim=-1)
        )
        
        # Fusion MLP with active dropout for MC Uncertainty estimation
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        
        # Final disaster decision heads
        self.assessment_head = nn.Linear(fusion_dim // 2, num_classes)
        self.hazard_index_head = nn.Sequential(
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, vis_embed=None, temp_embed=None, text_embed=None, active_modalities=("vis", "temp", "text")):
        batch_size = (
            vis_embed.shape[0] if vis_embed is not None else
            (temp_embed.shape[0] if temp_embed is not None else text_embed.shape[0])
        )
        device = (
            vis_embed.device if vis_embed is not None else
            (temp_embed.device if temp_embed is not None else text_embed.device)
        )
        
        # Controlled modality masking for ablation study
        if "vis" in active_modalities and vis_embed is not None:
            v_p = self.vis_proj(vis_embed)
        else:
            v_p = torch.zeros(batch_size, self.fusion_dim, device=device)
            
        if "temp" in active_modalities and temp_embed is not None:
            t_p = self.temp_proj(temp_embed)
        else:
            t_p = torch.zeros(batch_size, self.fusion_dim, device=device)
            
        if "text" in active_modalities and text_embed is not None:
            x_p = self.text_proj(text_embed)
        else:
            x_p = torch.zeros(batch_size, self.fusion_dim, device=device)
            
        concat_feats = torch.cat([v_p, t_p, x_p], dim=1)
        gates = self.gate_net(concat_feats) # (batch, 3)
        
        # Weight modalities dynamically
        v_w = v_p * gates[:, 0:1]
        t_w = t_p * gates[:, 1:2]
        x_w = x_p * gates[:, 2:3]
        
        fused_input = torch.cat([v_w, t_w, x_w], dim=1)
        fused_rep = self.fusion_mlp(fused_input)
        
        logits = self.assessment_head(fused_rep)
        hazard_score = self.hazard_index_head(fused_rep)
        return logits, hazard_score, gates, fused_rep

    def predict_with_uncertainty(self, vis_embed, temp_embed, text_embed, active_modalities=("vis", "temp", "text"), n_samples=25):
        """
        Monte Carlo Dropout sampling:
        Keeps dropout active at inference time to quantify epistemic uncertainty and predictive entropy.
        """
        self.train() # Enable dropout
        logits_samples = []
        hazard_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits, hazard, _, _ = self.forward(vis_embed, temp_embed, text_embed, active_modalities)
                probs = F.softmax(logits, dim=-1)
                logits_samples.append(probs.unsqueeze(0))
                hazard_samples.append(hazard.unsqueeze(0))
                
        all_probs = torch.cat(logits_samples, dim=0) # (samples, batch, num_classes)
        all_hazards = torch.cat(hazard_samples, dim=0) # (samples, batch, 1)
        
        mean_prob = all_probs.mean(dim=0)
        var_prob = all_probs.var(dim=0).sum(dim=-1) # Total predictive variance
        
        mean_hazard = all_hazards.mean(dim=0).squeeze(-1)
        var_hazard = all_hazards.var(dim=0).squeeze(-1)
        
        # Predictive entropy
        entropy = -torch.sum(mean_prob * torch.log(mean_prob + 1e-9), dim=-1)
        confidence = mean_prob.max(dim=-1).values
        
        return {
            "mean_probs": mean_prob,
            "mean_hazard": mean_hazard,
            "variance_hazard": var_hazard,
            "entropy": entropy,
            "confidence": confidence,
            "epistemic_uncertainty": var_prob
        }


# ============================================================================
# VERIFICATION UNIT TEST
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" VERIFYING ALL 5 DISASTERAI ARCHITECTURES (FROM RANDOM INITIALIZATION)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    # Dummy tensors simulating real data batches
    dummy_img = torch.randn(2, 2, 128, 128).to(device)
    dummy_seq = torch.randn(2, 24, 5).to(device)
    dummy_tokens = torch.randint(0, 100, (2, 32)).to(device)
    dummy_pmask = (dummy_tokens == 0).to(device)
    
    # 1. Custom CNN
    cnn = CustomCNNClassifier(in_ch=2, num_classes=3, feat_dim=64).to(device)
    c_logits, c_embed = cnn(dummy_img)
    print("1. Custom CNN Classifier:    Logits", c_logits.shape, "Embed", c_embed.shape)
    
    # 2. Attention U-Net
    unet = ProposedAttentionUNet(in_ch=2, out_ch=1, base=16).to(device)
    u_out = unet(dummy_img)
    print("2. Attention U-Net:          Mask Logits", u_out.shape)
    
    # 3. Environmental LSTM
    lstm = EnvironmentalLSTM(in_features=5, hidden_dim=64, num_layers=2, out_steps=1).to(device)
    trend, r_logits, t_embed, _ = lstm(dummy_seq)
    print("3. Environmental LSTM:        Trend", trend.shape, "Risk Logits", r_logits.shape, "Embed", t_embed.shape)
    
    # 4. Small Disaster Transformer
    transformer = SmallDisasterTransformer(vocab_size=800, d_model=64, num_layers=2, num_heads=4).to(device)
    txt_logits, txt_embed, attns = transformer(dummy_tokens, pad_mask=dummy_pmask)
    print("4. Disaster Transformer:     Logits", txt_logits.shape, "Embed", txt_embed.shape, f"Layers Attn: {len(attns)}")
    
    # 5. Multimodal Fusion Model & MC Uncertainty
    fusion = MultimodalFusionModel(visual_dim=64, temporal_dim=64, text_dim=64, fusion_dim=96, num_classes=3).to(device)
    f_logits, hazard_score, gates, fused_rep = fusion(c_embed, t_embed, txt_embed)
    print("5. Multimodal Fusion Model:  Logits", f_logits.shape, "Hazard", hazard_score.shape, "Gates", gates.shape)
    
    # Test MC Dropout Uncertainty
    unc = fusion.predict_with_uncertainty(c_embed, t_embed, txt_embed, n_samples=10)
    print("   MC Uncertainty Sampling:  Mean Conf", unc["confidence"].shape, "Entropy", unc["entropy"].shape)
    
    print("\n" + "="*70)
    print(" UNIT TEST PASSED: ALL 5 MODELS FUNCTIONAL AND ZERO PRETRAINED WEIGHTS!")
    print("="*70 + "\n")
