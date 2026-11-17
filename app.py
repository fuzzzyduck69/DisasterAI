import os
import io
import json
import base64
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_from_directory

from src.models import (
    CustomCNNClassifier,
    ProposedAttentionUNet,
    EnvironmentalLSTM,
    SmallDisasterTransformer,
    MultimodalFusionModel
)
from src.dataset import DisasterTokenizer
from src.xai_robustness import (
    compute_gradcam,
    compute_unet_saliency,
    get_transformer_attentions,
    compute_temporal_attribution,
    fgsm_attack
)
from src.sitrep import generate_situation_summary, format_markdown_summary

app = Flask(__name__, template_folder="templates", static_folder="static")

DATA_DIR = Path("data/cleaned")
OUTPUTS_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"

DEVICE = torch.device("cpu")

# Global models and data cache
models = {}
datasets_cache = {}

def stable_index(key, length, seed=42):
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % max(1, int(length))

def load_all_models():
    print("[DisasterAI Dashboard] Loading neural network checkpoints strictly trained from random initialization...", flush=True)
    
    # 1. Tokenizer
    tok = DisasterTokenizer()
    tok_path = OUTPUTS_DIR / "tokenizer_vocab.json"
    if tok_path.exists():
        tok.load(tok_path)
    models["tokenizer"] = tok

    scaler_path = OUTPUTS_DIR / "hydro_scalers.json"
    if scaler_path.exists():
        models["hydro_scalers"] = json.loads(scaler_path.read_text(encoding="utf-8"))
    else:
        # Compatibility fallback for old checkpoints only. New training creates this file.
        models["hydro_scalers"] = {
            "input_center": [0.0] * 5, "input_scale": [1.0] * 5,
            "target_center_ft": 0.0, "target_scale_ft": 1.0
        }
        print("[WARN] hydro_scalers.json missing; hydrology inference is using identity scaling.", flush=True)

    # 2. Vision CNN
    cnn = CustomCNNClassifier(in_ch=2, num_classes=3, feat_dim=64).to(DEVICE)
    cnn.load_state_dict(torch.load(CHECKPOINT_DIR / "custom_cnn.pt", map_location=DEVICE))
    cnn.eval()
    models["cnn"] = cnn

    # 3. Attention U-Net
    unet = ProposedAttentionUNet(in_ch=2, out_ch=1, base=16).to(DEVICE)
    unet.load_state_dict(torch.load(CHECKPOINT_DIR / "attention_unet.pt", map_location=DEVICE))
    unet.eval()
    models["unet"] = unet

    # 4. Environmental LSTM
    lstm = EnvironmentalLSTM(in_features=5, hidden_dim=64, num_layers=2, out_steps=1, feat_dim=64).to(DEVICE)
    lstm.load_state_dict(torch.load(CHECKPOINT_DIR / "environmental_lstm.pt", map_location=DEVICE))
    lstm.eval()
    models["lstm"] = lstm

    # 5. Small Disaster Transformer
    vocab_sz = tok.vocab_size + 10
    transformer = SmallDisasterTransformer(
        vocab_size=vocab_sz, d_model=64, num_layers=2, num_heads=4, num_classes=3, max_len=32
    ).to(DEVICE)
    transformer.load_state_dict(torch.load(CHECKPOINT_DIR / "disaster_transformer.pt", map_location=DEVICE))
    transformer.eval()
    models["transformer"] = transformer

    # 6. Multimodal Fusion Model
    fusion = MultimodalFusionModel(visual_dim=64, temporal_dim=64, text_dim=64, fusion_dim=96, num_classes=3).to(DEVICE)
    fusion.load_state_dict(torch.load(CHECKPOINT_DIR / "multimodal_fusion.pt", map_location=DEVICE))
    fusion.eval()
    models["fusion"] = fusion

    # 7. Load Data Manifests
    manifest_csv = DATA_DIR / "imagery/clean_manifest.csv"
    if manifest_csv.exists():
        df_manifest = pd.read_csv(manifest_csv)
        datasets_cache["manifest"] = df_manifest
    
    text_csv = DATA_DIR / "text/cleaned_disaster_reports.csv"
    if text_csv.exists():
        datasets_cache["text"] = pd.read_csv(text_csv)

    hydro_seq_file = DATA_DIR / "hydrology/hydro_sequences.npy"
    hydro_meta_file = DATA_DIR / "hydrology/sequence_manifest.csv"
    if hydro_seq_file.exists():
        datasets_cache["hydro_seqs"] = np.load(hydro_seq_file)
    if hydro_meta_file.exists():
        datasets_cache["hydro_meta"] = pd.read_csv(hydro_meta_file)

    print("[DisasterAI Dashboard] All 5 models and dataset manifests loaded successfully!", flush=True)

# Helper function to convert matplotlib figure to base64 string
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02, dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def render_sar_composite(s1_arr):
    # s1_arr: (2, 128, 128) -> normalized [0, 1]
    vv = s1_arr[0]
    vh = s1_arr[1]
    diff = np.clip(np.abs(vv - vh) * 2.0, 0.0, 1.0)
    
    # Create polarimetric RGB composite
    rgb = np.stack([vv, vh, diff], axis=-1)
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(rgb)
    ax.axis("off")
    return fig_to_base64(fig)

def render_mask(mask_arr, is_gt=True):
    # mask_arr: (128, 128)
    fig, ax = plt.subplots(figsize=(3, 3))
    if is_gt:
        # -1 = ignore (dark gray), 0 = dry (dark blue-gray), 1 = flood (bright cyan)
        vis = np.zeros((128, 128, 3), dtype=np.float32)
        vis[mask_arr == 0] = [0.12, 0.16, 0.22] # dry land
        vis[mask_arr == 1] = [0.15, 0.75, 0.95] # flood water
        vis[mask_arr < 0]  = [0.25, 0.25, 0.28] # ignore / nodata
        ax.imshow(vis)
    else:
        # Prediction probability [0, 1]
        im = ax.imshow(mask_arr, cmap="coolwarm", vmin=0.0, vmax=1.0)
    ax.axis("off")
    return fig_to_base64(fig)

def render_gradcam_overlay(s1_arr, cam_heatmap):
    # s1_arr: (2, 128, 128), cam_heatmap: (128, 128)
    vv = s1_arr[0]
    cam_colored = cm.turbo(cam_heatmap)[:, :, :3]
    vv_rgb = np.stack([vv, vv, vv], axis=-1)
    
    blended = 0.5 * vv_rgb + 0.5 * cam_colored
    blended = np.clip(blended, 0.0, 1.0)
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(blended)
    ax.axis("off")
    return fig_to_base64(fig)

def render_adversarial_noise(noise_arr):
    # noise_arr: (2, 128, 128)
    diff = np.abs(noise_arr[0])
    denom = diff.max() - diff.min()
    if denom > 1e-6:
        diff = (diff - diff.min()) / denom
        
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(diff, cmap="magma")
    ax.axis("off")
    return fig_to_base64(fig)


# ============================================================================
# API ROUTES
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chips", methods=["GET"])
def get_chips():
    if "manifest" not in datasets_cache:
        return jsonify({"error": "Manifest not loaded"}), 500
        
    df = datasets_cache["manifest"]
    chips = []
    for _, row in df.iterrows():
        chips.append({
            "chip_id": row["chip_id"],
            "country": row["country"],
            "split": row["split"],
            "flood_ratio": float(row["flood_ratio"]),
            "severity_tier": int(row["severity_tier"]),
            "mean_vv": float(row["mean_vv_norm"]),
            "mean_vh": float(row["mean_vh_norm"])
        })
    return jsonify({"chips": chips})

@app.route("/api/analyze", methods=["POST"])
def analyze_sample():
    try:
        data = request.get_json() or {}
        chip_id = data.get("chip_id")
        epsilon = float(data.get("epsilon", 0.0))
        custom_dispatch = data.get("custom_dispatch", "").strip()

        df_manifest = datasets_cache.get("manifest")
        if df_manifest is None:
            return jsonify({"error": "Dataset manifest unavailable"}), 500
            
        row = df_manifest[df_manifest["chip_id"] == chip_id]
        if len(row) == 0:
            # Fallback to first available chip
            row = df_manifest.iloc[[0]]
            chip_id = row.iloc[0]["chip_id"]
        row = row.iloc[0]

        # 1. Load SAR and Mask
        s1_arr = np.load(row["clean_s1_path"]).astype(np.float32) # (2, 128, 128)
        mask_arr = np.load(row["clean_mask_path"]).astype(np.int64) # (128, 128)
        chip_sev = int(row["severity_tier"])

        img_tensor = torch.from_numpy(s1_arr).unsqueeze(0).to(DEVICE) # (1, 2, 128, 128)
        gt_mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).to(DEVICE)

        # 2. Vision Models Inference
        cnn = models["cnn"]
        unet = models["unet"]

        # Clean CNN inference
        with torch.no_grad():
            clean_logits_c, vis_embed = cnn(img_tensor)
            clean_probs_c = F.softmax(clean_logits_c, dim=-1).squeeze(0).cpu().numpy().tolist()
            pred_class_c = int(np.argmax(clean_probs_c))

        # Grad-CAM heatmap
        cam_heatmap = compute_gradcam(cnn, img_tensor, target_class=pred_class_c, device=DEVICE)

        # Attention U-Net prediction
        with torch.no_grad():
            unet_logits = unet(img_tensor)
            unet_prob = torch.sigmoid(unet_logits).squeeze().cpu().numpy()
            pred_mask = (unet_prob >= 0.5).astype(np.float32)

        # Compute Dice Score
        valid_gt = (mask_arr >= 0)
        gt_binary = (mask_arr == 1)
        intersection = np.logical_and(pred_mask == 1, gt_binary & valid_gt).sum()
        cardinality = pred_mask[valid_gt].sum() + gt_binary[valid_gt].sum()
        dice_score = float((2.0 * intersection + 1e-6) / (cardinality + 1e-6)) if valid_gt.any() else 0.0
        predicted_flood_ratio = float(pred_mask[valid_gt].mean()) if valid_gt.any() else 0.0

        # 3. Adversarial Attack (FGSM) if epsilon > 0
        adv_noise_b64 = None
        perturbed_s1_b64 = None
        adv_probs_c = None
        adv_pred_class = None
        is_attacked = epsilon > 0.0

        if is_attacked:
            label_tensor = torch.tensor([chip_sev], dtype=torch.long, device=DEVICE)
            perturbed_tensor = fgsm_attack(cnn, img_tensor, label_tensor, epsilon=epsilon)
            
            with torch.no_grad():
                adv_logits_c, _ = cnn(perturbed_tensor)
                adv_probs_c = F.softmax(adv_logits_c, dim=-1).squeeze(0).cpu().numpy().tolist()
                adv_pred_class = int(np.argmax(adv_probs_c))
                
            perturbed_np = perturbed_tensor.squeeze(0).cpu().numpy()
            noise_diff = perturbed_np - s1_arr
            adv_noise_b64 = render_adversarial_noise(noise_diff)
            perturbed_s1_b64 = render_sar_composite(perturbed_np)

        # 4. Temporal Hydrology Modality
        lstm = models["lstm"]
        hydro_seqs = datasets_cache["hydro_seqs"]
        hydro_meta = datasets_cache["hydro_meta"]

        # Coherently select hydrological sequence matching severity tier
        hydro_candidates = hydro_meta.copy()
        if "split" in hydro_candidates.columns:
            split_candidates = hydro_candidates[hydro_candidates["split"].astype(str).str.lower() == str(row["split"]).lower()]
            if not split_candidates.empty:
                hydro_candidates = split_candidates
        matching_indices = hydro_candidates.index[hydro_candidates["severity_tier"].astype(int) == chip_sev].to_numpy()
        if len(matching_indices) == 0:
            matching_indices = hydro_candidates.index.to_numpy()
        h_idx = int(matching_indices[stable_index(chip_id, len(matching_indices))])

        env_seq_raw = hydro_seqs[h_idx].astype(np.float32).copy()  # [stage_norm, flow_norm, velocity, raw_stage, raw_flow]
        h_row = hydro_meta.iloc[h_idx]
        current_stage = float(env_seq_raw[-1, 3])
        current_discharge = float(env_seq_raw[-1, 4])
        current_rising_rate = float(env_seq_raw[-1, 2])
        basin_name = str(h_row.get("station_name", "Unknown Basin"))

        scalers = models.get("hydro_scalers", {})
        input_center = np.asarray(scalers.get("input_center", [0.0] * 5), dtype=np.float32)
        input_scale = np.asarray(scalers.get("input_scale", [1.0] * 5), dtype=np.float32)
        input_scale = np.where(input_scale > 1e-6, input_scale, 1.0)
        env_seq = (env_seq_raw - input_center) / input_scale
        seq_tensor = torch.from_numpy(env_seq.astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            trend_pred_norm, lstm_risk_logits, temp_embed, _ = lstm(seq_tensor)
            forecast_peak_ft = float(trend_pred_norm.squeeze().item() * float(scalers.get("target_scale_ft", 1.0)) + float(scalers.get("target_center_ft", 0.0)))
            lstm_probs = F.softmax(lstm_risk_logits, dim=-1).squeeze(0).cpu().numpy().tolist()

        # Temporal attribution - returns (24, 5) matrix; reduce to per-timestep importance
        temp_attr = compute_temporal_attribution(lstm, seq_tensor, target_class=chip_sev, device=DEVICE)
        if isinstance(temp_attr, np.ndarray) and temp_attr.ndim == 2:
            hour_importances = temp_attr.mean(axis=1).tolist()  # avg across 5 features per timestep
        elif isinstance(temp_attr, np.ndarray):
            hour_importances = temp_attr.flatten().tolist()
        else:
            hour_importances = [float(x) for x in temp_attr]

        # Plot the actual 24-step raw observations stored in the selected sequence.
        stage_series = [round(float(v), 2) for v in env_seq_raw[:, 3]]
        discharge_series = [round(float(v), 1) for v in env_seq_raw[:, 4]]
        hours_labels = [f"T-{(24 - i) * 0.25:.2f}h" for i in range(24)]

        # 5. Text Crisis Modality
        tokenizer = models["tokenizer"]
        transformer = models["transformer"]
        df_text = datasets_cache["text"]

        if custom_dispatch:
            dispatch_text = custom_dispatch
        else:
            text_candidates = df_text.copy()
            if "split" in text_candidates.columns:
                split_text = text_candidates[text_candidates["split"].astype(str).str.lower() == str(row["split"]).lower()]
                if not split_text.empty:
                    text_candidates = split_text
            text_pool = text_candidates[text_candidates["severity_tier"].astype(int) == chip_sev]["clean_text"].dropna().tolist()
            if not text_pool:
                text_pool = text_candidates["clean_text"].dropna().tolist()
            if not text_pool:
                raise RuntimeError("No crisis reports available for this dashboard split.")
            dispatch_text = text_pool[stable_index(chip_id, len(text_pool), seed=43)]

        token_ids, pad_mask = tokenizer.encode(dispatch_text)
        token_ids_tensor = torch.from_numpy(token_ids).unsqueeze(0).to(DEVICE)
        pad_mask_tensor = torch.from_numpy(pad_mask).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            tf_logits, text_embed, _ = transformer(token_ids_tensor, pad_mask=pad_mask_tensor)
            tf_probs = F.softmax(tf_logits, dim=-1).squeeze(0).cpu().numpy().tolist()

        token_importance, _ = get_transformer_attentions(transformer, token_ids_tensor, pad_mask=pad_mask_tensor, device=DEVICE)
        
        # Build token pills list
        decoded_words = [tokenizer.idx2word.get(int(t), "<unk>") for t in token_ids if t != tokenizer.pad_id]
        token_pills = []
        for i, word in enumerate(decoded_words):
            score = float(token_importance[i]) if i < len(token_importance) else 0.0
            token_pills.append({"word": word, "attention": round(score, 3)})

        # 6. Tri-Modal Fusion & Dynamic Gating
        fusion = models["fusion"]
        with torch.no_grad():
            fusion_logits, hazard_score, gating_weights, fused_embed = fusion(vis_embed, temp_embed, text_embed)
            fusion_probs = F.softmax(fusion_logits, dim=-1).squeeze(0).cpu().numpy().tolist()
            fusion_pred_class = int(np.argmax(fusion_probs))
            gating = gating_weights.squeeze(0).cpu().numpy().tolist() # [alpha_v, alpha_t, alpha_text]

        # 7. Epistemic Uncertainty Estimation (Monte Carlo Dropout)
        mc_samples = 10
        fusion.train() # Enable dropout
        mc_probs = []
        with torch.no_grad():
            for _ in range(mc_samples):
                logits_mc, _, _, _ = fusion(vis_embed, temp_embed, text_embed)
                probs_mc = F.softmax(logits_mc, dim=-1).squeeze(0).cpu().numpy()
                mc_probs.append(probs_mc)
        fusion.eval() # Reset to eval
        
        mc_probs_arr = np.array(mc_probs) # (10, 3)
        mean_probs = mc_probs_arr.mean(axis=0)
        prob_variance = mc_probs_arr.var(axis=0).sum() # epistemic dispersion
        epistemic_uncertainty = float(prob_variance)
        confidence = float(mean_probs[fusion_pred_class])

        # 8. Automated Civil Defense SITREP Generation
        robustness_status = "NOT_EVALUATED"
        robustness_status_path = OUTPUTS_DIR / "robustness_status.json"
        if robustness_status_path.exists():
            try:
                robustness_data = json.loads(robustness_status_path.read_text(encoding="utf-8"))
                robustness_status = str(robustness_data.get("status", "NOT_EVALUATED"))
            except Exception:
                robustness_status = "UNAVAILABLE"
        top_tokens = [p["word"] for p in sorted(token_pills, key=lambda x: x["attention"], reverse=True)[:5] if p["word"] not in ["[CLS]", "[SEP]", "."]]
        sitrep_dict = generate_situation_summary(
            event_id=f"DISASTERAI-{row['country'].upper()}-{chip_id}",
            disaster_class=fusion_pred_class,
            confidence=confidence,
            uncertainty=epistemic_uncertainty,
            flooded_ratio=predicted_flood_ratio,
            segmentation_dice=dice_score,
            trend_prediction=forecast_peak_ft,
            temporal_features={
                "current_stage": current_stage,
                "discharge_cfs": current_discharge,
                "velocity_ft_hr": current_rising_rate,
                "basin": basin_name
            },
            report_text=dispatch_text,
            key_text_tokens=top_tokens,
            modality_weights={
                "visual": gating[0],
                "hydrological": gating[1],
                "textual": gating[2]
            },
            robustness_status=f"{robustness_status}; inspect measured robustness_metrics.json"
        )
        sitrep_markdown = format_markdown_summary(sitrep_dict)

        # 9. Render Images to Base64
        sar_b64 = render_sar_composite(s1_arr)
        gt_mask_b64 = render_mask(mask_arr, is_gt=True)
        pred_mask_b64 = render_mask(unet_prob, is_gt=False)
        gradcam_b64 = render_gradcam_overlay(s1_arr, cam_heatmap)

        return jsonify({
            "status": "success",
            "chip_id": chip_id,
            "country": row["country"],
            "split": row["split"],
            "severity_tier": chip_sev,
            "cross_modal_pairing_note": "Severity-stratified proxy pairing; not verified same-event matching unless source metadata confirms it.",
            "predictions": {
                "multimodal_class": fusion_pred_class,
                "multimodal_confidence": round(confidence * 100, 1),
                "epistemic_uncertainty": round(epistemic_uncertainty, 4),
                "probabilities": [round(p, 4) for p in fusion_probs],
                "hazard_score": round(float(hazard_score.squeeze().item()), 4),
                "vision_only_class": pred_class_c,
                "vision_only_probs": [round(p, 4) for p in clean_probs_c],
                "lstm_class": int(np.argmax(lstm_probs)),
                "text_class": int(np.argmax(tf_probs)),
                "predicted_flood_ratio_pct": round(predicted_flood_ratio * 100, 2),
                "ground_truth_flood_ratio_pct": round(float(row["flood_ratio"]) * 100, 2),
                "segmentation_dice": round(dice_score, 4),
                "forecast_peak_ft": round(forecast_peak_ft, 2),
                "current_stage_ft": round(current_stage, 2),
                "stage_delta_ft": round(forecast_peak_ft - current_stage, 2),
            },
            "gating_weights": {
                "vision": round(gating[0] * 100, 1),
                "hydrology": round(gating[1] * 100, 1),
                "text": round(gating[2] * 100, 1)
            },
            "adversarial": {
                "is_attacked": is_attacked,
                "epsilon": epsilon,
                "clean_class": pred_class_c,
                "adv_class": adv_pred_class if is_attacked else None,
                "adv_probs": [round(p, 4) for p in adv_probs_c] if is_attacked else None,
                "noise_b64": adv_noise_b64,
                "perturbed_s1_b64": perturbed_s1_b64
            },
            "visualizations": {
                "sar_b64": sar_b64,
                "gt_mask_b64": gt_mask_b64,
                "pred_mask_b64": pred_mask_b64,
                "gradcam_b64": gradcam_b64
            },
            "temporal": {
                "basin_name": basin_name,
                "hours": hours_labels,
                "stage_series": stage_series,
                "discharge_series": discharge_series,
                "hour_importances": [round(x, 3) for x in hour_importances]
            },
            "text": {
                "dispatch": dispatch_text,
                "token_pills": token_pills
            },
            "sitrep": {
                "summary": sitrep_dict,
                "markdown": sitrep_markdown
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    ablation_json = OUTPUTS_DIR / "ablation_study.json"
    robustness_json = OUTPUTS_DIR / "robustness_metrics.json"

    ablation = []
    if ablation_json.exists():
        ablation = json.loads(ablation_json.read_text(encoding="utf-8"))

    robustness = {}
    if robustness_json.exists():
        robustness = json.loads(robustness_json.read_text(encoding="utf-8"))

    return jsonify({
        "ablation": ablation,
        "robustness": robustness
    })

@app.route("/outputs/<path:filename>")
def serve_output_file(filename):
    return send_from_directory(OUTPUTS_DIR, filename)

if __name__ == "__main__":
    load_all_models()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[DisasterAI] Dashboard Server Launching on http://127.0.0.1:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
