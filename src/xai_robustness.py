import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. EXPLAINABLE AI (XAI) MODULES
# ============================================================================

def compute_gradcam(model, image_tensor, target_class=None, device="cpu"):
    """
    Computes Grad-CAM heatmap for the custom CNN disaster classifier.
    Highlights spatial radar backscatter features influencing the prediction.
    """
    model.eval()
    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.from_numpy(image_tensor).float()
    x = image_tensor.to(device).unsqueeze(0) if image_tensor.ndim == 3 else image_tensor.to(device)
    x = x.clone().detach().requires_grad_(True)
    
    logits, embed = model(x)
    
    if target_class is None:
        target_class = logits.argmax(dim=-1).item()
        
    score = logits[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)
    
    gradients = model.gradients # (1, channels, H, W)
    activations = model.activations # (1, channels, H, W)
    
    if gradients is None or activations is None:
        return np.zeros((x.shape[-2], x.shape[-1]), dtype=np.float32)
        
    # Spatial global average pooling over gradients to get feature weights
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)
    
    cam = F.interpolate(cam, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    
    denom = cam.max() - cam.min()
    if denom > 1e-8:
        cam = (cam - cam.min()) / denom
    else:
        cam = np.zeros_like(cam)
        
    return cam


def compute_unet_saliency(model, image_tensor, device="cpu"):
    """
    Computes input-gradient saliency for the Attention U-Net segmentation model.
    Highlights sensitive boundary delineation pixels along flood embankments.
    """
    model.eval()
    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.from_numpy(image_tensor).float()
    x = image_tensor.to(device).unsqueeze(0) if image_tensor.ndim == 3 else image_tensor.to(device)
    x = x.clone().detach().requires_grad_(True)
    
    output = model(x)
    score = output.sum()
    model.zero_grad()
    score.backward()
    
    grad = x.grad.detach()
    saliency = torch.norm(grad * x.detach(), dim=1).squeeze().cpu().numpy()
    denom = saliency.max() - saliency.min()
    if denom > 1e-8:
        saliency = (saliency - saliency.min()) / denom
    else:
        saliency = np.zeros_like(saliency)
    return saliency


def get_transformer_attentions(model, input_ids, pad_mask=None, device="cpu"):
    """
    Extracts multi-head self-attention weights from the small Disaster Transformer.
    Returns per-token attention importance scores and the full attention matrix.
    """
    model.eval()
    if isinstance(input_ids, np.ndarray):
        input_ids = torch.from_numpy(input_ids).long()
    if pad_mask is not None and isinstance(pad_mask, np.ndarray):
        pad_mask = torch.from_numpy(pad_mask).bool()
        
    ids = input_ids.to(device).unsqueeze(0) if input_ids.ndim == 1 else input_ids.to(device)
    mask = pad_mask.to(device).unsqueeze(0) if (pad_mask is not None and pad_mask.ndim == 1) else (pad_mask.to(device) if pad_mask is not None else None)
    
    with torch.no_grad():
        logits, embed, layer_attentions = model(ids, pad_mask=mask)
        
    if not layer_attentions:
        return np.ones(ids.shape[-1]) / ids.shape[-1], None
        
    # Take attention matrix from the final encoder layer
    last_layer_attn = layer_attentions[-1] # (1, num_heads, seq_len, seq_len)
    avg_heads = last_layer_attn.mean(dim=1).squeeze(0).cpu().numpy() # (seq_len, seq_len)
    
    token_importance = avg_heads.mean(axis=0)
    denom = token_importance.max() - token_importance.min()
    if denom > 1e-8:
        token_importance = (token_importance - token_importance.min()) / denom
    return token_importance, avg_heads


def compute_temporal_attribution(model, seq_tensor, target_class=None, device="cpu"):
    """
    Gradient-based input feature attribution for environmental hydrological sequences.
    Identifies which time step and variable (river stage, streamflow, velocity) drove the forecast.
    """
    model.eval()
    if isinstance(seq_tensor, np.ndarray):
        seq_tensor = torch.from_numpy(seq_tensor).float()
    seq = seq_tensor.to(device).unsqueeze(0) if seq_tensor.ndim == 2 else seq_tensor.to(device)
    seq = seq.clone().detach().requires_grad_(True)
    
    trend, risk_logits, embed, _ = model(seq)
    
    if target_class is None:
        target_class = risk_logits.argmax(dim=-1).item()
        
    score = risk_logits[0, target_class]
    model.zero_grad()
    score.backward()
    
    grad = seq.grad.detach()
    attr = (grad * seq.detach()).squeeze(0).abs().cpu().numpy() # (seq_len, in_features)
    denom = attr.max() - attr.min()
    if denom > 1e-8:
        attr = (attr - attr.min()) / denom
    return attr


# ============================================================================
# 2. ADVERSARIAL ATTACKS (FGSM / PGD) & SENSOR CORRUPTIONS
# ============================================================================

def fgsm_attack(model, images, labels, epsilon=0.08, criterion=None):
    """
    Fast Gradient Sign Method (FGSM) adversarial attack on disaster imagery.
    x_adv = clip(x + eps * sign(grad_x(L)), 0, 1)
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
        
    images_adv = images.clone().detach().requires_grad_(True)
    logits, _ = model(images_adv)
    loss = criterion(logits, labels)
    
    model.zero_grad()
    loss.backward()
    
    data_grad = images_adv.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0.0, 1.0)
    return perturbed_images.detach()


def pgd_attack(model, images, labels, epsilon=0.08, alpha=0.02, steps=7, criterion=None):
    """
    Projected Gradient Descent (PGD) multi-step iterative adversarial attack.
    Iteratively searches for worst-case perturbations within the epsilon L_inf ball.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
        
    perturbed = images.clone().detach()
    # Random starting point within epsilon ball
    perturbed = perturbed + torch.empty_like(perturbed).uniform_(-epsilon, epsilon)
    perturbed = torch.clamp(perturbed, 0.0, 1.0).detach()
    
    for _ in range(steps):
        perturbed.requires_grad = True
        logits, _ = model(perturbed)
        loss = criterion(logits, labels)
        
        model.zero_grad()
        loss.backward()
        
        grad = perturbed.grad.data
        perturbed = perturbed.detach() + alpha * grad.sign()
        
        eta = torch.clamp(perturbed - images, min=-epsilon, max=epsilon)
        perturbed = torch.clamp(images + eta, min=0.0, max=1.0).detach()
        
    return perturbed


def apply_image_corruptions(image_tensor, corruption_type="gaussian_noise", severity=1.0):
    """
    Simulates real environmental and sensor corruptions:
    - 'gaussian_noise': radar thermal receiver noise
    - 'cloud_blur': atmospheric moisture attenuation
    - 'sensor_dropout': dropped telemetry packets / antenna pulse loss
    """
    img = image_tensor.clone()
    if corruption_type == "gaussian_noise":
        noise = torch.randn_like(img) * (0.10 * severity)
        img = torch.clamp(img + noise, 0.0, 1.0)
    elif corruption_type == "cloud_blur":
        k_size = int(3 + 2 * severity)
        if k_size % 2 == 0:
            k_size += 1
        pad = k_size // 2
        is_3d = (img.ndim == 3)
        inp = img.unsqueeze(0) if is_3d else img
        img = F.avg_pool2d(inp, kernel_size=k_size, stride=1, padding=pad)
        if is_3d:
            img = img.squeeze(0)
    elif corruption_type == "sensor_dropout":
        mask = (torch.rand_like(img) > (0.15 * severity)).float()
        img = img * mask
    return img


def evaluate_robustness_comparison(model, dataloader, device="cpu", epsilon=0.08):
    """
    Evaluates clean accuracy vs FGSM attacked vs PGD attacked vs Corrupted accuracy.
    Measures resilience and performance degradation under noise and adversarial attacks.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    clean_correct = 0
    fgsm_correct = 0
    pgd_correct = 0
    noise_correct = 0
    total = 0
    
    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["severity"].to(device)
        total += labels.size(0)
        
        # 1. Clean
        with torch.no_grad():
            logits, _ = model(images)
            clean_correct += (logits.argmax(dim=-1) == labels).sum().item()
            
        # 2. FGSM Attack
        adv_fgsm = fgsm_attack(model, images, labels, epsilon=epsilon, criterion=criterion)
        with torch.no_grad():
            logits_fgsm, _ = model(adv_fgsm)
            fgsm_correct += (logits_fgsm.argmax(dim=-1) == labels).sum().item()
            
        # 3. PGD Attack
        adv_pgd = pgd_attack(model, images, labels, epsilon=epsilon, alpha=epsilon / 4.0, steps=5, criterion=criterion)
        with torch.no_grad():
            logits_pgd, _ = model(adv_pgd)
            pgd_correct += (logits_pgd.argmax(dim=-1) == labels).sum().item()
            
        # 4. Sensor Corruption (Gaussian Radar Noise)
        corrupted = apply_image_corruptions(images, corruption_type="gaussian_noise", severity=1.0)
        with torch.no_grad():
            logits_noise, _ = model(corrupted)
            noise_correct += (logits_noise.argmax(dim=-1) == labels).sum().item()
            
    return {
        "clean_accuracy": round(float(clean_correct / max(total, 1)), 4),
        "fgsm_accuracy": round(float(fgsm_correct / max(total, 1)), 4),
        "pgd_accuracy": round(float(pgd_correct / max(total, 1)), 4),
        "corrupted_accuracy": round(float(noise_correct / max(total, 1)), 4),
        "fgsm_degradation": round(float((clean_correct - fgsm_correct) / max(total, 1)), 4),
        "pgd_degradation": round(float((clean_correct - pgd_correct) / max(total, 1)), 4),
        "corrupted_degradation": round(float((clean_correct - noise_correct) / max(total, 1)), 4),
    }


# ============================================================================
# VERIFICATION UNIT TEST
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" VERIFYING XAI & ADVERSARIAL ROBUSTNESS SUITE")
    print("="*70)
    
    from models import CustomCNNClassifier, ProposedAttentionUNet, EnvironmentalLSTM, SmallDisasterTransformer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = torch.rand(2, 128, 128).to(device)
    lbl = torch.tensor([1]).to(device)
    
    cnn = CustomCNNClassifier().to(device)
    cam = compute_gradcam(cnn, img, device=device)
    print("1. Grad-CAM Heatmap shape:      ", cam.shape, f"Range: [{cam.min():.2f}, {cam.max():.2f}]")
    
    unet = ProposedAttentionUNet().to(device)
    sal = compute_unet_saliency(unet, img, device=device)
    print("2. Attention U-Net Saliency:    ", sal.shape, f"Range: [{sal.min():.2f}, {sal.max():.2f}]")
    
    fgsm_res = fgsm_attack(cnn, img.unsqueeze(0), lbl, epsilon=0.08)
    print("3. FGSM Perturbed Image shape:  ", fgsm_res.shape)
    
    pgd_res = pgd_attack(cnn, img.unsqueeze(0), lbl, epsilon=0.08, steps=5)
    print("4. PGD Perturbed Image shape:   ", pgd_res.shape)
    
    lstm = EnvironmentalLSTM().to(device)
    seq = torch.rand(24, 5).to(device)
    attr = compute_temporal_attribution(lstm, seq, device=device)
    print("5. Temporal Attribution shape:  ", attr.shape)
    
    trans = SmallDisasterTransformer().to(device)
    tokens = torch.randint(0, 100, (32,)).to(device)
    weights, _ = get_transformer_attentions(trans, tokens, device=device)
    print("6. Transformer Attention weights:", len(weights))
    
    print("\n" + "="*70)
    print(" UNIT TEST PASSED: ALL XAI & ROBUSTNESS ALGORITHMS FUNCTIONAL!")
    print("="*70 + "\n")
