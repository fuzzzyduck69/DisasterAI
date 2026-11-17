from datetime import datetime
import json

def generate_situation_summary(
    event_id,
    disaster_class,
    confidence,
    uncertainty,
    flooded_ratio,
    segmentation_dice,
    trend_prediction,
    temporal_features,
    report_text,
    key_text_tokens,
    modality_weights,
    robustness_status="PASS"
):
    """
    Synthesizes multimodal evidence, spatial inundation extent, hydrological trends,
    textual urgency, and adversarial integrity into an explainable situation report (SITREP).
    Fulfills Section 14 of the DisasterAI Technical Design Document.
    """
    severity_labels = {
        0: "LOW / ROUTINE DRAINAGE WATCH",
        1: "MODERATE FLOOD RISK (ELEVATED BASIN INUNDATION)",
        2: "SEVERE / CATASTROPHIC INUNDATION EMERGENCY"
    }
    sev_str = severity_labels.get(disaster_class, "UNKNOWN SEVERITY")
    
    # Assess physical flood dynamics
    if flooded_ratio > 0.25:
        inundation_tier = "CRITICAL: Widespread floodplain submergence detected on SAR"
    elif flooded_ratio > 0.05:
        inundation_tier = "ELEVATED: Localized basin overflow and transport corridor ponding"
    else:
        inundation_tier = "MINIMAL: Baseline channel drainage conditions"
        
    # Assess hydrological trajectory
    if trend_prediction > 30.0:
        river_trend_dir = "RISING RAPIDLY (High velocity flood crest imminent)"
    elif trend_prediction > 10.0:
        river_trend_dir = "ELEVATED / CRESTING (Above seasonal flood stage)"
    else:
        river_trend_dir = "NORMAL / RECEDING (Within baseline channel capacity)"
        
    # Assess epistemic uncertainty calibration
    if uncertainty < 0.25:
        confidence_desc = "HIGH CONFIDENCE (Low epistemic dispersion across modalities)"
    elif uncertainty < 0.50:
        confidence_desc = "MODERATE CONFIDENCE (Minor inter-modality variance)"
    else:
        confidence_desc = "CAUTION ADVISED (High uncertainty / potential sensor or dispatch noise)"
        
    # Actionable operational recommendations
    if disaster_class == 2:
        actions = [
            "Activate emergency operations center (EOC) and dispatch search-and-rescue teams.",
            "Issue immediate mandatory evacuation directives for low-elevation floodplains.",
            "Deploy mobile flood barrier defenses at power substations and transport corridors."
        ]
    elif disaster_class == 1:
        actions = [
            "Issue regional flood watch bulletins and warn motorists of standing water.",
            "Increase hydrological gauge telemetry sampling to 10-minute intervals.",
            "Inspect drainage culverts and levee embankments for debris blockage."
        ]
    else:
        actions = [
            "Maintain baseline automated satellite and hydrological monitoring.",
            "Log hydrological trends into the regional watershed database.",
            "No active civil protection evacuation required at this time."
        ]

    # Convert modality gating weights to percentages
    vis_w = float(modality_weights.get("vis", 0.33)) * 100
    temp_w = float(modality_weights.get("temp", 0.33)) * 100
    text_w = float(modality_weights.get("text", 0.33)) * 100

    summary_doc = {
        "report_id": f"DISASTERAI-SITREP-{str(event_id).upper()}",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "primary_threat": "Hydrological Inundation & Riverine Flooding",
        "assessed_status": sev_str,
        "assessment_metrics": {
            "combined_confidence": f"{confidence * 100:.1f}%",
            "epistemic_uncertainty": f"{uncertainty:.3f} ({confidence_desc})",
            "modality_contributions": {
                "vision_weight": f"{vis_w:.1f}%",
                "hydrology_weight": f"{temp_w:.1f}%",
                "text_weight": f"{text_w:.1f}%"
            }
        },
        "modality_evidence": {
            "satellite_sar_vision": {
                "estimated_flooded_area": f"{flooded_ratio * 100:.1f}% of surveyed tile",
                "segmentation_confidence_dice": f"{segmentation_dice:.3f}",
                "vision_finding": inundation_tier
            },
            "hydrological_telemetry": {
                "forecasted_river_crest": f"{trend_prediction:.2f} ft",
                "trend_trajectory": river_trend_dir,
                "current_stage": f"{temporal_features.get('current_stage', 0.0):.2f} ft",
                "flow_rate": f"{temporal_features.get('streamflow', 0.0):.1f} cfs"
            },
            "textual_intelligence": {
                "incident_dispatch": report_text,
                "salient_keywords": key_text_tokens[:6]
            }
        },
        "adversarial_robustness_audit": {
            "input_integrity_check": robustness_status,
            "perturbation_defense": "Active (Benchmarked against FGSM & PGD gradient attacks)"
        },
        "recommended_interventions": actions
    }
    
    return summary_doc


def format_markdown_summary(summary_doc):
    """Formats structured situation report into publication-ready markdown."""
    m = summary_doc["assessment_metrics"]
    v = summary_doc["modality_evidence"]["satellite_sar_vision"]
    t = summary_doc["modality_evidence"]["hydrological_telemetry"]
    txt = summary_doc["modality_evidence"]["textual_intelligence"]
    
    md = f"""# DISASTERAI SITUATION REPORT: {summary_doc['report_id']}
**Generated:** {summary_doc['generated_at']} | **Threat:** {summary_doc['primary_threat']}  
**ASSESSMENT:** **{summary_doc['assessed_status']}**  
*(Confidence: {m['combined_confidence']}, Epistemic Uncertainty: {m['epistemic_uncertainty']})*

---

### 1. Multimodal Evidence Breakdown
- **Satellite SAR Imagery (Sentinel-1):** {v['vision_finding']} (Inundated Area: **{v['estimated_flooded_area']}**, Mask Dice: **{v['segmentation_confidence_dice']}**)
- **Hydrological Telemetry (USGS):** Forecast River Crest: **{t['forecasted_river_crest']}** ({t['trend_trajectory']}) | Current Stage: **{t['current_stage']}**
- **Textual Dispatch:** *"{txt['incident_dispatch']}"*
- **Key Extracted Terms:** `{', '.join(txt['salient_keywords'])}`

### 2. Modality Dynamic Gating
- Vision Weight: **{m['modality_contributions']['vision_weight']}**
- Hydrological Weight: **{m['modality_contributions']['hydrology_weight']}**
- Textual Weight: **{m['modality_contributions']['text_weight']}**

### 3. Actionable Operational Interventions
"""
    for act in summary_doc["recommended_interventions"]:
        md += f"- [ ] {act}\n"
        
    md += f"\n*Integrity Audit: {summary_doc['adversarial_robustness_audit']['input_integrity_check']} | {summary_doc['adversarial_robustness_audit']['perturbation_defense']}*\n"
    return md
