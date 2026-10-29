"""Human-Like Industrial Engineering Assistant.
Provides conversational, empathetic, authoritative reliability engineering answers
grounded in hybrid BM25 + dense knowledge retrieval and live Digital Twin telemetry.
Tolerates typos, colloquial phrasing, and provides interactive follow-up suggestions.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional
from .hybrid_retriever import HybridRetriever, NLPTypoNormalizer

SYSTEM_ENGINEERING_PERSONA = """You are a Senior Industrial Reliability Specialist & Tribology Engineering Assistant for an enterprise Condition Monitoring Digital Twin platform.

Your communication style:
1. Conversational & Empathetic: Speak like an experienced, helpful reliability engineer walking alongside a technician in the plant. Acknowledge what the user is experiencing.
2. Grounded & Exact: Use the live machine telemetry (model_result) as ground truth. Never contradict the predicted fault class or invent imaginary sensor values.
3. Structured & Clear: Use clear Markdown with headings (###), bullet points, bold key terms, and step-by-step checklists.
4. Mathematical & Physics-Informed: When discussing characteristic frequencies (BPFO, BPFI, BSF, FTF), provide the exact kinematic formulas rendered in KaTeX LaTeX ($...$ or $$...$$).
5. Standards-Compliant: Reference ISO 10816-3 vibration severity limits (Zone A: Good, Zone B: Acceptable, Zone C: Alert, Zone D: Critical Danger) where relevant.
6. Friendly Decision Support: Emphasize that recommendations provide expert decision support to assist qualified plant engineers.
"""


def generate_human_response(question: str, result: Optional[Dict[str, Any]], contexts: List[Dict[str, Any]], corrections: List[str]) -> Dict[str, Any]:
    """Generates a comprehensive, human-like technical response grounded in telemetry and RAG."""
    q_lower = question.lower()
    parts = []

    # 1. Natural Empathetic Opening & Interpretation
    if corrections:
        parts.append(f"*(Interpreted technical query: {', '.join(corrections)})*\n")

    fault = result.get("fault", "healthy") if result else "healthy"
    conf = result.get("confidence", 0.0) if result else 0.0
    health = result.get("health", 100.0) if result else 100.0
    alert_lvl = result.get("alert", {}).get("level", "normal") if result else "normal"
    rpm = result.get("sensors", {}).get("rpm", 1797.0) if result else 1797.0
    rms = result.get("sensors", {}).get("vibration_rms", 0.05) if result else 0.05
    char_hz = result.get("physics", {}).get("characteristic_hz", {}) if result else {}
    supports_phys = result.get("physics", {}).get("supports_prediction") if result else None

    fault_titles = {
        "healthy": "Healthy Baseline Operation",
        "inner_race": "Inner Raceway Defect (BPFI)",
        "outer_race": "Outer Raceway Defect (BPFO)",
        "ball": "Rolling Element Spall / Defect (BSF)"
    }
    current_title = fault_titles.get(fault, fault.replace("_", " ").title())

    # Greeting / Opening
    if any(g in q_lower for g in ["hi", "hello", "hey", "good morning", "greetings"]):
        parts.append(f"Hello! I am your Senior Reliability & Diagnostics Assistant. I'm actively monitoring your digital twin asset (**B001**). How can I assist with your vibration telemetry, kinematic analysis, or maintenance protocol today?")
    elif any(w in q_lower for w in ["what to do", "fix", "repair", "replace", "action", "maintenance", "troubleshoot"]):
        parts.append(f"### Maintenance & Action Protocol for Asset B001\n"
                     f"Based on the active telemetry, the bearing is evaluated in **{alert_lvl.upper()}** condition with an estimated health index of **{health:.1f}/100** ({current_title}). Here is the recommended maintenance action plan:")
        
        parts.append(
            "\n#### 1. Immediate Field Verification Checklist\n"
            "- [ ] **Acoustic & Thermal Survey**: Inspect bearing housing with an infrared thermometer. Normal operating range is below 70°C; readings > 80°C indicate severe friction or lubrication starvation.\n"
            "- [ ] **Lubrication Inspection**: Verify grease condition (color, contamination, consistency). Look for blackening, hardening, or metallic particulate.\n"
            "- [ ] **Mounting & Looseness Check**: Ensure housing hold-down bolts are torqued to specification and check for soft foot or mechanical play.\n"
            "- [ ] **Shaft Alignment**: Verify angular and radial alignment between the motor and driven load using a laser alignment tool (target tolerance < 0.05 mm).\n"
        )
        if alert_lvl in ["degradation", "critical"]:
            parts.append(
                "#### 2. Corrective Schedule\n"
                f"- **Priority**: High. Plan replacement of bearing during the next scheduled downtime window.\n"
                f"- **Precaution**: Reduce operating speed or load if vibration RMS exceeds 0.28 g (ISO 10816 Zone C/D boundary).\n"
                f"- **Procurement**: Standard replacement bearing: SKF 6205-2RS1 C3 (Clearance Group C3)."
            )
        else:
            parts.append(
                "#### 2. Monitoring Routine\n"
                "- Continue continuous high-frequency envelope monitoring.\n"
                "- Repeat spectrum measurement at identical RPM after 24 operating hours to observe degradation slope."
            )

    elif any(w in q_lower for w in ["bpfo", "bpfi", "bsf", "ftf", "formula", "calculate", "frequency", "kinematic", "math"]):
        fr = rpm / 60.0
        parts.append(f"### Analytical Kinematics & Characteristic Pass Frequencies\n"
                     f"For rolling element bearings, characteristic frequencies describe the rate at which rolling elements strike localized defects. For shaft rotational speed $f_r = \\frac{{\\text{{RPM}}}}{{60}} = \\frac{{{rpm:.0f}}}{{60}} = {fr:.2f}\\,\\text{{Hz}}$:\n")
        
        parts.append(
            "$$\\begin{aligned}\n"
            "\\text{BPFO} &= \\frac{n}{2} f_r \\left(1 - \\frac{d}{D_p}\\cos\\alpha\\right) &&\\text{(Outer Race Pass Frequency)} \\\\\n"
            "\\text{BPFI} &= \\frac{n}{2} f_r \\left(1 + \\frac{d}{D_p}\\cos\\alpha\\right) &&\\text{(Inner Race Pass Frequency)} \\\\\n"
            "\\text{BSF}  &= \\frac{D_p}{2d} f_r \\left[1 - \\left(\\frac{d}{D_p}\\cos\\alpha\\right)^2\\right] &&\\text{(Ball Spin Frequency)} \\\\\n"
            "\\text{FTF}  &= \\frac{1}{2} f_r \\left(1 - \\frac{d}{D_p}\\cos\\alpha\\right) &&\\text{(Cage / Train Frequency)}\n"
            "\\end{aligned}$$\n"
        )
        if char_hz:
            parts.append(f"**Theoretical Frequencies Calculated for Current Speed ({rpm:.0f} RPM)**:\n"
                         f"- **BPFO**: `{char_hz.get('BPFO', 0):.2f} Hz` (Outer Race)\n"
                         f"- **BPFI**: `{char_hz.get('BPFI', 0):.2f} Hz` (Inner Race)\n"
                         f"- **BSF**: `{char_hz.get('BSF', 0):.2f} Hz` (Rolling Element)\n"
                         f"- **FTF**: `{char_hz.get('FTF', 0):.2f} Hz` (Cage)\n"
                         f"- **Shaft ($f_r$)**: `{fr:.2f} Hz`")

    elif any(w in q_lower for w in ["why", "explain", "diagnosis", "predict", "evidence", "sure", "confidence"]):
        parts.append(f"### Diagnostic Evidence & AI Attribution Analysis\n"
                     f"The multi-layer neural network diagnosed **{current_title}** with **{conf:.1%} confidence**.\n")
        
        if supports_phys is True:
            parts.append("✅ **Physics Agreement**: The Hilbert envelope spectrum demonstrates a distinct, resonant spectral peak coinciding with the theoretical characteristic fault frequency within the ±3% tolerance window.")
        elif supports_phys is False:
            parts.append("⚠️ **Physics Discrepancy**: While the statistical classifier predicts a defect, prominent harmonic peaks are not clearly distinguished above the 3.0× median noise floor. A repeat measurement is advised.")
        else:
            parts.append("ℹ️ **Baseline Assessment**: The machine exhibits healthy harmonic behavior consistent with baseline factory acceptance criteria.")

        parts.append(f"\n**Telemetry Signature Readout**:\n"
                     f"- **Vibration RMS**: `{rms:.4f} g`\n"
                     f"- **Operational Speed**: `{rpm:.0f} RPM`\n"
                     f"- **Alert Status**: `{alert_lvl.upper()}`\n"
                     f"- **Health Index**: `{health:.1f} / 100`")

    elif any(w in q_lower for w in ["iso", "standard", "limits", "vibration limit", "threshold", "safe", "dangerous"]):
        parts.append("### ISO 10816-3 Vibration Severity Evaluation\n"
                     "Under **ISO 10816-3** (Mechanical vibration evaluation for industrial machines on rigid or flexible supports):\n\n"
                     "| Vibration Velocity (mm/s RMS) | Severity Class | Operational Status |\n"
                     "| :--- | :--- | :--- |\n"
                     "| **< 1.4 mm/s** | **Zone A** | Newly commissioned / Excellent |\n"
                     "| **1.4 – 2.8 mm/s** | **Zone B** | Acceptable for unrestricted continuous operation |\n"
                     "| **2.8 – 4.5 mm/s** | **Zone C** | Warning: Unsatisfactory for long-term continuous service |\n"
                     "| **> 4.5 mm/s** | **Zone D** | Critical: Danger of immediate machine damage |\n\n"
                     f"Current machine vibration RMS is **{rms:.3f} g**, corresponding to **Zone { 'A' if rms < 0.1 else 'B' if rms < 0.2 else 'C' if rms < 0.35 else 'D'}**.")
    else:
        # General response connecting knowledge base and telemetry
        parts.append(f"### Condition Monitoring Assessment for Asset B001\n"
                     f"The digital twin reports active state: **{current_title}** ({conf:.0%} confidence, Health: {health:.1f}/100).\n")
        if contexts:
            parts.append("#### Grounded Engineering Documentation Notes:")
            for c in contexts[:2]:
                snippet = " ".join(c["text"].split())[:280]
                parts.append(f"- **[{c['source']}]**: {snippet}...")

    # 4. Reference Sources Section
    sources = sorted({c["source"] for c in contexts}) if contexts else []
    
    # 5. Suggested Next Queries
    suggestions = [
        "What is the step-by-step relubrication checklist?",
        "Explain BPFO and BPFI kinematic equations",
        "How is the Mahalanobis health index calculated?"
    ]
    if fault != "healthy":
        suggestions[0] = f"Recommended replacement procedure for {current_title}"

    return {
        "answer": "\n\n".join(parts),
        "sources": sources,
        "corrections": corrections,
        "suggestions": suggestions,
        "provider": "hybrid-bm25-rag"
    }


class HumanAssistant:
    """Enterprise RAG Engineering Assistant with BM25 Lexical Analysis and Typo Tolerance."""

    def __init__(self, retriever: HybridRetriever, top_k: int = 4):
        self.retriever = retriever
        self.k = top_k
        self.provider = "hybrid-bm25-rag"

    def answer(self, question: str, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # 1. Search hybrid retriever (BM25 + Dense RRF + NLP Typo Correction)
        search_results = self.retriever.search(question, k=self.k)
        
        # 2. Extract corrections made
        corrections = []
        if search_results and "corrections" in search_results[0]:
            corrections = search_results[0]["corrections"]

        # 3. Generate structured human-like engineering answer
        return generate_human_response(question, result, search_results, corrections)
