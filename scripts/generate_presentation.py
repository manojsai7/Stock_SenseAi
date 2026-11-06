"""
StockSense AI Defense Presentation Generator
Generates a polished 12-slide PowerPoint presentation in 16:9 widescreen format
matching the exact Part 9 specifications.
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]  # completely blank

    # Theme Colors
    BG_DARK = RGBColor(15, 23, 42)       # Slate 900
    CARD_BG = RGBColor(30, 41, 59)       # Slate 800
    CARD_BORDER = RGBColor(51, 65, 85)   # Slate 700
    ACCENT_BLUE = RGBColor(14, 165, 233) # Sky 500
    ACCENT_CYAN = RGBColor(6, 182, 212)  # Cyan 500
    ACCENT_AMBER = RGBColor(245, 158, 11)# Amber 500
    ACCENT_EMERALD = RGBColor(16, 185, 129) # Emerald 500
    TEXT_MAIN = RGBColor(248, 250, 252)  # Slate 50
    TEXT_MUTED = RGBColor(148, 163, 184) # Slate 400
    TEXT_LIGHT = RGBColor(203, 213, 225) # Slate 300

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_DARK
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category_text="STOCKSENSE AI • B.TECH CAPSTONE DEFENSE"):
        # Category Tracker
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_CYAN

        # Title
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.6))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_MAIN

    def add_card(slide, left, top, width, height, title="", border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = border_color
        card.line.width = Pt(1.5)

        if title:
            tbox = slide.shapes.add_textbox(Inches(left + 0.25), Inches(top + 0.2), Inches(width - 0.5), Inches(0.4))
            tf = tbox.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = ACCENT_BLUE

        return card

    # ==================== SLIDE 1: TITLE SLIDE ====================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Accent decorative bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(0.15), Inches(3.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT_CYAN
    bar.line.fill.background()

    # Title & Subtitle box
    tbox = s1.shapes.add_textbox(Inches(1.2), Inches(1.7), Inches(11.0), Inches(3.5))
    tf = tbox.text_frame
    tf.word_wrap = True
    
    p0 = tf.paragraphs[0]
    p0.text = "STOCKSENSE AI"
    p0.font.size = Pt(36)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_CYAN

    p1 = tf.add_paragraph()
    p1.text = "Next-Day Stock Price & Return Prediction with Production vs. Research Deep Learning Dual Architecture"
    p1.font.size = Pt(20)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_MAIN
    p1.space_before = Pt(8)

    p2 = tf.add_paragraph()
    p2.text = "Empirical Financial Forecasting • Stationary Representation • EXP-06 Prospective Evaluation Stream"
    p2.font.size = Pt(13)
    p2.font.color.rgb = TEXT_MUTED
    p2.space_before = Pt(6)

    # Meta card
    add_card(s1, 1.2, 5.0, 10.9, 1.6, "Candidate & Defense Details")
    detail_box = s1.shapes.add_textbox(Inches(1.5), Inches(5.5), Inches(10.3), Inches(0.9))
    dtf = detail_box.text_frame
    dtf.word_wrap = True
    dp1 = dtf.paragraphs[0]
    dp1.text = "Candidate: Final Year Engineering Candidate  |  Roll No: CAPSTONE-2026-CSE-042  |  Department: Computer Science & Engineering"
    dp1.font.size = Pt(12)
    dp1.font.color.rgb = TEXT_LIGHT

    dp2 = dtf.add_paragraph()
    dp2.text = "Supervised By: Department Faculty Committee  |  Date: November 2026  |  Repository: Stock_SenseAi (v2.0-rc2)"
    dp2.font.size = Pt(12)
    dp2.font.color.rgb = ACCENT_BLUE
    dp2.space_before = Pt(4)


    # ==================== SLIDE 2: PROBLEM STATEMENT ====================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "The Financial Time-Series Prediction Challenge")

    # Card 1: Non-Stationarity
    add_card(s2, 0.8, 1.6, 3.6, 5.2, "1. Severe Non-Stationarity")
    box = s2.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(3.1), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    items1 = [
        "Raw asset prices P_t exhibit time-varying distributions, stochastic drift, and regime shifts.",
        "Price levels are non-stationary, causing direct regression to suffer from scale dependence and phantom lag-1 memorization.",
        "Direct Price Prediction Trap: A naive price regression merely learns y_hat = y_{t-1} + epsilon, yielding deceptively low MAE without genuine edge."
    ]
    for it in items1:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(8)

    # Card 2: Market Efficiency
    add_card(s2, 4.85, 1.6, 3.6, 5.2, "2. Market Efficiency & Noise")
    box = s2.shapes.add_textbox(Inches(5.1), Inches(2.2), Inches(3.1), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    items2 = [
        "Weak-Form Efficient Market Hypothesis (EMH): All historical market information is already discounted into current asset prices.",
        "Signal-to-Noise Ratio (SNR) in daily equity returns is exceptionally low (< 0.05).",
        "Overfitting Danger: Complex deep networks easily memorize transient noise or fold-specific micro-trends, failing catastrophically out-of-sample."
    ]
    for it in items2:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(8)

    # Card 3: The StockSense Solution
    add_card(s2, 8.9, 1.6, 3.6, 5.2, "3. The StockSense Formulation")
    box = s2.shapes.add_textbox(Inches(9.15), Inches(2.2), Inches(3.1), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    items3 = [
        "Stationary Target: Predict 1-step ahead stationary return r_{t+1} = (P_{t+1} - P_t) / P_t rather than raw price.",
        "Scale Invariance: Uses relative stationary features (returns, ratios, normalized ATR/OBV) to reduce price scale dependence.",
        "Rigorous Benchmark: Every model is judged directly against Naive Persistence (r_hat = 0, P_hat = P_t) and Linear Ridge.",
        "Dual-Metric Scoring: Track both Price MAE/RMSE and Directional Accuracy / MCC."
    ]
    for it in items3:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(8)


    # ==================== SLIDE 3: SYSTEM ARCHITECTURE ====================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Dual-Engine Architecture: Production vs. Research Challenger")

    add_card(s3, 0.8, 1.6, 5.7, 5.2, "PRODUCTION ENGINE (V1)", ACCENT_EMERALD)
    box = s3.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(5.2), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    v1_items = [
        "Pipeline Version: stationary-return-v1 (Frozen, Protected)",
        "Model Architecture: 1D-CNN + Bidirectional LSTM Hybrid (112,417 parameters)",
        "Weights Path: models_storage_stationary/RELIANCE_NS/best_model.h5",
        "Immutability: SHA-256 fingerprint verified on every startup; strict zero write access during research sprints.",
        "Role: Serves live, audited production inference to client applications, API endpoints, and production dashboard cards.",
        "Serving Status: 100% active, zero downtime, isolated environment."
    ]
    for it in v1_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "✓ " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(7)

    add_card(s3, 6.8, 1.6, 5.7, 5.2, "RESEARCH CHALLENGER ENGINE (V2)", ACCENT_AMBER)
    box = s3.shapes.add_textbox(Inches(7.05), Inches(2.2), Inches(5.2), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    v2_items = [
        "Pipeline Version: exploratory-v2-gru (EXP-06, Seed 42)",
        "Model Architecture: 2-Layer Gated Recurrent Unit (GRU, 44,065 parameters)",
        "Weights Path: models_storage_v2/RELIANCE_NS/best_model.keras",
        "Independent Registry: models_storage_v2/registry_v2.json tracks dedicated scalers, training metadata, and model SHA-256.",
        "Shadow Evaluation: Operates strictly in shadow mode alongside V1. Does NOT replace production without passing prospective gate.",
        "Prospective Evaluation: Real-time tri-model logging (V1, V2, Naive) with SHA-256 audit trail in prospective_predictions_log.csv."
    ]
    for it in v2_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "⚡ " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(7)


    # ==================== SLIDE 4: DATASET & PREPROCESSING ====================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Dataset Integrity, Audit Methodology & Feature Engineering")

    add_card(s4, 0.8, 1.6, 5.7, 5.2, "Audited Dataset & Quarantine Pipeline")
    box = s4.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(5.2), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    d_items = [
        "Asset Under Study: RELIANCE.NS (National Stock Exchange of India, Daily OHLCV).",
        "Raw Ingestion: 1,241 historical daily trading sessions.",
        "Audit Findings: Identified 8 anomalous observations requiring quarantine:",
        "  • 6 Official Exchange Holidays recorded with zero volume.",
        "  • 1 Data Provider Outage (March 18, 2025 Bhavcopy reconciled).",
        "  • 1 Unfinished/Incomplete Candle (October 7, 2026).",
        "Frozen Master Dataset: dataset_v2_audited.csv (1,233 strictly verified trading sessions).",
        "Data Immutability: SHA-256 fingerprint ca9cfb09... recorded in dataset_manifest_v2.json; gitattributes enforces LF."
    ]
    for it in d_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = ("• " if not it.startswith("  ") else "") + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(5)

    add_card(s4, 6.8, 1.6, 5.7, 5.2, "7-Feature Stationary Representation")
    box = s4.shapes.add_textbox(Inches(7.05), Inches(2.2), Inches(5.2), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    f_items = [
        "Sequence Window: 30 consecutive trading sessions (T=30, F=7).",
        "Feature 1: Daily Log Returns — ln(Close_t / Close_{t-1}) (Stationary)",
        "Feature 2: High-to-Low Ratio — ln(High_t / Low_t) (Intraday volatility)",
        "Feature 3: Close-to-Open Ratio — ln(Close_t / Open_t) (Intraday momentum)",
        "Feature 4: Normalized Volume Change — ln(Vol_t / Vol_{t-1})",
        "Feature 5: 14-Day Average True Range (ATR) scaled to Close price",
        "Feature 6: 20-Day Rolling Volatility of Log Returns",
        "Feature 7: Normalized On-Balance Volume (OBV) trend indicator",
        "Strict Normalization: StandardScaler fit strictly on training fold only; zero test-set leakage."
    ]
    for it in f_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(5)


    # ==================== SLIDE 5: EXPERIMENTAL DESIGN ====================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Experimental Protocol: Chronological CV & Leakage Elimination")

    add_card(s5, 0.8, 1.6, 3.6, 5.2, "1. Chronological Folds")
    box = s5.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(3.1), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    items_cv = [
        "Standard K-fold cross validation shuffles samples across time, creating massive lookahead leakage.",
        "We implement 3 Expanding Chronological Folds:",
        "  • Fold 1: Train 60%, Val 20%, Test 20%",
        "  • Fold 2: Train 70%, Val 15%, Test 15%",
        "  • Fold 3: Train 80%, Val 10%, Test 10%",
        "Temporal Integrity: Every test session strictly succeeds all training & validation observations."
    ]
    for it in items_cv:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = ("• " if not it.startswith("  ") else "") + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s5, 4.85, 1.6, 3.6, 5.2, "2. Purge & Embargo")
    box = s5.shapes.add_textbox(Inches(5.1), Inches(2.2), Inches(3.1), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    items_pe = [
        "Sequence Overlap Risk: With window T=30, sequences near split boundaries share historical input bars.",
        "Purging: We discard the 29 sequences spanning the boundary between train and validation sets.",
        "Embargo: A 5-session quarantine window after validation prevents residual auto-correlation spillover into test evaluation.",
        "Zero Information Contamination guaranteed across all experimental splits."
    ]
    for it in items_pe:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s5, 8.9, 1.6, 3.6, 5.2, "3. Evaluation Metrics")
    box = s5.shapes.add_textbox(Inches(9.15), Inches(2.2), Inches(3.1), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    items_m = [
        "Price MAE / RMSE: Reconstructed price error in Rupees: P_{t+1} = P_t * (1 + r_{t+1}).",
        "Directional Accuracy (DA): Percentage of sessions where sign(r_hat) matches sign(r_true).",
        "Matthews Correlation Coefficient (MCC): Measures balanced directional quality regardless of class imbalance.",
        "UP & DOWN Recall: Crucial asymmetric metric to detect one-sided bullish bias in models."
    ]
    for it in items_m:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)


    # ==================== SLIDE 6: MODEL ARCHITECTURE COMPARISON ====================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Architectural Comparison: Inductive Biases & Parameters")

    # 3 model columns
    add_card(s6, 0.8, 1.6, 3.6, 5.2, "GRU Challenger (EXP-06)")
    box = s6.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(3.1), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    items_gru = [
        "Parameters: 44,065 trainable weights (Streamlined).",
        "Layers: 2 Gated Recurrent Unit layers (64 units -> 32 units) + Dropout (0.2) + Dense(1).",
        "Inductive Bias: Gated update & reset mechanisms retain multi-scale momentum without separate memory cell.",
        "Efficiency: 60% fewer parameters than CNN-LSTM; faster convergence, reduced variance on low-SNR financial returns.",
        "Status: Selected Research Challenger."
    ]
    for it in items_gru:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s6, 4.85, 1.6, 3.6, 5.2, "CNN-LSTM Hybrid (V1)")
    box = s6.shapes.add_textbox(Inches(5.1), Inches(2.2), Inches(3.1), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    items_cnn = [
        "Parameters: 112,417 trainable weights.",
        "Layers: Conv1D (filter=64, k=3) -> MaxPooling1D -> Bidirectional LSTM (64 units) -> Dense.",
        "Inductive Bias: Convolution filters detect local candlestick patterns (engulfing, pins) while BiLSTM models sequence context.",
        "Capacity: Higher expressive power, but more susceptible to parameter redundancy in noisy regimes.",
        "Status: Production Standard."
    ]
    for it in items_cnn:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s6, 8.9, 1.6, 3.6, 5.2, "Baselines & Controls")
    box = s6.shapes.add_textbox(Inches(9.15), Inches(2.2), Inches(3.1), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    items_base = [
        "Naive Persistence: r_hat = 0, P_{t+1} = P_t. Represents strict random-walk hypothesis.",
        "Linear Ridge Regressor: L2-penalized linear mapping; establishes benchmark for non-linear deep models.",
        "Standard LSTM: 2-layer LSTM (64 -> 32 units, 52,289 params); tested to isolate gate-architecture difference vs GRU.",
        "Rigorous Rule: Deep models must outperform Naive to justify complexity."
    ]
    for it in items_base:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)


    # ==================== SLIDE 7: OFFLINE EXPERIMENTAL RESULTS ====================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Offline Cross-Fold Performance & Asymmetric Recall Analysis")

    # Performance Table Card
    add_card(s7, 0.8, 1.6, 6.8, 5.2, "3-Fold Cross-Validation Leaderboard")
    box = s7.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(6.3), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    res_text = [
        "Model Leaderboard (Cross-Fold Mean Price MAE):",
        "  1. GRU V2 (EXP-06):       ₹13.29  (Fold 3 Test: ₹12.66)",
        "  2. Naive Persistence:    ₹13.34  (Fold 3 Test: ₹12.71)",
        "  3. LSTM Standard:        ₹13.39  (Fold 3 Test: ₹12.78)",
        "  4. CNN-LSTM (EXP-05):     ₹13.42  (Fold 3 Test: ₹12.82)",
        "  5. Linear Ridge:         ₹14.18  (Fold 3 Test: ₹13.65)",
        "",
        "Key Quantitative Takeaways:",
        "• GRU delivers lowest overall Price MAE across all three chronological development folds.",
        "• Margin over Naive: Only ₹0.05 / share (₹13.29 vs ₹13.34) — exploratory result, not reliable superiority.",
        "• Directional Accuracy: 54.0% with +0.1214 MCC on Fold 3; encouraging exploratory signal.",
        "• All deep architectures outperform Ridge, confirming non-linear feature interaction."
    ]
    for it in res_text:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = it
        p.font.size = Pt(11)
        p.font.color.rgb = ACCENT_CYAN if "GRU V2" in it else TEXT_LIGHT
        p.space_after = Pt(4)

    # Asymmetry Card
    add_card(s7, 7.9, 1.6, 4.6, 5.2, "Directional Asymmetry & Market Bias", ACCENT_AMBER)
    box = s7.shapes.add_textbox(Inches(8.15), Inches(2.2), Inches(4.1), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    asym_items = [
        "Pronounced Asymmetry: Model is substantially better at upward moves than downward moves.",
        "UP Recall: 78.7% (Model successfully captures upward bullish momentum).",
        "DOWN Recall: 32.1% (Model frequently under-predicts market pullbacks and sell-offs).",
        "Macro Cause: The 2024-2026 Indian equity secular bull trend biased gradient updates toward positive return forecasts.",
        "Honest Assessment: Model has not mastered market drawdowns; claiming superiority would be scientifically inaccurate."
    ]
    for it in asym_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)


    # ==================== SLIDE 8: THE HOLDOUT QUESTION & INTEGRITY ====================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Research Integrity: Why the Confirmatory Holdout Gate is Locked")

    add_card(s8, 0.8, 1.6, 5.7, 5.2, "Historical Inspection & Preregistered Criteria")
    box = s8.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(5.2), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    h_items = [
        "Preregistered Evaluation Gate: Requires a minimum of 60-100 genuinely uninspected trading sessions.",
        "The Inspection Reality: The May 14 - October 6, 2026 evaluation window had already been inspected during EXP-05 exploratory research.",
        "Post-Audit Inventory: Following data quarantine and deduplication, only 12 historical sessions remained genuinely uninspected prior to October.",
        "Decision: Strictly declare NO-GO for Confirmatory Historical Evaluation.",
        "Refusal to Cheat: Many academic projects silently test on previously seen data and claim 'state-of-the-art'. We explicitly locked the gate."
    ]
    for it in h_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(7)

    add_card(s8, 6.8, 1.6, 5.7, 5.2, "Dual-Track Methodology & Hardware Transparency")
    box = s8.shapes.add_textbox(Inches(7.05), Inches(2.2), Inches(5.2), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    track_items = [
        "Track 1 (Exploratory Development): Conducted rigorous 3-fold cross-validation on chronological historical splits to identify best architecture (GRU).",
        "Track 2 (Prospective Shadow Evaluation): Freeze the GRU model and log genuine prospective daily forecasts against real-time market settlements.",
        "Runtime Disclosure: The exploratory experiments ran on local CPU (Python 3.13.2, TF 2.20.0, Keras 3.11.2, ~12.99s/fit).",
        "Hardware Honesty: The Colab GPU notebook is fully packaged (stocksense_exp06_source.zip) for scaled multi-asset training, but we do NOT falsely label CPU runs as GPU-trained.",
        "Prospective Gold Standard: True out-of-sample validity can ONLY come from forward-in-time logging."
    ]
    for it in track_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(7)


    # ==================== SLIDE 9: PROSPECTIVE EVALUATION STREAM ====================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Prospective Shadow Serving Engine (EXP-06)")

    add_card(s9, 0.8, 1.6, 5.7, 5.2, "Evaluation Architecture & Logging Protocol")
    box = s9.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(5.2), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    p_items = [
        "Shadow Engine: ProspectiveV2Service generates locked daily predictions without modifying production behavior.",
        "Trio Synchronization: Every market session logs exactly three forecasts simultaneously:",
        "  1. Production V1 (CNN-LSTM)",
        "  2. Challenger V2 (GRU)",
        "  3. Naive Baseline (r=0, Close_t)",
        "Log Destination: research/EXP-06/evaluation/prospective_predictions_log.csv.",
        "Registry Tracker: locked_evaluation_registry_v2.json tracks active evaluation sessions, target dates, and cryptographic hashes.",
        "Idempotency Guarantee: Multiple calls on the same trading cutoff update in-place or reject duplicates; zero duplicate session entries."
    ]
    for it in p_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = ("• " if not it.startswith("  ") else "") + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(5)

    add_card(s9, 6.8, 1.6, 5.7, 5.2, "Live Session Verification (Nov 6 -> Nov 9, 2026)")
    box = s9.shapes.add_textbox(Inches(7.05), Inches(2.2), Inches(5.2), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    live_items = [
        "Cutoff Session: November 6, 2026 (Completed NSE Session)",
        "Input Close Price: ₹1,320.00",
        "Target Forecast Session: November 9, 2026 (Next NSE Trading Session)",
        "",
        "Logged Model Predictions:",
        "• Challenger V2 (GRU): ₹1,322.86 (+0.2169% / +21.7 bps) [BULLISH]",
        "• Production V1 (CNN-LSTM): ₹1,319.33 (-0.0509% / -5.1 bps) [BEARISH]",
        "• Naive Baseline: ₹1,320.00 (0.00% / 0 bps) [NEUTRAL]",
        "",
        "Divergence Insight: V2 and V1 take opposing directional stances, providing a decisive empirical test once Nov 9 market closes."
    ]
    for it in live_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = it
        p.font.size = Pt(11)
        p.font.color.rgb = ACCENT_AMBER if "V2" in it else (ACCENT_EMERALD if "V1" in it else TEXT_LIGHT)
        p.space_after = Pt(4)


    # ==================== SLIDE 10: FULL-STACK IMPLEMENTATION ====================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Full-Stack System Architecture & User Interface")

    add_card(s10, 0.8, 1.6, 3.6, 5.2, "FastAPI High-Speed Backend")
    box = s10.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(3.1), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    b_items = [
        "FastAPI Service with strict Pydantic schemas and typed responses.",
        "/api/research/challenger-prediction/{symbol}: Serves isolated V2 GRU prediction.",
        "/api/research/exp06-comparison: Delivers synchronized 4-model comparison matrix and prospective log count.",
        "Error Guardrails: Validates symbol whitelist (RELIANCE.NS); returns HTTP 404 for untracked equities.",
        "Stale Data Guard: Returns HTTP 503 if market data is out-of-sync."
    ]
    for it in b_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s10, 4.85, 1.6, 3.6, 5.2, "React 18 + Vite Frontend")
    box = s10.shapes.add_textbox(Inches(5.1), Inches(2.2), Inches(3.1), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    f_items = [
        "Component: ResearchComparisonPanel.jsx embedded in production dashboard.",
        "Dedicated Hero Card: Highlights GRU V2 with 'EXPERIMENTAL — SHADOW EVALUATION' badge and CPU disclosure.",
        "4-Model Comparative Registry: Displays Production V1, Challenger V2, EXP-05 Colab, and Naive in parallel.",
        "Forecast Trajectory Chart: Interactive Recharts visualization showing price and return trajectories.",
        "Dynamic Target: Computes target session strictly from latest completed market candle."
    ]
    for it in f_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s10, 8.9, 1.6, 3.6, 5.2, "Deployment Architecture")
    box = s10.shapes.add_textbox(Inches(9.15), Inches(2.2), Inches(3.1), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    d_items2 = [
        "Docker Containerization: Multi-stage Dockerfile packaging models_storage_v2/ and models_storage_stationary/.",
        "Render Cloud Backend: Production deployment configured via render.yaml; preserves PREDICTION_PIPELINE_VERSION: stationary-return-v1.",
        "Vercel Edge Frontend: Optimized Vite build with gzip compression and code splitting.",
        "Zero Environment Cross-Talk: Production pipeline completely decoupled from experimental weights."
    ]
    for it in d_items2:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)


    # ==================== SLIDE 11: ENGINEERING RIGOR ====================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Engineering Rigor: 196 Tests, Hash Auditing & Isolation")

    add_card(s11, 0.8, 1.6, 5.7, 5.2, "Comprehensive Verification Suite")
    box = s11.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(5.2), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    t_items = [
        "196 Automated Tests Passing (100% test suite green):",
        "  • test_v2_serving_and_prospective.py (11 tests): Shadow prediction, prospective logging, symbol isolation, trio comparison.",
        "  • test_exp06_readiness.py (15 tests): Dataset integrity, quarantine compliance, SHA-256 fingerprint verification.",
        "  • test_research_v2_registry.py (16 tests): Schema validation, scaler isolation, model loading.",
        "  • 154 Existing Production Tests: Complete regression coverage of stationary return pipeline, API security, and data transformations.",
        "Vite Frontend Build: 3,185 modules transformed cleanly with 0 errors in 34.25s."
    ]
    for it in t_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = ("• " if not it.startswith("  ") else "") + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s11, 6.8, 1.6, 5.7, 5.2, "Cryptographic & Architectural Guarantees")
    box = s11.shapes.add_textbox(Inches(7.05), Inches(2.2), Inches(5.2), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    g_items = [
        "V1 Production Weights Protected: File models_storage_stationary/RELIANCE_NS/best_model.h5 hash untouched.",
        "V2 Weights Fingerprinted: SHA-256 ffeecbfb... verified against registry_v2.json on every cold start.",
        "Audited Dataset Hash: dataset_manifest_v2.json locks canonical dataset fingerprint (ca9cfb09...).",
        "Single-Equity Isolation: V2 endpoints return clean HTTP 404 for TCS, INFY, etc., preventing unauthorized cross-ticker inference.",
        "No Production Overwrite: Docker and Render environment variables strictly maintain V1 as production default."
    ]
    for it in g_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "✓ " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)


    # ==================== SLIDE 12: FUTURE WORK & CONCLUSION ====================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(s12, "Future Roadmap, Academic Contributions & Conclusion")

    add_card(s12, 0.8, 1.6, 5.7, 5.2, "Prospective Roadmap & Future Research")
    box = s12.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(5.2), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    fut_items = [
        "Confirmatory Gate Graduation: Accumulate 60-100 prospective sessions in prospective_predictions_log.csv.",
        "Formal Diebold-Mariano Test: Statistically test if GRU's ₹0.05 edge over Naive is statistically significant at p < 0.05.",
        "DOWN Recall Remediation: Integrate asymmetric loss functions (e.g. LinEx loss) to penalize missed downward drawdowns.",
        "Multi-Asset Expansion: Scale audited preprocessing pipeline to top 10 Nifty 50 constituents using Google Colab GPU training package.",
        "Attention & TFT Models: Benchmark Temporal Fusion Transformers (TFT) against current GRU/CNN-LSTM baselines."
    ]
    for it in fut_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s12, 6.8, 1.6, 5.7, 5.2, "Capstone Summary & Key Contributions")
    box = s12.shapes.add_textbox(Inches(7.05), Inches(2.2), Inches(5.2), Inches(4.3))
    tf = box.text_frame
    tf.word_wrap = True
    concl_items = [
        "1. Relative Stationary Formulation: Eliminated price scale dependence and trivial lag-1 memorization via 7 relative features.",
        "2. Production vs Research Dual Architecture: Protected production V1 while operationalizing V2 GRU in live shadow mode.",
        "3. Scientific Integrity First: Refused to claim false victory on inspected holdouts; established genuine prospective evaluation protocol.",
        "4. Industry-Standard Engineering: 196 automated tests, Docker packaging, FastAPI backend, React dashboard with live trajectory charts.",
        "Conclusion: StockSense AI bridges the critical gap between academic financial machine learning and safe, transparent production deployment."
    ]
    for it in concl_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = "★ " + it
        p.font.size = Pt(11)
        p.font.color.rgb = ACCENT_CYAN if "Conclusion" in it else TEXT_LIGHT
        p.space_after = Pt(6)

    # Output path
    output_dir = Path("research/presentation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "StockSense_AI_Defense.pptx"
    prs.save(str(output_path))
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    create_deck()

