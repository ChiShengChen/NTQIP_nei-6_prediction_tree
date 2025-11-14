#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train an XGBoost model on the EDAS dataset from a single CSV.

Features:
- PM-only style features based on mapped EDAS columns
- Adds requested categorical features:
  - Injury Type (Blunt, Penetrating)
  - Transfer In (Yes/No)
  - PHAS_INTUB_YNS_L_AS_TEXT (Yes, No, Unk, NaN)
- Includes ΔSBP feature: sbp_minus_emssbp_ge20 = 1 if (SBP - EMSSBP) >= 20 else 0

Assumptions:
- By default, NEI-6 label is auto-generated from EDAS columns in the same CSV.
- Alternatively, you can use a provided binary label column with --label-source column and --label-column.
- Column mapping below remaps EDAS columns to canonical names used by feature engineering.
"""

import os
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split


# Remap EDAS columns to canonical names used by feature creation
COLUMN_MAPPING: Dict[str, str] = {
    'Age': 'AGEyears',
    'Gender': 'SEX',
    'EDAS_GCS': 'TOTALGCS',
    'EDAS_SBP': 'SBP',
    'EDAS_PULSE': 'PULSERATE',
    'EDAS_TEMP': 'TEMPERATURE',
    'ARR_TIME': 'ARRIVALTIME',
    'ARR_HOURS': 'HOSPITALARRIVALHRS',
    'PHAS_GCSSC_L': 'EMSTOTALGCS',
    'PHAS_SBPS_L': 'EMSSBP',
    'PHAS_PULSES_L': 'EMSPULSERATE',
    'PHAS_URRS_L': 'EMSRESPIRATORYRATE',
    'PHAS_SAO2S_L': 'EMSPULSEOXIMETRY',
    'PHAS_GCS_EOS_L': 'EMSGCSEYE',
    'PHAS_GCS_MRS_L': 'EMSGCSMOTOR',
    'PHAS_GCS_VRS_L': 'EMSGCSVERBAL',
    # '--could not find column--': 'PREHOSPITALCARDIACARREST',  # left intentionally unmapped
}

# Canonical names for added categorical features
CANON_INJURY_TYPE = 'InjuryType'  # expected values like 'Blunt', 'Penetrating'
CANON_TRANSFER_IN = 'TransferIn'  # expected values like 'Yes', 'No'
CANON_INTUB_TEXT = 'PHAS_INTUB_YNS_L_AS_TEXT'  # expected values: 'Yes', 'No', 'unk', or NaN


def coalesce_column(df: pd.DataFrame, candidates: List[str], new_name: str) -> None:
    """
    If any candidate column exists (case-sensitive), create/rename to new_name.
    Leaves existing new_name intact if already present.
    """
    if new_name in df.columns:
        return
    for c in candidates:
        if c in df.columns:
            df.rename(columns={c: new_name}, inplace=True)
            return


def remap_edas_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply COLUMN_MAPPING if source columns exist.
    Also coalesce the additional requested categorical features into canonical names.
    """
    df = df.copy()
    # Apply mapping (only for existing columns)
    present_map = {src: tgt for src, tgt in COLUMN_MAPPING.items() if src in df.columns}
    if present_map:
        df = df.rename(columns=present_map)

    # Coalesce additional requested categorical features
    coalesce_column(df, ['Injury Type', 'Injury_Type', 'injury_type'], CANON_INJURY_TYPE)
    coalesce_column(df, ['Transfer In', 'Transfer_In', 'transfer_in'], CANON_TRANSFER_IN)
    coalesce_column(df, ['PHAS_INTUB_YNS_L_AS_TEXT', 'phas_intub_yns_l_as_text'], CANON_INTUB_TEXT)

    return df


def _collect_proc_codes_from_row(row: pd.Series, proc_code_cols: List[str]) -> set:
    codes = set()
    for c in proc_code_cols:
        if c not in row or pd.isna(row[c]):
            continue
        val = row[c]
        if isinstance(val, (int, float)) and not pd.isna(val):
            s = str(int(val))
            if s:
                codes.add(s.strip().upper())
        else:
            # Handle None, NaN, or empty values
            if val is None or pd.isna(val):
                continue
            s = str(val)
            if not s or s.lower() in ('none', 'nan', ''):
                continue
            # split lists like "code1;code2, code3 | code4"
            parts = []
            found_sep = False
            for sep in [';', ',', '|']:
                if sep in s:
                    parts = [p.strip() for p in s.split(sep) if p.strip()]
                    found_sep = True
                    break
            if found_sep and parts:
                for p in parts:
                    code = p.strip().upper()
                    if code:
                        codes.add(code)
            else:
                code = s.strip().upper()
                if code:
                    codes.add(code)
    return codes


def _infer_proc_code_columns(df: pd.DataFrame, explicit: Optional[List[str]] = None) -> List[str]:
    if explicit:
        return [c for c in explicit if c in df.columns]
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if lc in ('icdprocedurecode', 'icd10_pcode', 'procedurecodes', 'procedure_codes'):
            candidates.append(c)
            continue
        if ('icd' in lc and 'proc' in lc) or ('procedure' in lc and ('code' in lc or 'cd' in lc)):
            candidates.append(c)
        # common wide formats like ICDPROC1..ICDPROCn, PROC_CODE1..n
        if lc.startswith('icdproc') or lc.startswith('icdprocedure') or lc.startswith('proc_code'):
            candidates.append(c)
    # de-dup while preserving order
    seen = set(); out = []
    for c in candidates:
        if c not in seen:
            out.append(c); seen.add(c)
    return out


def build_enhanced_nei6_label_from_edas(
    df: pd.DataFrame,
    proc_code_cols: Optional[List[str]] = None,
) -> pd.Series:
    """
    Build NEI-6 style label from EDAS CSV in-place columns (no external joins).
    Mirrors logic used in TQIP scripts:
    - packed RBC >= 5 units within 4 hours (BLOOD4HOURS or TRANS_BLOOD_4HOURS)
    - angiography performed or <=24h (using PR_A_ELAPSEDSC_L or PR_ELAPSEDSC_L timestamps)
    - any hemorrhage control operation (HMRRHGCTRLSURGTYPE > 0)
    - cerebral monitor present (TBICEREBRALMONITORHRS > 0 or CerebralMonitorMins > 0)
    - chest tube or central line ICD-10 procedure codes present (from icd10_codes, PR_ICD10_S_L, or ICD 10 Proc code)
    """
    # Packed RBC >= 5 units (prefer BLOOD4HOURS if present)
    rbc_col = 'BLOOD4HOURS' if 'BLOOD4HOURS' in df.columns else ('TRANS_BLOOD_4HOURS' if 'TRANS_BLOOD_4HOURS' in df.columns else None)
    if rbc_col is not None:
        packed_rbc_5units = (pd.to_numeric(df[rbc_col], errors='coerce') >= 5.0)
    else:
        packed_rbc_5units = pd.Series(False, index=df.index)

    # Angiography (flag or within 24 hours using specified timestamp columns)
    angio = pd.Series(False, index=df.index)
    if 'ANGIOGRAPHY' in df.columns:
        angio = (pd.to_numeric(df['ANGIOGRAPHY'], errors='coerce') == 1)
    
    # Check timestamp columns for angiography within 24 hours
    angio_timestamp_col = None
    for col in ['PR_A_ELAPSEDSC_L', 'PR_ELAPSEDSC_L']:
        if col in df.columns:
            angio_timestamp_col = col
            break
    
    if angio_timestamp_col is not None:
        angio_hrs = (pd.to_numeric(df[angio_timestamp_col], errors='coerce') <= 24)
        angio = angio | angio_hrs.fillna(False)
    elif 'ANGIOGRAPHYHRS' in df.columns:  # fallback to old column name
        angio_hrs = (pd.to_numeric(df['ANGIOGRAPHYHRS'], errors='coerce') <= 24)
        angio = angio | angio_hrs.fillna(False)

    # Any hemorrhage control operation
    if 'HMRRHGCTRLSURGTYPE' in df.columns:
        any_operation = (pd.to_numeric(df['HMRRHGCTRLSURGTYPE'], errors='coerce') > 0)
    else:
        any_operation = pd.Series(False, index=df.index)

    # Cerebral monitoring
    if 'TBICEREBRALMONITORHRS' in df.columns:
        brain = (pd.to_numeric(df['TBICEREBRALMONITORHRS'], errors='coerce') > 0)
    elif 'CerebralMonitorMins' in df.columns:
        brain = (pd.to_numeric(df['CerebralMonitorMins'], errors='coerce') > 0)
    else:
        brain = pd.Series(False, index=df.index)

    # Procedure codes: chest tube and central line
    # Use specified procedure code columns: icd10_codes, PR_ICD10_S_L, or ICD 10 Proc code
    chest_tube_codes = {'0W9930Z', '0W9B30Z'}
    central_line_codes = {'05HM33Z', '05HN33Z', '06HM33Z', '06HN33Z', '05H533Z', '05H633Z'}
    
    # Find procedure code column from specified candidates
    proc_cols = []
    if proc_code_cols:
        # Use explicitly provided columns if available
        proc_cols = [c for c in proc_code_cols if c in df.columns]
    else:
        # Try specified column names in order
        for col in ['icd10_codes', 'PR_ICD10_S_L', 'ICD 10 Proc code']:
            if col in df.columns:
                proc_cols = [col]
                break
    
    chest_tube = pd.Series(False, index=df.index)
    central_line = pd.Series(False, index=df.index)
    if proc_cols:
        # Evaluate per-row set inclusion
        code_sets = df[proc_cols].apply(
            lambda row: _collect_proc_codes_from_row(row, proc_cols), axis=1
        )
        chest_tube = code_sets.apply(lambda s: len(s & chest_tube_codes) > 0).values
        central_line = code_sets.apply(lambda s: len(s & central_line_codes) > 0).values

    y = (
        packed_rbc_5units.fillna(False)
        | angio.fillna(False)
        | any_operation.fillna(False)
        | pd.Series(brain).fillna(False)
        | pd.Series(chest_tube).fillna(False)
        | pd.Series(central_line).fillna(False)
    ).astype(int)
    return y


def create_pm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build PM-only style features from canonical columns.
    Mirrors behavior from the TQIP PM feature builder, with EDAS additions.
    """
    X = pd.DataFrame(index=df.index)

    # Raw hospital features
    if 'AGEyears' in df.columns:
        X['AGEyears'] = pd.to_numeric(df['AGEyears'], errors='coerce')
    if 'SEX' in df.columns:
        X['SEX'] = df['SEX']
    for col in ['TOTALGCS', 'SBP', 'PULSERATE', 'TEMPERATURE']:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors='coerce')

    # Arrival time derived features
    if 'ARRIVALTIME' in df.columns:
        hours = pd.to_datetime(df['ARRIVALTIME'], errors='coerce').dt.hour
        X['evening_arrival'] = ((hours >= 18) | (hours <= 6)).astype(int)
    if 'HOSPITALARRIVALHRS' in df.columns:
        hrs = pd.to_numeric(df['HOSPITALARRIVALHRS'], errors='coerce')
        X['HOSPITALARRIVALHRS'] = hrs
        X['transport_time_15min'] = (hrs < 0.25).astype(int)
        X['transport_time_30min'] = (hrs < 0.5).astype(int)

    # Raw prehospital features
    for col in [
        'EMSTOTALGCS', 'EMSSBP', 'EMSPULSERATE', 'EMSRESPIRATORYRATE',
        'EMSPULSEOXIMETRY', 'EMSGCSEYE', 'EMSGCSMOTOR', 'EMSGCSVERBAL'
    ]:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors='coerce')

    # PREHOSPITALCARDIACARREST (optional if ever present)
    if 'PREHOSPITALCARDIACARREST' in df.columns:
        X['PREHOSPITALCARDIACARREST'] = (
            pd.to_numeric(df['PREHOSPITALCARDIACARREST'], errors='coerce')
            .fillna(0).astype(int)
        )

    # Derived prehospital indicators (exclude severe composite; keep moderate/mild)
    if 'EMSPULSERATE' in X.columns and 'EMSSBP' in X.columns:
        si = X['EMSPULSERATE'] / (X['EMSSBP'] + 1e-8)
        X['ems_shock_index'] = si.replace([np.inf, -np.inf], np.nan)
    if 'EMSTOTALGCS' in X.columns:
        gcs = X['EMSTOTALGCS']
        X['ems_gcs_moderate'] = ((gcs >= 9) & (gcs <= 12)).astype(int)
        X['ems_gcs_mild'] = (gcs >= 13).astype(int)
    if 'EMSSBP' in X.columns:
        sbp = X['EMSSBP']
        X['ems_sbp_low'] = (sbp <= 90).astype(int)
        X['ems_sbp_critical'] = (sbp <= 70).astype(int)
        X['ems_sbp_high'] = (sbp >= 180).astype(int)
    if 'EMSPULSERATE' in X.columns:
        pulse = X['EMSPULSERATE']
        X['ems_tachycardia'] = (pulse >= 100).astype(int)
        X['ems_bradycardia'] = (pulse <= 60).astype(int)
    if 'EMSRESPIRATORYRATE' in X.columns:
        rr = X['EMSRESPIRATORYRATE']
        X['ems_tachypnea'] = (rr >= 30).astype(int)
        X['ems_bradypnea'] = (rr <= 10).astype(int)
    if 'EMSPULSEOXIMETRY' in X.columns:
        spo2 = X['EMSPULSEOXIMETRY']
        X['ems_hypoxia'] = (spo2 < 92).astype(int)
        X['ems_severe_hypoxia'] = (spo2 < 88).astype(int)

    # ΔSBP: SBP - EMSSBP >= 20
    if 'SBP' in X.columns and 'EMSSBP' in X.columns:
        delta = X['SBP'] - X['EMSSBP']
        X['sbp_minus_emssbp_ge20'] = (delta >= 20).astype(int)

    # Added categorical features (kept raw here; one-hot later)
    if CANON_INJURY_TYPE in df.columns:
        X[CANON_INJURY_TYPE] = df[CANON_INJURY_TYPE].astype('string')
    if CANON_TRANSFER_IN in df.columns:
        X[CANON_TRANSFER_IN] = df[CANON_TRANSFER_IN].astype('string')
    if CANON_INTUB_TEXT in df.columns:
        X[CANON_INTUB_TEXT] = df[CANON_INTUB_TEXT].astype('string')

    return X


def build_features_and_label(
    df_raw: pd.DataFrame,
    label_source: str,
    label_column: str,
    proc_code_cols: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Remap columns, build features, and extract labels.
    Returns:
      X (design matrix), y (numpy array), ids_df (DataFrame with identifiers)
    """
    df = remap_edas_columns(df_raw)

    # Prefer existing incident key if present, otherwise fallback to row index
    id_col = None
    for cand in ['inc_key', 'INC_KEY', 'Inc_Key', 'IncID', 'RecordID', 'record_id']:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        df = df.copy()
        df['RecordId'] = np.arange(len(df), dtype=int)
        id_col = 'RecordId'

    # Build features
    X_pm = create_pm_features(df)
    # Labels
    if label_source == 'auto-nei6':
        y_series = build_enhanced_nei6_label_from_edas(df, proc_code_cols=proc_code_cols)
        y = y_series.astype(int).values
    else:
        if label_column not in df.columns:
            raise ValueError(
                f"Label column '{label_column}' not found in CSV. "
                f"Available columns: {list(df.columns)[:20]}..."
            )
        y = pd.to_numeric(df[label_column], errors='coerce').fillna(0).astype(int).values
    # IDs/meta
    ids_df = df[[id_col]].rename(columns={id_col: 'Id'})

    return X_pm, y, ids_df


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost on EDAS CSV (PM-only style features).')
    parser.add_argument('--input-csv', type=str, default=None, help='Path to EDAS CSV file')
    parser.add_argument('--label-source', type=str, choices=['auto-nei6', 'column'], default='auto-nei6',
                        help='Use auto-generated NEI-6 label or a provided label column')
    parser.add_argument('--label-column', type=str, default='label', help='Name of binary label column in CSV (if --label-source column)')
    parser.add_argument('--proc-code-columns', type=str, default=None,
                        help='Comma-separated list of procedure code column names (optional). '
                             'If omitted, columns are inferred by name patterns.')
    parser.add_argument('--test-size', type=float, default=0.2, help='Holdout fraction for test set')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    print('🎯 EDAS XGBoost')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = args.input_csv or os.path.join(base_dir, 'edas_dataset.csv')
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"📥 Reading CSV: {input_csv}")
    df_raw = pd.read_csv(input_csv)
    print(f"🧮 Rows: {len(df_raw):,}, Columns: {len(df_raw.columns):,}")

    # Parse proc code columns if provided
    proc_cols = None
    if args.proc_code_columns:
        proc_cols = [c.strip() for c in args.proc_code_columns.split(',') if c.strip()]

    # Build features and labels
    X_pm, y, ids_df = build_features_and_label(
        df_raw,
        label_source=args.label_source,
        label_column=args.label_column,
        proc_code_cols=proc_cols,
    )
    print(f"📦 Feature rows: {len(X_pm):,}, Positive rate: {np.mean(y):.3%}")

    # Prepare design matrix
    X = X_pm.copy()
    # Categorical columns to one-hot encode
    cat_cols = [
        c for c in [
            'SEX', 'EMSGCSEYE', 'EMSGCSMOTOR', 'EMSGCSVERBAL',
            CANON_INJURY_TYPE, CANON_TRANSFER_IN, CANON_INTUB_TEXT
        ] if c in X.columns
    ]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    # Deduplicate if any
    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()]
    # Clean feature names for XGBoost (remove [, ], < characters)
    X.columns = X.columns.str.replace('[', '_', regex=False)
    X.columns = X.columns.str.replace(']', '_', regex=False)
    X.columns = X.columns.str.replace('<', '_', regex=False)
    # Impute numeric with median
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].apply(lambda s: s.fillna(s.median()))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Calculate scale_pos_weight for class imbalance
    pos_count = np.sum(y_train)
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # Model
    clf = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=args.random_state,
        eval_metric='auc',
    )

    t0 = time.time(); print('⏳ Fitting XGBoost ...')
    clf.fit(X_train, y_train)
    fit_sec = time.time() - t0
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    y_pred = (y_proba >= 0.5).astype(int)
    print(f"🏆 XGB AUC={auc:.4f} (fit {fit_sec:.1f}s)")

    # Outputs
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(base_dir, f'xgb_edas_pm_only_sbpDelta_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({'model': clf, 'feature_names': list(X.columns)}, os.path.join(out_dir, 'best_model.joblib'))
    # Save basic meta
    pd.DataFrame({
        'Rows': [len(X)],
        'PositiveRate': [float(np.mean(y))],
        'NumFeatures': [X.shape[1]],
    }).to_csv(os.path.join(out_dir, 'meta.csv'), index=False)
    # Save ids and labels
    pd.DataFrame({'Id': ids_df['Id'], 'label': y}).to_csv(os.path.join(out_dir, 'ids_labels.csv'), index=False)

    # Feature importance
    try:
        feature_names = list(X.columns)
        importances = getattr(clf, 'feature_importances_', None)
        if importances is not None and len(importances) == len(feature_names):
            fi_df = (
                pd.DataFrame({'feature': feature_names, 'importance': importances})
                .sort_values('importance', ascending=False)
            )
            fi_df.to_csv(os.path.join(out_dir, 'feature_importance_xgb.csv'), index=False)
            top_k = 30
            top_df = fi_df.head(top_k).iloc[::-1]
            plt.figure(figsize=(10, max(6, len(top_df) * 0.35)))
            sns.barplot(x='importance', y='feature', data=top_df, color='#2E86AB')
            plt.title('XGBoost Feature Importance (top 30) — EDAS')
            plt.xlabel('Importance (Gain)')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'feature_importance_top30.png'), dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"⚠️ Failed to compute XGB feature importance: {e}")

    # ROC and Confusion Matrix
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f"XGB AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('XGB EDAS — ROC')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'roc.png'), dpi=300, bbox_inches='tight')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Output: {out_dir}")


if __name__ == '__main__':
    main()

