#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a RandomForest NEI-6 model (2015–2019) using PM features only
and WITHOUT EMS severity composites, plus an added indicator feature:
sbp_minus_emssbp_ge20 = 1 if (SBP - EMSSBP) >= 20, else 0.
"""

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split


def load_year_trauma(base_dir: str, year: int) -> pd.DataFrame:
    if year in (2015, 2016):
        path = os.path.join(base_dir, 'TQIP', f'PUF AY {year}', 'CSV', 'PUF_PM.csv')
    else:
        path = os.path.join(base_dir, 'TQIP', f'PUF AY {year}', 'CSV', 'PUF_TRAUMA.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_year_proc(base_dir: str, year: int) -> tuple[pd.DataFrame, str, str]:
    if year in (2015, 2016):
        path = os.path.join(base_dir, 'TQIP', f'PUF AY {year}', 'CSV', 'PUF_PCODE.csv')
        id_col, code_col = 'INC_KEY', 'ICD10_PCODE'
    else:
        path = os.path.join(base_dir, 'TQIP', f'PUF AY {year}', 'CSV', 'PUF_ICDPROCEDURE.csv')
        id_col, code_col = 'Inc_Key', 'ICDPROCEDURECODE'
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path), id_col, code_col


def build_enhanced_nei6_label(df: pd.DataFrame, df_proc: pd.DataFrame, year: int, id_col_proc: str) -> pd.Series:
    rbc_col = 'BLOOD4HOURS' if year >= 2017 else 'TRANS_BLOOD_4HOURS'
    if rbc_col in df.columns:
        packed_rbc_5units = (pd.to_numeric(df[rbc_col], errors='coerce') >= 5.0)
    else:
        packed_rbc_5units = pd.Series(False, index=df.index)

    if 'ANGIOGRAPHY' in df.columns:
        angio = (df['ANGIOGRAPHY'] == 1)
    else:
        angio = pd.Series(False, index=df.index)
    if 'ANGIOGRAPHYHRS' in df.columns:
        angio_hrs = (pd.to_numeric(df['ANGIOGRAPHYHRS'], errors='coerce') <= 24)
        angio = angio | angio_hrs.fillna(False)

    if 'HMRRHGCTRLSURGTYPE' in df.columns:
        any_operation = (pd.to_numeric(df['HMRRHGCTRLSURGTYPE'], errors='coerce') > 0)
    else:
        any_operation = pd.Series(False, index=df.index)

    if 'TBICEREBRALMONITORHRS' in df.columns:
        brain = (pd.to_numeric(df['TBICEREBRALMONITORHRS'], errors='coerce') > 0)
    elif 'CerebralMonitorMins' in df.columns:
        brain = (pd.to_numeric(df['CerebralMonitorMins'], errors='coerce') > 0)
    else:
        brain = pd.Series(False, index=df.index)

    chest_tube_codes = {'0W9930Z', '0W9B30Z'}
    central_line_codes = {'05HM33Z', '05HN33Z', '06HM33Z', '06HN33Z', '05H533Z', '05H633Z'}
    chest_tube = pd.Series(False, index=df.index)
    central_line = pd.Series(False, index=df.index)
    if df_proc is not None and id_col_proc in df_proc.columns:
        proc = df_proc.copy()
        proc[id_col_proc] = proc[id_col_proc].astype(str)
        code_col = 'ICD10_PCODE' if year in (2015, 2016) else 'ICDPROCEDURECODE'
        proc[code_col] = proc[code_col].astype(str).str.strip().str.upper()
        by_key = proc.groupby(id_col_proc)[code_col].apply(set)
        inc_key_col = 'inc_key' if 'inc_key' in df.columns else ('INC_KEY' if 'INC_KEY' in df.columns else 'Inc_Key')
        keys = df[inc_key_col].astype(str)
        st = by_key.reindex(keys).apply(lambda s: s if isinstance(s, set) else set())
        chest_tube = st.apply(lambda s: len(s & chest_tube_codes) > 0).values
        central_line = st.apply(lambda s: len(s & central_line_codes) > 0).values
    y = (packed_rbc_5units.fillna(False) | angio.fillna(False) | any_operation.fillna(False) | pd.Series(brain).fillna(False) | pd.Series(chest_tube).fillna(False) | pd.Series(central_line).fillna(False)).astype(int)
    return y


def create_pm_features(df: pd.DataFrame) -> pd.DataFrame:
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
    for col in ['EMSTOTALGCS', 'EMSSBP', 'EMSPULSERATE', 'EMSRESPIRATORYRATE', 'EMSPULSEOXIMETRY', 'EMSGCSEYE', 'EMSGCSMOTOR', 'EMSGCSVERBAL']:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors='coerce')
    if 'PREHOSPITALCARDIACARREST' in df.columns:
        X['PREHOSPITALCARDIACARREST'] = pd.to_numeric(df['PREHOSPITALCARDIACARREST'], errors='coerce').fillna(0).astype(int)
    # Derived prehospital indicators (exclude ems_gcs_severe; keep moderate/mild)
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
    # Added delta feature: SBP - EMSSBP >= 20
    if 'SBP' in X.columns and 'EMSSBP' in X.columns:
        delta = X['SBP'] - X['EMSSBP']
        X['sbp_minus_emssbp_ge20'] = (delta >= 20).astype(int)
    return X


def main():
    print('🎯 RandomForest Enhanced NEI-6 (2015–2019) — PM-only + ΔSBP≥20, NO EMS severity composites')
    base_dir = os.path.dirname(os.path.abspath(__file__))

    frames = []
    metas = []
    for year in [2015, 2016, 2017, 2018, 2019]:
        print(f"➡️  Year {year} ...")
        df = load_year_trauma(base_dir, year)
        df_proc, id_col, code_col = load_year_proc(base_dir, year)
        if 'inc_key' in df.columns:
            pass
        elif 'INC_KEY' in df.columns:
            df = df.rename(columns={'INC_KEY': 'inc_key'})
        elif 'Inc_Key' in df.columns:
            df = df.rename(columns={'Inc_Key': 'inc_key'})
        else:
            raise RuntimeError('Cannot find inc_key column in trauma/PM file')
        df['inc_key'] = df['inc_key'].astype(str)
        y = build_enhanced_nei6_label(df, df_proc, year, id_col)
        metas.append({'Year': year, 'PositiveRate': float(np.mean(y)), 'Count': int(len(y))})
        X_pm = create_pm_features(df)
        feat = df[['inc_key']].merge(X_pm.assign(inc_key=df['inc_key']), on='inc_key', how='left')
        feat = feat.assign(label=y.values, Year=year)
        frames.append(feat)

    agg = pd.concat(frames, axis=0, ignore_index=True)
    print(f"📦 Aggregate rows: {len(agg):,}, Positive rate: {agg['label'].mean():.3%}")
    meta_df = pd.DataFrame(metas)

    y = agg['label'].astype(int).values
    feature_cols = [c for c in agg.columns if c not in ('inc_key', 'label', 'Year')]
    X = agg[feature_cols].copy()
    cat_cols = [c for c in ['SEX', 'EMSGCSEYE', 'EMSGCSMOTOR', 'EMSGCSVERBAL'] if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()]
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].apply(lambda s: s.fillna(s.median()))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=25,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )

    t0 = time.time(); print('⏳ Fitting RandomForest ...')
    clf.fit(X_train, y_train)
    fit_sec = time.time() - t0
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    y_pred = (y_proba >= 0.5).astype(int)
    print(f"🏆 RF AUC={auc:.4f} (fit {fit_sec:.1f}s)")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(base_dir, f'rf_enhanced_pm_only_sbpDelta_noSeverity_2015to2019_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({'model': clf, 'feature_names': list(X.columns)}, os.path.join(out_dir, 'best_model.joblib'))
    meta_df.to_csv(os.path.join(out_dir, 'year_meta.csv'), index=False)
    pd.DataFrame({'Inc_Key': agg['inc_key'], 'Year': agg['Year'], 'label': agg['label']}).to_csv(os.path.join(out_dir, 'ids_labels.csv'), index=False)

    # Feature importance
    try:
        feature_names = list(X.columns)
        importances = getattr(clf, 'feature_importances_', None)
        if importances is not None and len(importances) == len(feature_names):
            fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
            fi_df.to_csv(os.path.join(out_dir, 'feature_importance_rf.csv'), index=False)
            top_k = 30
            top_df = fi_df.head(top_k).iloc[::-1]
            plt.figure(figsize=(10, max(6, len(top_df) * 0.35)))
            sns.barplot(x='importance', y='feature', data=top_df, color='#8172B2')
            plt.title('RandomForest Feature Importance (top 30) — PM-only + ΔSBP≥20')
            plt.xlabel('Importance (Gini)')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'feature_importance_top30.png'), dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"⚠️ Failed to compute RF feature importance: {e}")

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f"RF AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('RF PM-only + ΔSBP≥20 (2015–2019) — no severity composites')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'roc.png'), dpi=300, bbox_inches='tight')
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues'); plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Output: {out_dir}")


if __name__ == '__main__':
    main()


