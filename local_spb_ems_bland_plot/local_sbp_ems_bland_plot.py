#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local SBP/EMSSBP Bland-Altman Analysis
Read data from a single CSV file and generate Bland-Altman plot and statistics summary
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set font for potential Chinese characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def read_csv_with_encodings(path: Path) -> pd.DataFrame:
    """Try multiple encodings to read CSV file"""
    encodings = ['utf-8', 'latin-1']
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
            continue
    raise last_err  # type: ignore[misc]


def plot_bland_altman(sbp: np.ndarray, emssbp: np.ndarray, out_path: Path, 
                       title_suffix: str = "", max_points: int = 50000,
                       delta_threshold: float = 0.0):
    """
    Plot Bland-Altman plot
    
    Bland-Altman plot is used to assess agreement between two measurement methods:
    - X-axis: Mean of the two measurements
    - Y-axis: Difference between the two measurements
    
    Parameters:
        delta_threshold: Difference threshold, only keep points where |SBP - EMSSBP| >= delta_threshold
                         Default: 0.0 mmHg (no filtering)
    """
    # Clean data: only keep points where both SBP and EMSSBP are valid
    mask = ~(np.isnan(sbp) | np.isnan(emssbp))
    sbp_clean = sbp[mask]
    emssbp_clean = emssbp[mask]
    original_total = len(sbp_clean)
    
    if len(sbp_clean) == 0:
        print(f"  ⚠️  No valid data for Bland-Altman plot")
        return
    
    # Downsample to improve plotting speed
    downsampled = False
    if len(sbp_clean) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(sbp_clean), size=max_points, replace=False)
        sbp_clean = sbp_clean[idx]
        emssbp_clean = emssbp_clean[idx]
        downsampled = True
    
    # Calculate mean and difference
    mean_vals = (sbp_clean + emssbp_clean) / 2.0
    diff_vals = sbp_clean - emssbp_clean
    
    # Filter data based on delta_threshold
    filtered = False
    if delta_threshold > 0:
        filter_mask = np.abs(diff_vals) >= delta_threshold
        before_filter_count = len(diff_vals)
        mean_vals = mean_vals[filter_mask]
        diff_vals = diff_vals[filter_mask]
        filtered_count = len(diff_vals)
        if filtered_count == 0:
            print(f"  ⚠️  No data points after filtering (threshold={delta_threshold}), skipping plot")
            return
        print(f"  📊 Filtered: {before_filter_count:,} → {filtered_count:,} points (|Delta| >= {delta_threshold})")
        filtered = True
    
    # Calculate statistics (based on filtered data)
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals, ddof=1)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    
    # Scatter plot
    ax.scatter(mean_vals, diff_vals, s=8, alpha=0.4, edgecolors='none', color='#2E86AB')
    
    # Mean difference line
    ax.axhline(y=mean_diff, color='red', linestyle='--', linewidth=2, 
               label=f'Mean difference: {mean_diff:.2f} mmHg')
    
    # 95% Limits of Agreement (LoA)
    ax.axhline(y=upper_limit, color='darkred', linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'+1.96 SD: {upper_limit:.2f} mmHg')
    ax.axhline(y=lower_limit, color='darkred', linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'-1.96 SD: {lower_limit:.2f} mmHg')
    
    # Zero line (ideal case)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Mean of SBP and EMSSBP (mmHg)', fontsize=12)
    ax.set_ylabel('Difference (SBP - EMSSBP) (mmHg)', fontsize=12)
    ax.set_title(f'Bland-Altman Plot: SBP vs EMSSBP{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='best', fontsize=10)
    
    # Add statistics text box
    textstr = f'n = {len(diff_vals):,}'
    info_parts = []
    if downsampled:
        info_parts.append(f'downsampled from {original_total:,}')
    if filtered:
        info_parts.append(f'filtered: |Δ| ≥ {delta_threshold}')
    if info_parts:
        textstr += f'\n({", ".join(info_parts)})'
    textstr += f'\nMean difference: {mean_diff:.2f} mmHg\n'
    textstr += f'SD of difference: {std_diff:.2f} mmHg\n'
    textstr += f'95% LoA: [{lower_limit:.2f}, {upper_limit:.2f}] mmHg'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # Set axis ranges to make plot proportions more reasonable
    x_min, x_max = np.nanmin(mean_vals), np.nanmax(mean_vals)
    x_range = x_max - x_min
    ax.set_xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)
    y_range = max(abs(upper_limit), abs(lower_limit)) * 1.2
    ax.set_ylim(-y_range, y_range)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Bland-Altman plot saved: {out_path}")


def compute_statistics(sbp: np.ndarray, emssbp: np.ndarray) -> dict:
    """Calculate statistics"""
    mask = ~(np.isnan(sbp) | np.isnan(emssbp))
    sbp_clean = sbp[mask]
    emssbp_clean = emssbp[mask]
    
    if len(sbp_clean) == 0:
        return {}
    
    delta = sbp_clean - emssbp_clean
    
    stats_dict = {
        'n': len(sbp_clean),
        'sbp_mean': np.mean(sbp_clean),
        'sbp_std': np.std(sbp_clean),
        'sbp_median': np.median(sbp_clean),
        'sbp_min': np.min(sbp_clean),
        'sbp_max': np.max(sbp_clean),
        'emssbp_mean': np.mean(emssbp_clean),
        'emssbp_std': np.std(emssbp_clean),
        'emssbp_median': np.median(emssbp_clean),
        'emssbp_min': np.min(emssbp_clean),
        'emssbp_max': np.max(emssbp_clean),
        'delta_mean': np.mean(delta),
        'delta_std': np.std(delta),
        'delta_median': np.median(delta),
        'delta_min': np.min(delta),
        'delta_max': np.max(delta),
        'correlation': np.corrcoef(sbp_clean, emssbp_clean)[0, 1],
    }
    
    # Paired t-test
    try:
        t_stat, p_value = stats.ttest_rel(sbp_clean, emssbp_clean)
        stats_dict['paired_t_stat'] = t_stat
        stats_dict['paired_t_pvalue'] = p_value
    except:
        pass
    
    # 95% Limits of Agreement
    stats_dict['upper_loa'] = stats_dict['delta_mean'] + 1.96 * stats_dict['delta_std']
    stats_dict['lower_loa'] = stats_dict['delta_mean'] - 1.96 * stats_dict['delta_std']
    
    return stats_dict


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Local SBP/EMSSBP Bland-Altman Analysis: Read data from a single CSV file')
    parser.add_argument('--input_csv', required=True,
                       help='Input CSV file path (must contain SBP and EMSSBP columns)')
    parser.add_argument('--sbp_col', default='SBP',
                       help='SBP column name (default: SBP)')
    parser.add_argument('--emssbp_col', default='EMSSBP',
                       help='EMSSBP column name (default: EMSSBP)')
    parser.add_argument('--outdir', default='local_spb_ems_bland_plot',
                       help='Output directory (default: local_spb_ems_bland_plot)')
    parser.add_argument('--max_points', type=int, default=50000,
                       help='Maximum number of points for scatter plot (will downsample if exceeded, default: 50000)')
    parser.add_argument('--delta_threshold', type=float, default=0.0,
                       help='Difference threshold: only plot points where |SBP - EMSSBP| >= threshold (default: 0.0, no filtering)')
    args = parser.parse_args()
    
    # Output directory
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("📊 Local SBP/EMSSBP Bland-Altman Analysis")
    print("="*80)
    print(f"📁 Input file: {args.input_csv}")
    print(f"📁 Output directory: {out_dir}")
    print(f"📋 SBP column: {args.sbp_col}")
    print(f"📋 EMSSBP column: {args.emssbp_col}")
    print()
    
    # Read CSV file
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"❌ Input file does not exist: {input_path}")
        return 1
    
    try:
        df = read_csv_with_encodings(input_path)
        print(f"✅ Successfully read CSV file: {len(df):,} rows")
    except Exception as e:
        print(f"❌ Failed to read CSV file: {e}")
        return 1
    
    # Check required columns
    if args.sbp_col not in df.columns:
        print(f"❌ SBP column not found: {args.sbp_col}")
        print(f"   Available columns: {', '.join(df.columns[:10])}...")
        return 1
    
    if args.emssbp_col not in df.columns:
        print(f"❌ EMSSBP column not found: {args.emssbp_col}")
        print(f"   Available columns: {', '.join(df.columns[:10])}...")
        return 1
    
    # Extract data
    sbp = pd.to_numeric(df[args.sbp_col], errors='coerce').values
    emssbp = pd.to_numeric(df[args.emssbp_col], errors='coerce').values
    
    # Check valid data
    mask = ~(np.isnan(sbp) | np.isnan(emssbp))
    valid_count = mask.sum()
    total_count = len(df)
    
    print(f"📊 Data statistics:")
    print(f"   Total rows: {total_count:,}")
    print(f"   Valid data (both SBP and EMSSBP have values): {valid_count:,} ({valid_count/total_count*100:.1f}%)")
    
    if valid_count == 0:
        print(f"❌ No valid data, cannot perform analysis")
        return 1
    
    # Calculate statistics
    print(f"\n📈 Calculating statistics...")
    stats_dict = compute_statistics(sbp, emssbp)
    
    # Save statistics summary
    stats_df = pd.DataFrame([stats_dict])
    stats_path = out_dir / 'statistics_summary.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"  ✅ Statistics summary saved: {stats_path}")
    
    # Print statistics summary
    print("\n📊 Statistics summary:")
    print(stats_df.to_string(index=False))
    
    # Generate Bland-Altman plot
    print(f"\n📊 Generating Bland-Altman plot...")
    plot_bland_altman(sbp, emssbp,
                     out_dir / 'bland_altman_plot.png',
                     title_suffix='',
                     max_points=args.max_points,
                     delta_threshold=args.delta_threshold)
    
    print("\n" + "="*80)
    print("✅ Analysis completed!")
    print(f"📁 All results saved in: {out_dir}")
    print(f"   - Bland-Altman plot: {out_dir / 'bland_altman_plot.png'}")
    print(f"   - Statistics summary: {out_dir / 'statistics_summary.csv'}")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

