#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Real-time vs Post-hoc GCS Bland-Altman Analysis
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


def plot_bland_altman(real_time_gcs: np.ndarray, post_hoc_gcs: np.ndarray, out_path: Path, 
                       title_suffix: str = "", max_points: int = 50000,
                       delta_threshold: float = 0.0):
    """
    Plot Bland-Altman plot
    
    Bland-Altman plot is used to assess agreement between two measurement methods:
    - X-axis: Mean of the two measurements
    - Y-axis: Difference between the two measurements
    
    Parameters:
        delta_threshold: Difference threshold, only keep points where |Post-hoc GCS - Real-time GCS| >= delta_threshold
                         Default: 0.0 (no filtering)
    """
    # Clean data: only keep points where both Real-time GCS and Post-hoc GCS are valid
    mask = ~(np.isnan(real_time_gcs) | np.isnan(post_hoc_gcs))
    rt_gcs_clean = real_time_gcs[mask]
    ph_gcs_clean = post_hoc_gcs[mask]
    original_total = len(rt_gcs_clean)
    
    if len(rt_gcs_clean) == 0:
        print(f"  ⚠️  No valid data for Bland-Altman plot")
        return
    
    # Downsample to improve plotting speed
    downsampled = False
    if len(rt_gcs_clean) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(rt_gcs_clean), size=max_points, replace=False)
        rt_gcs_clean = rt_gcs_clean[idx]
        ph_gcs_clean = ph_gcs_clean[idx]
        downsampled = True
    
    # Calculate mean and difference
    mean_vals = (rt_gcs_clean + ph_gcs_clean) / 2.0
    diff_vals = ph_gcs_clean - rt_gcs_clean
    
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
               label=f'Mean difference: {mean_diff:.2f}')
    
    # 95% Limits of Agreement (LoA)
    ax.axhline(y=upper_limit, color='darkred', linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'+1.96 SD: {upper_limit:.2f}')
    ax.axhline(y=lower_limit, color='darkred', linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'-1.96 SD: {lower_limit:.2f}')
    
    # Zero line (ideal case)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Mean of Real-time and Post-hoc GCS', fontsize=12)
    ax.set_ylabel('Difference (Post-hoc GCS - Real-time GCS)', fontsize=12)
    ax.set_title(f'Bland-Altman Plot: Real-time vs Post-hoc GCS{title_suffix}', fontsize=14, fontweight='bold')
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
    textstr += f'\nMean difference: {mean_diff:.2f}\n'
    textstr += f'SD of difference: {std_diff:.2f}\n'
    textstr += f'95% LoA: [{lower_limit:.2f}, {upper_limit:.2f}]'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # Set axis ranges to make plot proportions more reasonable
    x_min, x_max = np.nanmin(mean_vals), np.nanmax(mean_vals)
    x_range = x_max - x_min
    ax.set_xlim(max(2.5, x_min - x_range * 0.05), min(15.5, x_max + x_range * 0.05))
    y_range = max(abs(upper_limit), abs(lower_limit)) * 1.2
    ax.set_ylim(-y_range, y_range)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Bland-Altman plot saved: {out_path}")


def compute_statistics(real_time_gcs: np.ndarray, post_hoc_gcs: np.ndarray) -> dict:
    """Calculate statistics"""
    mask = ~(np.isnan(real_time_gcs) | np.isnan(post_hoc_gcs))
    rt_gcs_clean = real_time_gcs[mask]
    ph_gcs_clean = post_hoc_gcs[mask]
    
    if len(rt_gcs_clean) == 0:
        return {}
    
    delta = ph_gcs_clean - rt_gcs_clean
    
    stats_dict = {
        'n': len(rt_gcs_clean),
        'real_time_gcs_mean': np.mean(rt_gcs_clean),
        'real_time_gcs_std': np.std(rt_gcs_clean),
        'real_time_gcs_median': np.median(rt_gcs_clean),
        'real_time_gcs_min': np.min(rt_gcs_clean),
        'real_time_gcs_max': np.max(rt_gcs_clean),
        'post_hoc_gcs_mean': np.mean(ph_gcs_clean),
        'post_hoc_gcs_std': np.std(ph_gcs_clean),
        'post_hoc_gcs_median': np.median(ph_gcs_clean),
        'post_hoc_gcs_min': np.min(ph_gcs_clean),
        'post_hoc_gcs_max': np.max(ph_gcs_clean),
        'delta_mean': np.mean(delta),
        'delta_std': np.std(delta),
        'delta_median': np.median(delta),
        'delta_min': np.min(delta),
        'delta_max': np.max(delta),
        'correlation': np.corrcoef(rt_gcs_clean, ph_gcs_clean)[0, 1],
    }
    
    # Paired t-test
    try:
        t_stat, p_value = stats.ttest_rel(ph_gcs_clean, rt_gcs_clean)
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
        description='Local Real-time vs Post-hoc GCS Bland-Altman Analysis: Read data from a single CSV file')
    parser.add_argument('--input_csv', required=True,
                       help='Input CSV file path (must contain Real-time GCS and Post-hoc GCS columns)')
    parser.add_argument('--real_time_col', default='TOTALGCS',
                       help='Real-time GCS column name (default: TOTALGCS). Alternatives: EMSTOTALGCS, TOTALGCS')
    parser.add_argument('--post_hoc_col', default='TBIHIGHESTTOTALGCS',
                       help='Post-hoc GCS column name (default: TBIHIGHESTTOTALGCS)')
    parser.add_argument('--outdir', default='local_gcs_bland_plot',
                       help='Output directory (default: local_gcs_bland_plot)')
    parser.add_argument('--max_points', type=int, default=50000,
                       help='Maximum number of points for scatter plot (will downsample if exceeded, default: 50000)')
    parser.add_argument('--delta_threshold', type=float, default=0.0,
                       help='Difference threshold: only plot points where |Post-hoc GCS - Real-time GCS| >= threshold (default: 0.0, no filtering)')
    args = parser.parse_args()
    
    # Output directory
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("📊 Local Real-time vs Post-hoc GCS Bland-Altman Analysis")
    print("="*80)
    print(f"📁 Input file: {args.input_csv}")
    print(f"📁 Output directory: {out_dir}")
    print(f"📋 Real-time GCS column: {args.real_time_col}")
    print(f"📋 Post-hoc GCS column: {args.post_hoc_col}")
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
    if args.real_time_col not in df.columns:
        print(f"❌ Real-time GCS column not found: {args.real_time_col}")
        print(f"   Available columns: {', '.join(df.columns[:10])}...")
        return 1
    
    if args.post_hoc_col not in df.columns:
        print(f"❌ Post-hoc GCS column not found: {args.post_hoc_col}")
        print(f"   Available columns: {', '.join(df.columns[:10])}...")
        return 1
    
    # Extract data
    real_time_gcs = pd.to_numeric(df[args.real_time_col], errors='coerce').values
    post_hoc_gcs = pd.to_numeric(df[args.post_hoc_col], errors='coerce').values
    
    # Check valid data
    mask = ~(np.isnan(real_time_gcs) | np.isnan(post_hoc_gcs))
    valid_count = mask.sum()
    total_count = len(df)
    
    print(f"📊 Data statistics:")
    print(f"   Total rows: {total_count:,}")
    print(f"   Valid data (both Real-time GCS and Post-hoc GCS have values): {valid_count:,} ({valid_count/total_count*100:.1f}%)")
    
    if valid_count == 0:
        print(f"❌ No valid data, cannot perform analysis")
        return 1
    
    # Calculate statistics
    print(f"\n📈 Calculating statistics...")
    stats_dict = compute_statistics(real_time_gcs, post_hoc_gcs)
    
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
    plot_bland_altman(real_time_gcs, post_hoc_gcs,
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

