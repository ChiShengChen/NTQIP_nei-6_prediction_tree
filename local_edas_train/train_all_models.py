#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train all models on the EDAS dataset sequentially.

This script runs all available model training scripts in order:
1. RandomForest
2. XGBoost
3. HistGradientBoosting
4. CatBoost
5. LightGBM

All models will use the same input CSV and parameters.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_training_script(script_name: str, input_csv: str, label_source: str = 'auto-nei6', 
                       label_column: str = 'label', proc_code_columns: str = None,
                       test_size: float = 0.2, random_state: int = 42) -> bool:
    """
    Run a training script and return True if successful, False otherwise.
    """
    base_dir = Path(__file__).parent
    script_path = base_dir / script_name
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_name}")
        return False
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting: {script_name}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        '--input-csv', input_csv,
        '--label-source', label_source,
        '--label-column', label_column,
        '--test-size', str(test_size),
        '--random-state', str(random_state),
    ]
    
    if proc_code_columns:
        cmd.extend(['--proc-code-columns', proc_code_columns])
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(base_dir),
            check=True,
            capture_output=False,  # Show output in real-time
        )
        print(f"\n✅ Completed: {script_name}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed: {script_name} (exit code: {e.returncode})\n")
        return False
    except Exception as e:
        print(f"\n❌ Error running {script_name}: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Train all EDAS models sequentially',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models with default CSV
  python train_all_models.py
  
  # Train all models with custom CSV
  python train_all_models.py --input-csv /path/to/data.csv
  
  # Train all models with custom parameters
  python train_all_models.py --input-csv data.csv --test-size 0.3 --random-state 123
        """
    )
    parser.add_argument('--input-csv', type=str, default=None,
                        help='Path to EDAS CSV file (default: edas_dataset_example.csv)')
    parser.add_argument('--label-source', type=str, choices=['auto-nei6', 'column'], 
                        default='auto-nei6',
                        help='Use auto-generated NEI-6 label or a provided label column')
    parser.add_argument('--label-column', type=str, default='label',
                        help='Name of binary label column in CSV (if --label-source column)')
    parser.add_argument('--proc-code-columns', type=str, default=None,
                        help='Comma-separated list of procedure code column names (optional)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Holdout fraction for test set (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--skip-models', type=str, default=None,
                        help='Comma-separated list of models to skip (e.g., "rf,xgb")')
    args = parser.parse_args()
    
    # Determine input CSV
    base_dir = Path(__file__).parent
    if args.input_csv:
        input_csv = args.input_csv
    else:
        input_csv = str(base_dir / 'edas_dataset_example.csv')
    
    if not os.path.exists(input_csv):
        print(f"❌ Input CSV not found: {input_csv}")
        sys.exit(1)
    
    # Define all models to train
    models = [
        ('train_edas_rf_full.py', 'RandomForest'),
        ('train_edas_xgb_full.py', 'XGBoost'),
        ('train_edas_hgb_full.py', 'HistGradientBoosting'),
        ('train_edas_catboost_full.py', 'CatBoost'),
        ('train_edas_lgbm_full.py', 'LightGBM'),
    ]
    
    # Parse skip models
    skip_models = set()
    if args.skip_models:
        skip_models = {m.strip().lower() for m in args.skip_models.split(',')}
    
    # Filter out skipped models
    model_mapping = {
        'rf': 'train_edas_rf_full.py',
        'randomforest': 'train_edas_rf_full.py',
        'xgb': 'train_edas_xgb_full.py',
        'xgboost': 'train_edas_xgb_full.py',
        'hgb': 'train_edas_hgb_full.py',
        'histgradientboosting': 'train_edas_hgb_full.py',
        'catboost': 'train_edas_catboost_full.py',
        'lgbm': 'train_edas_lgbm_full.py',
        'lightgbm': 'train_edas_lgbm_full.py',
    }
    
    models_to_run = []
    for script_name, model_name in models:
        script_base = script_name.replace('train_edas_', '').replace('_full.py', '')
        if script_base in skip_models or model_name.lower() in skip_models:
            print(f"⏭️  Skipping: {model_name} ({script_name})")
            continue
        models_to_run.append((script_name, model_name))
    
    if not models_to_run:
        print("❌ No models to train (all skipped)")
        sys.exit(1)
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 EDAS Model Training - Batch Execution")
    print("="*80)
    print(f"📥 Input CSV: {input_csv}")
    print(f"📊 Label source: {args.label_source}")
    if args.label_source == 'column':
        print(f"📋 Label column: {args.label_column}")
    print(f"🔀 Test size: {args.test_size}")
    print(f"🎲 Random state: {args.random_state}")
    print(f"📦 Models to train: {len(models_to_run)}")
    for script_name, model_name in models_to_run:
        print(f"   - {model_name}")
    print("="*80 + "\n")
    
    # Run all models
    results = {}
    for script_name, model_name in models_to_run:
        success = run_training_script(
            script_name,
            input_csv,
            label_source=args.label_source,
            label_column=args.label_column,
            proc_code_columns=args.proc_code_columns,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        results[model_name] = success
    
    # Print summary
    print("\n" + "="*80)
    print("📊 Training Summary")
    print("="*80)
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    if successful:
        print(f"\n✅ Successful ({len(successful)}):")
        for name in successful:
            print(f"   - {name}")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for name in failed:
            print(f"   - {name}")
    
    print("\n" + "="*80)
    
    # Exit with error code if any failed
    if failed:
        sys.exit(1)
    else:
        print("🎉 All models trained successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()

