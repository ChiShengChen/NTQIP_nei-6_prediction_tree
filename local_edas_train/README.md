# EDAS Model Training

This directory contains training scripts for multiple machine learning models on the EDAS dataset.

## Available Scripts

### Individual Model Training Scripts

- `train_edas_rf_full.py`: RandomForest classifier
- `train_edas_xgb_full.py`: XGBoost classifier
- `train_edas_hgb_full.py`: HistGradientBoosting classifier (includes permutation importance)
- `train_edas_catboost_full.py`: CatBoost classifier
- `train_edas_lgbm_full.py`: LightGBM classifier

### Batch Training Script

- `train_all_models.py`: Train all models sequentially with the same parameters

## Usage

### Batch Training (Recommended)

Train all models sequentially with a single command:

```bash
# Train all models with default CSV
python train_all_models.py

# Train all models with custom CSV
python train_all_models.py --input-csv /path/to/your/data.csv

# Train all models with custom parameters
python train_all_models.py --input-csv data.csv --test-size 0.3 --random-state 123

# Skip specific models
python train_all_models.py --skip-models "rf,xgb"
```

**Available skip options:** `rf`, `xgb`, `hgb`, `catboost`, `lgbm`

### Individual Model Training

#### RandomForest

#### Basic Usage

```bash
python train_edas_rf_full.py --input-csv edas_dataset_example.csv
```

#### Using Custom CSV File

```bash
python train_edas_rf_full.py --input-csv /path/to/your/data.csv
```

#### Using Provided Label Column (Instead of Auto-Generated NEI-6 Label)

```bash
python train_edas_rf_full.py --input-csv data.csv --label-source column --label-column your_label_column_name
```

#### XGBoost

```bash
python train_edas_xgb_full.py --input-csv edas_dataset_example.csv
```

#### HistGradientBoosting

```bash
python train_edas_hgb_full.py --input-csv edas_dataset_example.csv
```

**Note:** HGB script includes permutation importance analysis in addition to standard feature importance.

#### CatBoost

```bash
python train_edas_catboost_full.py --input-csv edas_dataset_example.csv
```

#### LightGBM

```bash
python train_edas_lgbm_full.py --input-csv edas_dataset_example.csv
```

### Common Parameters

All scripts support the same command-line arguments:

- `--input-csv`: Path to EDAS CSV file
- `--label-source`: `auto-nei6` (default) or `column`
- `--label-column`: Name of label column (if using `--label-source column`)
- `--proc-code-columns`: Comma-separated procedure code column names
- `--test-size`: Holdout fraction (default: 0.2)
- `--random-state`: Random seed (default: 42)

## CSV File Format Requirements

### Required Columns (for Feature Engineering)

**Basic Patient Information:**
- `Age` or `AGEyears`: Age
- `Gender` or `SEX`: Gender (M/F)
- `EDAS_GCS` or `TOTALGCS`: Total GCS score
- `EDAS_SBP` or `SBP`: Systolic blood pressure
- `EDAS_PULSE` or `PULSERATE`: Pulse rate
- `EDAS_TEMP` or `TEMPERATURE`: Temperature

**Arrival Time:**
- `ARR_TIME` or `ARRIVALTIME`: Arrival time (datetime format)
- `ARR_HOURS` or `HOSPITALARRIVALHRS`: Hours to hospital arrival

**Prehospital (EMS) Data:**
- `PHAS_GCSSC_L` or `EMSTOTALGCS`: EMS total GCS
- `PHAS_SBPS_L` or `EMSSBP`: EMS systolic blood pressure
- `PHAS_PULSES_L` or `EMSPULSERATE`: EMS pulse rate
- `PHAS_URRS_L` or `EMSRESPIRATORYRATE`: EMS respiratory rate
- `PHAS_SAO2S_L` or `EMSPULSEOXIMETRY`: EMS pulse oximetry
- `PHAS_GCS_EOS_L` or `EMSGCSEYE`: EMS GCS eye
- `PHAS_GCS_MRS_L` or `EMSGCSMOTOR`: EMS GCS motor
- `PHAS_GCS_VRS_L` or `EMSGCSVERBAL`: EMS GCS verbal

**Categorical Features (Optional but Recommended):**
- `Injury Type` / `Injury_Type` / `injury_type`: Injury type (Blunt/Penetrating)
- `Transfer In` / `Transfer_In` / `transfer_in`: Transfer status (Yes/No)
- `PHAS_INTUB_YNS_L_AS_TEXT`: Intubation status (Yes/No/unk)

### Columns Required for NEI-6 Label Generation

**Blood Products:**
- `BLOOD4HOURS` or `TRANS_BLOOD_4HOURS`: Number of blood units transfused within 4 hours

**Angiography:**
- `ANGIOGRAPHY`: Angiography flag (0/1)
- `PR_A_ELAPSEDSC_L` or `PR_ELAPSEDSC_L`: Angiography timestamp (hours)

**Surgery:**
- `HMRRHGCTRLSURGTYPE`: Hemorrhage control surgery type (>0 indicates surgery performed)

**Cerebral Monitoring:**
- `TBICEREBRALMONITORHRS` or `CerebralMonitorMins`: Cerebral monitoring time

**Procedure Codes:**
- `icd10_codes` or `PR_ICD10_S_L` or `ICD 10 Proc code`: ICD-10 procedure codes
  - Supports multiple delimiters: semicolon(;), comma(,), pipe(|)
  - Examples: `0W9930Z;05HM33Z` or `0W9930Z,05HM33Z`

### ID Column (Optional)

The script will automatically search for one of the following column names as ID:
- `inc_key`, `INC_KEY`, `Inc_Key`
- `IncID`, `RecordID`, `record_id`

If none exist, it will automatically generate a `RecordId` column.

## Output

After training completes, a timestamped output folder will be created in the script directory, containing:

- `best_model.joblib`: Trained model
- `meta.csv`: Dataset metadata
- `ids_labels.csv`: ID and label mapping
- `feature_importance_*.csv`: Feature importance ranking (model-specific)
- `feature_importance_top30.png`: Visualization of top 30 important features
- `permutation_importance_hgb.csv`: Permutation importance (HGB only)
- `permutation_importance_top30.png`: Permutation importance visualization (HGB only)
- `roc.png`: ROC curve plot
- `confusion_matrix.png`: Confusion matrix

**Output folder naming:**
- RandomForest: `rf_edas_pm_only_sbpDelta_YYYYMMDD_HHMMSS`
- XGBoost: `xgb_edas_pm_only_sbpDelta_YYYYMMDD_HHMMSS`
- HistGradientBoosting: `hgb_edas_pm_only_sbpDelta_YYYYMMDD_HHMMSS`
- CatBoost: `catboost_edas_pm_only_sbpDelta_YYYYMMDD_HHMMSS`
- LightGBM: `lgbm_edas_pm_only_sbpDelta_YYYYMMDD_HHMMSS`

## Notes

1. All numeric columns can contain missing values (NaN), which will be automatically handled by the script
2. Procedure code columns support multiple codes separated by semicolon, comma, or pipe
3. Datetime columns should use standard format (e.g., `2023-01-15 14:30:00`)
4. Default train/test split is 80/20, which can be adjusted via the `--test-size` parameter
