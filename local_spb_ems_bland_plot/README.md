# Local SBP/EMSSBP Bland-Altman Analysis

A simplified tool for SBP/EMSSBP Bland-Altman analysis that reads data from a single CSV file.

## Features

- Read SBP and EMSSBP data from a single CSV file
- Generate a Bland-Altman plot
- Output statistics summary CSV file

## Usage

### Basic Usage

```bash
python local_sbp_ems_bland_plot.py --input_csv example_data.csv
```

### Full Parameters

```bash
python local_sbp_ems_bland_plot.py \
    --input_csv example_data.csv \
    --sbp_col SBP \
    --emssbp_col EMSSBP \
    --outdir output \
    --max_points 50000 \
    --delta_threshold 0.0
```

### Parameters

- `--input_csv` (required): Input CSV file path, must contain SBP and EMSSBP columns
- `--sbp_col` (optional): SBP column name, default: `SBP`
- `--emssbp_col` (optional): EMSSBP column name, default: `EMSSBP`
- `--outdir` (optional): Output directory, default: `local_spb_ems_bland_plot`
- `--max_points` (optional): Maximum number of points for scatter plot (will downsample if exceeded), default: 50000
- `--delta_threshold` (optional): Difference threshold, only plot points where |SBP - EMSSBP| >= threshold, default: 0.0 (no filtering)

## Output Files

The script will generate the following files in the output directory:

1. **`bland_altman_plot.png`**: Bland-Altman plot
2. **`statistics_summary.csv`**: Statistics summary containing:
   - Sample size (n)
   - Mean, standard deviation, median, min, max for SBP and EMSSBP
   - Delta (difference) statistics
   - Correlation coefficient
   - Paired t-test results
   - 95% Limits of Agreement (LoA)

## CSV File Format

The input CSV file must contain the following columns (or use `--sbp_col` and `--emssbp_col` to specify):

- `SBP`: Systolic blood pressure (hospital measurement)
- `EMSSBP`: Systolic blood pressure (EMS measurement)

Other columns will be ignored.

## Example

An example CSV file (`example_data.csv`) is provided in this directory. You can test the script with:

```bash
python local_sbp_ems_bland_plot.py --input_csv example_data.csv
```

## Example with Custom Column Names

If your CSV file uses different column names:

```bash
python local_sbp_ems_bland_plot.py \
    --input_csv your_data.csv \
    --sbp_col HospitalSBP \
    --emssbp_col EMSSBP
```

## Example with Filtering

To only plot points where the difference is >= 20 mmHg:

```bash
python local_sbp_ems_bland_plot.py \
    --input_csv example_data.csv \
    --delta_threshold 20.0
```

