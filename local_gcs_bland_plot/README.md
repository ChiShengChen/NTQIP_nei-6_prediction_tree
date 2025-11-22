# Local Real-time vs Post-hoc GCS Bland-Altman Analysis

A simplified tool for comparing Real-time GCS and Post-hoc GCS measurements using Bland-Altman analysis.

## Features

- Read Real-time GCS and Post-hoc GCS data from a single CSV file
- Generate a Bland-Altman plot
- Output statistics summary CSV file

## Usage

### Basic Usage

```bash
python local_gcs_bland_plot.py --input_csv example_data.csv
```

### Full Parameters

```bash
python local_gcs_bland_plot.py \
    --input_csv example_data.csv \
    --real_time_col TOTALGCS \
    --post_hoc_col TBIHIGHESTTOTALGCS \
    --outdir output \
    --max_points 50000 \
    --delta_threshold 0.0
```

### Parameters

- `--input_csv` (required): Input CSV file path, must contain Real-time GCS and Post-hoc GCS columns
- `--real_time_col` (optional): Real-time GCS column name, default: `TOTALGCS`. Alternatives: `EMSTOTALGCS`, `TOTALGCS`
- `--post_hoc_col` (optional): Post-hoc GCS column name, default: `TBIHIGHESTTOTALGCS`
- `--outdir` (optional): Output directory, default: `local_gcs_bland_plot`
- `--max_points` (optional): Maximum number of points for scatter plot (will downsample if exceeded), default: 50000
- `--delta_threshold` (optional): Difference threshold, only plot points where |Post-hoc GCS - Real-time GCS| >= threshold, default: 0.0 (no filtering)

## Output Files

The script will generate the following files in the output directory:

1. **`bland_altman_plot.png`**: Bland-Altman plot
2. **`statistics_summary.csv`**: Statistics summary containing:
   - Sample size (n)
   - Mean, standard deviation, median, min, max for Real-time GCS and Post-hoc GCS
   - Delta (difference) statistics
   - Correlation coefficient
   - Paired t-test results
   - 95% Limits of Agreement (LoA)

## CSV File Format

The input CSV file must contain the following columns (or use `--real_time_col` and `--post_hoc_col` to specify):

- `TOTALGCS` (or `EMSTOTALGCS`): Real-time GCS score (measured at the scene or on admission)
- `TBIHIGHESTTOTALGCS`: Post-hoc GCS score (highest GCS measured after admission, typically for TBI patients)

GCS scores range from 3 to 15. Other columns will be ignored.

## Example

An example CSV file (`example_data.csv`) is provided in this directory. You can test the script with:

```bash
python local_gcs_bland_plot.py --input_csv example_data.csv
```

## Example with Custom Column Names

If your CSV file uses different column names:

```bash
python local_gcs_bland_plot.py \
    --input_csv your_data.csv \
    --real_time_col EMSTOTALGCS \
    --post_hoc_col TBIHIGHESTTOTALGCS
```

## Example with Filtering

To only plot points where the difference is >= 2 points:

```bash
python local_gcs_bland_plot.py \
    --input_csv example_data.csv \
    --delta_threshold 2.0
```

## Notes

- GCS (Glasgow Coma Scale) scores range from 3 (deep coma) to 15 (fully conscious)
- Real-time GCS is typically measured at the scene (EMS) or on hospital admission
- Post-hoc GCS (TBIHIGHESTTOTALGCS) is the highest GCS score recorded after admission, often used for TBI patients
- The Bland-Altman plot shows the agreement between these two measurement methods

