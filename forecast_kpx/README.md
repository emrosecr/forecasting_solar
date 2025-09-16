# KPX Solar Energy Forecasting

A comprehensive machine learning pipeline for forecasting daily solar energy generation in South Korea (KPX) using ERA5 meteorological data.

## Overview

This project implements a complete solar energy forecasting system with the following key features:

- **Two forecasting tracks**: Anomaly-based and total value predictions
- **Correlation analysis**: Spatial correlation maps between energy anomalies and irradiance
- **Multiple model types**: Baseline statistical models and Random Forest with gridded features
- **No-leakage validation**: Strict temporal separation with train/validation/test splits
- **Google Colab compatible**: Designed for cloud computing environments

## Project Structure

```
forecast_kpx/
├── config.yaml                 # Configuration file
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── io_load.py              # Data loading and preprocessing
│   ├── anomalies.py            # Anomaly calculation functions
│   ├── corr_maps.py            # Correlation mapping
│   ├── features.py             # Feature engineering
│   ├── models_baseline.py      # Baseline statistical models
│   ├── models_rf_grid.py       # Random Forest models
│   ├── eval.py                 # Evaluation metrics
│   └── plotting.py             # Visualization functions
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01_correlations.ipynb   # Correlation analysis
│   ├── 02_train_baselines.py   # Baseline model training
│   └── 03_train_rf_grid.py     # Random Forest training
└── outputs/                    # Generated outputs
    ├── correlation_maps/       # Correlation analysis results
    ├── features/              # Feature engineering outputs
    ├── models/                # Trained models
    ├── metrics/               # Evaluation metrics
    └── plots/                 # Visualization plots
```

## Installation

### For Google Colab

1. Upload the project files to your Google Drive or GitHub
2. In Colab, install dependencies:
```python
!pip install -r requirements.txt
```

### Local Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Requirements

### KPX Solar Generation Data
You need three CSV files:

1. **Original Data**: `solarenergy2.csv`
   - **Format**: CSV
   - **Columns**: `date`, `gen_mwh` (or similar value column)
   - **Content**: Raw daily solar generation in MWh

2. **Base Anomaly**: `solarenergy2_daily_anomaly_base_only.csv`
   - **Format**: CSV  
   - **Columns**: `date`, anomaly values
   - **Content**: DOY-based anomalies (no detrending)

3. **Detrended Anomaly**: `solarenergy2_daily_anomaly_detrended.csv`
   - **Format**: CSV
   - **Columns**: `date`, anomaly values  
   - **Content**: Detrended DOY-based anomalies

- **Period**: 2017-2021 (minimum)
- **Location**: Place all files in `data/` directory

### ERA5 Meteorological Data
- **Format**: NetCDF or Zarr
- **Domain**: Extended (lat: -10° to 60°, lon: 60°E to 250°E)
- **Variables**: `ssrd` (surface solar radiation), `tcc` (total cloud cover), `tcwv` (total column water vapor), `u10`, `v10`, `t2m`
- **Resolution**: Daily aggregated from hourly data
- **Location**: Configure path in `config.yaml`

## Configuration

Edit `config.yaml` to specify:

```yaml
data:
  kpx_original: "data/solarenergy2.csv"
  kpx_anomaly_base: "data/solarenergy2_daily_anomaly_base_only.csv"
  kpx_anomaly_detrended: "data/solarenergy2_daily_anomaly_detrended.csv"
  era5_path: "data/era5_*.nc"
  
timeframe:
  start: "2017-01-01"
  end: "2021-12-31"
  train_end: "2019-12-31"
  val_end: "2020-12-31"
  test_end: "2021-12-31"
  
models:
  random_forest:
    n_estimators: 500
    max_features: "sqrt"
    pca_components: 10
```

## Data Setup

Before running the pipeline, organize your data files:

1. **Place your CSV files** in the `data/` directory:
   - `solarenergy2.csv` (original data)
   - `solarenergy2_daily_anomaly_base_only.csv` (base anomalies)
   - `solarenergy2_daily_anomaly_detrended.csv` (detrended anomalies)

2. **Run the setup script** to verify your data format:
   ```bash
   python setup_data.py
   ```

3. **Update configuration** if needed:
   ```bash
   cp config_sample.yaml config.yaml
   # Edit config.yaml with your specific file paths
   ```

## Usage

### Complete Pipeline

Run the entire forecasting pipeline:

```bash
python main.py
```

### Individual Components

Run specific components:

```bash
# Skip correlation analysis
python main.py --skip-correlations

# Skip baseline training
python main.py --skip-baselines

# Skip Random Forest training
python main.py --skip-rf

# Only create plots (requires trained models)
python main.py --plots-only
```

### Google Colab

```python
# Run in Colab cell
%run main.py
```

### Jupyter Notebooks

For interactive analysis, use the notebooks:

1. `01_correlations.ipynb` - Correlation analysis
2. `02_train_baselines.py` - Baseline models (convert to .ipynb)
3. `03_train_rf_grid.py` - Random Forest models (convert to .ipynb)

## Methodology

### Time Periods
- **Training**: 2017-2019
- **Validation**: 2020
- **Test**: 2021

### Forecasting Tracks

#### 1. Anomaly-based Forecasting
- **Target**: Detrended daily anomalies (`anom_mwh_detrended`)
- **Method**: Linear detrending → DOY climatology → anomaly calculation
- **Exclusion**: Known bad data window excluded from baseline fitting

#### 2. Total Value Forecasting
- **Target**: Raw daily generation (`gen_mwh`)
- **Method**: Direct prediction of total energy values

### Feature Engineering

#### Local Features (Korea Box)
- Area-weighted means: `ssrd_kr_sum`, `tcc_kr`, `tcwv_kr`, `wind10_kr`, `t2m_kr`
- Calendar features: `doy`, `sinDoy`, `cosDoy`, `dow`
- Lag features: Energy lags (1, 2, 3, 7 days), meteorological lags (1, 2, 3 days)

#### Extended Features (East Asia/Pacific)
- **Option A**: PCA compression of gridded data (memory efficient)
- **Option B**: Flattened grid features (high dimensional)
- Upstream region means and EOF/PCA components

### Models

#### Baseline Models
1. **Persistence**: Yesterday's value, same day last year
2. **Seasonal**: Day-of-year climatology
3. **Linear Regression**: Simple feature set

#### Random Forest Models
- **Local Features**: Korea-area weighted features
- **Extended Features**: East Asia/Pacific domain features
- **Hyperparameters**: 500 trees, sqrt features, optimized for Google Colab

### Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error (total targets only)
- **Skill Score**: Improvement over persistence (anomaly targets)

## Outputs

### Correlation Maps
- **Pearson correlation**: Energy anomalies vs SSRD
- **Spearman correlation**: Rank-based correlation
- **Visualization**: Cartopy maps with Korea outline

### Trained Models
- **Baseline**: `outputs/models/baseline_stats.pkl`
- **Random Forest**: `outputs/models/rf_grid_*.pkl`

### Evaluation Results
- **Metrics**: JSON files with detailed metrics
- **Comparison tables**: CSV files comparing all models
- **Plots**: Prediction vs truth, residual diagnostics, time series

### Feature Analysis
- **Importance**: Feature importance for Random Forest models
- **Correlation maps**: Spatial correlation patterns

## Three-Site Box Mode (Area-Weighted + Capacity-Weighted)

This mode aligns climate drivers with the physical aggregation of KPX generation across three named PV sites:

- Per-site boxes: One fixed-size square box (configurable half-side, default 0.5°) around each site:
  - `안산연성정수장태양광`
  - `세종시폐기물내립장태양광`
  - `영암에프원태양광`
- Within each box, daily climate variables are computed as area-weighted means (cos(latitude) weights). Intensive variables are averaged (e.g., `ssrd_sum`, `t2m`, `wind10`, `tcc`).
- Cross-site combine: Per-site series are combined to a single daily climate driver using capacity-weighted mean (weights from KPX `solarenergy2.csv`; falls back to equal if unavailable).
- Two forecasting tracks are supported for the combined driver:
  - Climate-total: capacity-weighted daily means (no detrending)
  - Climate-anom: detrended/DOY anomalies using the same exclusion window used for generation anomalies

Outputs added:
- `outputs/features/local/boxed_features.parquet` (both total and anomaly columns)
- `outputs/correlation_maps/boxed_correlation_summary.csv`
- `outputs/plots/boxed_corr_[VAR].png`

Integration points (no CLI changes):
- Correlations: Series-level correlations between generation anomaly and boxed climate anomalies run before spatial maps.
- Features: Local feature set now includes boxed drivers (total + anomaly) and their lags alongside existing calendar/energy lags.
- Models: Both baseline and RF consume the augmented local features; extended features path remains unchanged.

## Memory Management

The code is optimized for Google Colab with:

- **Grid downsampling**: Configurable stride for extended features
- **PCA compression**: Dimensionality reduction for gridded data
- **Memory limits**: Configurable maximum memory usage
- **Chunked processing**: Efficient handling of large datasets

## Key Features

### No-Leakage Validation
- All transformations fitted on training data only
- Validation/test data uses frozen parameters
- Temporal ordering strictly maintained

### Robust Anomaly Calculation
- Linear detrending with DOY climatology
- Exclusion of known bad data periods
- Handles missing values gracefully

### Comprehensive Evaluation
- Multiple metrics for different target types
- Statistical significance testing
- Model comparison across all approaches

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `pca_components` or enable `grid_downsample_extended`
2. **Missing data**: Check data file paths in `config.yaml`
3. **Import errors**: Ensure all dependencies are installed

### Google Colab Specific

1. **File upload**: Use Colab's file upload or mount Google Drive
2. **Memory limits**: Use `!pip install --upgrade --no-deps` for large packages
3. **Runtime restart**: May be needed for memory-intensive operations

## Citation

If you use this code in your research, please cite:

```
KPX Solar Energy Forecasting Pipeline
A comprehensive machine learning system for solar energy prediction
using ERA5 meteorological data and gridded Random Forest models.
```

## License

This project is provided as-is for research and educational purposes.
