
# HydroAnalysis: A Modular Hydrological Data Analysis Framework

HydroAnalysis is a modular Python package for hydrological data analysis, specifically designed for tasks such as precipitation dataset comparison, discharge-based flood event analysis, precipitation correction, temporal disaggregation, and insightful visualization. It is intended for research and operational workflows involving observed and modeled hydrometeorological data.

---

## 📦 Features

- 📈 Read and preprocess discharge & precipitation data from Excel/CSV
- 🌊 Identify and analyze flood events from discharge records
- ☔ Compare multiple precipitation datasets against observed data
- 🔧 Apply scaling-based corrections to model precipitation
- ⏱️ Disaggregate hourly precipitation data to half-hourly resolution
- 🗺️ Visualize station locations and spatial scaling patterns
- 📊 Create comparison and time series plots
- 🔀 Command-line interface for reproducible workflows

---

## 🛠️ Installation

```bash
git clone https://github.com/Saurav-JSU/hydroanalysis.git
cd hydroanalysis
pip install -r requirements.txt
```

---

## 📂 Input Requirements

HydroAnalysis supports both `.csv` and `.xlsx` formats.

### 1. **Discharge Data**
- **Required Columns**: `Date`, `Discharge`
- Optional: Multiple stations (each as a separate column), or pass `station_id`
- Format:
    ```csv
    Date,Discharge
    2020-01-01,123.4
    2020-01-02,130.1
    ```

### 2. **Precipitation Data**
- **Required Columns**: `Date`, `Station_ABC`, `Station_DEF`, ...
- Format:
    ```csv
    Date,Station_ABC,Station_DEF
    2020-01-01,3.4,2.1
    2020-01-02,5.7,3.3
    ```

### 3. **Station Metadata**
- Required for maps and spatial analyses.
- **Columns**: `Station_ID`, `Latitude`, `Longitude`, (`Type` optional)
- Format:
    ```csv
    Station_ID,Latitude,Longitude,Type
    Station_ABC,27.71,85.32,Precip
    Station_DEF,27.72,85.30,Discharge
    ```

---

## 🚀 Usage

### Python API Example

```python
from hydroanalysis.core.data_io import read_precipitation_data
from hydroanalysis.precipitation.comparison import calculate_accuracy_metrics

obs = read_precipitation_data("observed.xlsx")
pred = read_precipitation_data("model_output.xlsx")

metrics, merged = calculate_accuracy_metrics(obs, pred)
print(metrics)
```

---

## ⚙️ CLI Usage

Run any of these directly from your terminal:

### 1. **Flood Event Analysis**
```bash
python cli.py flood-events \
  --discharge path/to/discharge.csv \
  --precipitation path/to/precip.csv \
  --station Station_ABC \
  --percentile 95 \
  --duration 2 \
  --buffer 7 \
  --output results/floods
```

### 2. **Dataset Comparison**
```bash
python cli.py compare-datasets \
  --observed path/to/observed.csv \
  --datasets path/to/model1.csv path/to/model2.csv \
  --dataset-names GPM CHIRPS \
  --metadata station_metadata.csv \
  --output results/comparison
```

### 3. **Correction**
```bash
python cli.py correct-precipitation \
  --datasets-dir results/comparison \
  --metadata station_metadata.csv \
  --output results/corrected \
  --monthly-factors
```

### 4. **High-Resolution Disaggregation**
```bash
python cli.py create-high-resolution \
  --datasets-dir results/comparison \
  --dataset-files model1.csv model2.csv \
  --dataset-names GPM CHIRPS \
  --metadata station_metadata.csv \
  --output results/highres
```

---

## 🔁 Workflow Overview

```mermaid
graph TD
    A[read_precipitation_data()] --> B[calculate_accuracy_metrics()]
    B --> C[rank_datasets()]
    C --> D[calculate_scaling_factors()]
    D --> E[apply_scaling_factors()]
    E --> F[disaggregate_to_half_hourly()]
    A2[read_discharge_data()] --> G[identify_flood_events()]
    G --> H[plot_discharge()]
```

---

## 📁 Project Structure

```
hydroanalysis/
│
├── core/
│   ├── data_io.py
│   ├── utils.py
├── discharge/
│   └── flood_events.py
├── precipitation/
│   ├── comparison.py
│   ├── correction.py
│   └── disaggregation.py
├── visualization/
│   ├── comparison_plots.py
│   ├── timeseries.py
│   └── maps.py
├── cli.py
└── config/
    └── config.yaml
```

---

## 📊 Visualization Examples

- **Comparison Scatter Plots**
- **Time Series (Precipitation & Discharge)**
- **Flood Peak Annotation**
- **Scaling Factor Maps**
- **Best Dataset Pie Charts**

---

## ✅ Best Practices

- Make sure all files have a `Date` column in `YYYY-MM-DD` format.
- Use consistent station names across datasets.
- Validate your metadata to include lat/lon.
- Use CLI for reproducibility and batch processing.
- Log files will be saved as defined in `CONFIG`.

---

## 🧠 Future Enhancements

- Support for netCDF inputs
- Snowmelt and evapotranspiration modules
- Integration with USGS API and GEE exports

---

## 📬 Contact

**Saurav Bhattarai**  
Graduate Researcher, Jackson State University  
Feel free to raise issues or feature requests!

---

## 📄 License

MIT License – see the `LICENSE` file for details.
