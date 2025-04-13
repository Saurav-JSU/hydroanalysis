
# HydroAnalysis: Modular Hydrological Data Analysis Toolkit

HydroAnalysis is a flexible Python-based framework for conducting comprehensive hydrological data analysis, with support for **precipitation correction**, **flood event identification**, **dataset comparison**, **temporal disaggregation**, and **visualization**. The toolkit is especially suited for workflows involving **observed and satellite-derived** precipitation or discharge datasets.

---

## 🔧 What This Toolkit Can Do

- ✅ Read precipitation and discharge data from `.csv` or `.xlsx`
- 🌊 Detect and analyze flood events based on discharge
- 📊 Compare observed vs. modeled precipitation at multiple stations
- ⚙️ Apply linear corrections (scaling) to model outputs
- ⏱️ Disaggregate hourly precipitation into half-hourly
- 🗺️ Map station metadata and spatially visualize scaling factors
- 📈 Create rich time series and scatter plots
- 🖥️ Run interactively in Python or via CLI for automation

---

## 📁 Directory Structure & Data Organization

Organize your project like this:

```
project/
├── data/
│   ├── discharge.csv
│   ├── observed_precip.xlsx
│   ├── model1.csv
│   ├── model2.csv
│   └── station_metadata.csv
├── results/
│   └── [outputs go here]
├── config/
│   └── config.yaml (optional)
├── hydroanalysis/
│   └── [core package files]
└── cli.py
```

---

## 📥 Input File Formats

### 1. Discharge Data (`.csv` or `.xlsx`)
- Required columns: `Date`, `Discharge`
```csv
Date,Discharge
2020-01-01,125.3
2020-01-02,133.8
```

### 2. Precipitation Data
- Required columns: `Date`, `Station_ABC`, `Station_DEF`, ...
```csv
Date,Station_ABC,Station_DEF
2020-01-01,4.2,3.9
2020-01-02,6.1,4.5
```

### 3. Station Metadata
- Columns: `Station_ID`, `Latitude`, `Longitude`, [`Type` optional]
```csv
Station_ID,Latitude,Longitude,Type
Station_ABC,27.7,85.3,Precip
Station_DEF,27.8,85.2,Discharge
```

---

## 🚀 How to Run the Code

### 🔁 Option 1: Use the Python API
```python
from hydroanalysis.core.data_io import read_precipitation_data
from hydroanalysis.precipitation.comparison import calculate_accuracy_metrics

obs_df = read_precipitation_data("data/observed_precip.xlsx")
pred_df = read_precipitation_data("data/model1.csv")
metrics, merged = calculate_accuracy_metrics(obs_df, pred_df)
```

### 🔁 Option 2: Use the Command Line Interface (CLI)

> 🔧 CLI Entry point: `cli.py`

```bash
python cli.py [command] [--arguments]
```

---

## 🔧 CLI Commands & Tasks

### 🔹 1. Identify and Analyze Flood Events
```bash
python cli.py flood-events   --discharge data/MDD.xlsx   --precipitation data/observed_precip.xlsx   --station "550.05"   --percentile 95   --duration 2   --buffer 7   --output results/floods
```

### 🔹 2. Compare Precipitation Datasets
```bash
python cli.py compare-datasets   --observed data/observed_precip.xlsx   --datasets data/model1.csv data/model2.csv   --dataset-names GPM CHIRPS   --metadata data/station_metadata.csv   --output results/comparison
```

### 🔹 3. Apply Correction (Scaling Factors)
```bash
python cli.py correct-precipitation   --datasets-dir results/comparison   --metadata data/station_metadata.csv   --output results/corrected   --monthly-factors
```

### 🔹 4. Create High-Resolution (Half-Hourly) Data
```bash
python cli.py create-high-resolution   --datasets-dir results/comparison   --dataset-files data/model1.csv data/model2.csv   --dataset-names GPM CHIRPS   --metadata data/station_metadata.csv   --output results/highres
```

### 🔹 5. Download Precipitation Data for Flood Events
```bash
python cli.py download-flood-precipitation --floods-dir results/floods --metadata data/station_metadata.csv --dataset era5

## ⚙️ Flexibility & Extensibility

### You Can:
- Run only the tasks you want (each command is independent).
- Use CLI for batch processing or API for interactive analysis.
- Plug in your own datasets by matching the column format.
- Modify thresholds, station selections, and file paths as needed.
- Extend the config via `hydroanalysis/config.py` or a custom `config.yaml`.

---

## 📊 Visualizations You Get

- 📈 Time series of discharge and precipitation
- 📉 Comparison scatter plots (with RMSE, MAE, KGE, etc.)
- 🗺️ Station location maps
- 🧭 Spatial distribution of scaling factors
- 🧾 Ranked dataset performance summaries

---

## 🧠 Tips & Best Practices

- ✅ Ensure your `Date` column is parseable (`YYYY-MM-DD`)
- ✅ All datasets should span the **same date range** for comparison
- ❌ Avoid NaNs unless you handle them in preprocessing
- 🧪 Test with a few stations before scaling up
- 🛠️ Enable detailed logs in `CONFIG` for debugging

---

## 🙋 FAQ

- **Q: Can I run everything with one command?**  
  Not currently, but you can chain the CLI commands in a shell script or Python driver.

- **Q: Can I use only a part of this package?**  
  Absolutely. Every module is callable independently.

- **Q: Does this support netCDF?**  
  Not yet. Planned for future releases.

---

## 🧑‍💻 Maintainer

**Saurav Bhattarai**  
Graduate Researcher, Jackson State University  
📬 Reach out for issues, collaborations, or suggestions.

---

## 📄 License

MIT License — See `LICENSE` file for details.
