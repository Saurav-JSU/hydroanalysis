"""
HydroAnalysis
=============

A comprehensive Python package for hydrological data analysis, focusing on:
1. Discharge and precipitation data processing
2. Flood event identification and analysis
3. Precipitation dataset comparison and correction
4. High-resolution precipitation disaggregation
"""

__version__ = '0.1.0'

# Import core functionality
from hydroanalysis.core.data_io import (
    read_discharge_data, 
    read_precipitation_data, 
    read_station_metadata,
    save_data
)

from hydroanalysis.core.utils import (
    setup_logging, 
    haversine_distance
)

# Import discharge modules
from hydroanalysis.discharge.flood_events import (
    identify_flood_events,
    analyze_flood_events
)

# Import precipitation modules
from hydroanalysis.precipitation.comparison import (
    calculate_accuracy_metrics,
    rank_datasets,
    identify_best_dataset_per_station
)

from hydroanalysis.precipitation.correction import (
    calculate_scaling_factors,
    apply_scaling_factors
)

from hydroanalysis.precipitation.disaggregation import (
    get_time_resolution,
    disaggregate_to_half_hourly,
    create_high_resolution_precipitation,
    create_high_resolution_precipitation_from_corrected
)

# Import visualization tools
from hydroanalysis.visualization.timeseries import (
    plot_discharge,
    plot_precipitation,
    plot_flood_analysis
)

from hydroanalysis.visualization.comparison_plots import (
    plot_dataset_comparison,
    plot_scaling_factor,
    plot_best_datasets
)

from hydroanalysis.visualization.maps import (
    plot_station_map,
    plot_scaling_factors_map
)