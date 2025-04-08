"""
Configuration settings for the HydroAnalysis package.
"""

# Default configuration
DEFAULT_CONFIG = {
    # Logging settings
    'logging': {
        'level': 'INFO',
        'file': 'hydroanalysis.log',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    },
    
    # Flood event identification settings
    'flood_events': {
        'percentile_threshold': 95,  # 95th percentile for flood threshold
        'min_duration': 2,           # Minimum flood duration in days
        'buffer_days': 7             # Days before/after flood for analysis
    },
    
    # Precipitation dataset comparison settings
    'comparison': {
        'min_data_points': 10,       # Minimum number of data points for valid comparison
        'extreme_value_threshold': 500  # Threshold for extreme precipitation values (mm)
    },
    
    # Precipitation correction settings
    'correction': {
        'min_scaling_factor': 0.1,   # Minimum allowed scaling factor
        'max_scaling_factor': 10.0,  # Maximum allowed scaling factor
        'default_factor': 1.0        # Default factor when calculation fails
    },
    
    # Disaggregation settings
    'disaggregation': {
        'default_resolution': 60,    # Default time resolution in minutes
        'target_resolution': 30      # Target resolution for disaggregation in minutes
    },
    
    # Visualization settings
    'visualization': {
        'dpi': 300,                  # DPI for saved figures
        'figsize': (10, 6),          # Default figure size (width, height) in inches
        'style': 'ggplot'            # Matplotlib style
    },
    
    # Output settings
    'output': {
        'default_dir': 'hydroanalysis_results'  # Default output directory
    }
}

# Attempt to load user configuration if it exists
try:
    import importlib.util
    import os
    
    user_config_path = os.path.expanduser('~/.hydroanalysis/config.py')
    
    if os.path.exists(user_config_path):
        spec = importlib.util.spec_from_file_location('user_config', user_config_path)
        user_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_config_module)
        
        if hasattr(user_config_module, 'CONFIG'):
            # Merge user configuration with defaults
            import copy
            CONFIG = copy.deepcopy(DEFAULT_CONFIG)
            
            # Update with user settings
            for section, settings in user_config_module.CONFIG.items():
                if section in CONFIG:
                    CONFIG[section].update(settings)
                else:
                    CONFIG[section] = settings
        else:
            CONFIG = DEFAULT_CONFIG
    else:
        CONFIG = DEFAULT_CONFIG
except Exception as e:
    # If anything goes wrong, use default configuration
    CONFIG = DEFAULT_CONFIG
    print(f"Warning: Could not load user configuration. Using defaults. Error: {e}")