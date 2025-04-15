@echo off
setlocal EnableDelayedExpansion

echo Starting comparisons for all flood events...

rem Loop through all flood directories
for /d %%G in (results\floods\flood_*) do (
    echo Processing %%G...
    
    rem Check if the necessary files exist
    if exist "%%G\precipitation.csv" (
        echo   Checking available datasets...
        
        rem Initialize variables
        set "DATASETS="
        set "DATASET_NAMES="
        set "DATASET_COUNT=0"
        
        rem Check for ERA5
        if exist "%%G\gee_era5\daily_precipitation.csv" (
            rem Check if file has content (not empty)
            for %%F in ("%%G\gee_era5\daily_precipitation.csv") do (
                if %%~zF GTR 0 (
                    echo     Found ERA5 data
                    set DATASETS=!DATASETS! "%%G\gee_era5\daily_precipitation.csv"
                    set DATASET_NAMES=!DATASET_NAMES! ERA5
                    set /a DATASET_COUNT+=1
                ) else (
                    echo     ERA5 file exists but is empty. Skipping.
                )
            )
        )
        
        rem Check for IMERG
        if exist "%%G\gee_imerg\daily_precipitation.csv" (
            rem Check if file has content (not empty)
            for %%F in ("%%G\gee_imerg\daily_precipitation.csv") do (
                if %%~zF GTR 0 (
                    echo     Found IMERG data
                    set DATASETS=!DATASETS! "%%G\gee_imerg\daily_precipitation.csv"
                    set DATASET_NAMES=!DATASET_NAMES! IMERG
                    set /a DATASET_COUNT+=1
                ) else (
                    echo     IMERG file exists but is empty. Skipping.
                )
            )
        )
        
        rem Check for GSMaP
        if exist "%%G\gee_gsmap\daily_precipitation.csv" (
            rem Check if file has content (not empty)
            for %%F in ("%%G\gee_gsmap\daily_precipitation.csv") do (
                if %%~zF GTR 0 (
                    echo     Found GSMaP data
                    set DATASETS=!DATASETS! "%%G\gee_gsmap\daily_precipitation.csv"
                    set DATASET_NAMES=!DATASET_NAMES! GSMaP
                    set /a DATASET_COUNT+=1
                ) else (
                    echo     GSMaP file exists but is empty. Skipping.
                )
            )
        ) else (
            if exist "%%G\gee_gsmap" (
                echo     GSMaP folder exists but missing data file. You may need to download GSMaP data.
            )
        )
        
        rem Only run comparison if at least one dataset exists
        if !DATASET_COUNT! GTR 0 (
            echo   Running comparison with !DATASET_COUNT! datasets...
            python cli.py compare-datasets ^
                --observed "%%G\precipitation.csv" ^
                --datasets!DATASETS! ^
                --dataset-names!DATASET_NAMES! ^
                --metadata data\station_metadata.xlsx ^
                --output "%%G\comparison"
            echo   Comparison complete for %%G
        ) else (
            echo   No valid precipitation datasets found in %%G. Skipping.
        )
    ) else (
        echo   Missing observed precipitation file in %%G. Skipping.
    )
)

echo All comparisons completed!
echo.
echo NOTE: If GSMaP data is missing, you can download it with:
echo python cli.py download-flood-precipitation --floods-dir results/floods --metadata data/station_metadata.xlsx --dataset gsmap --resolution both
echo.
endlocal
pause