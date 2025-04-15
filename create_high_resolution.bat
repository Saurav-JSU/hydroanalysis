@echo off
setlocal EnableDelayedExpansion

echo Starting high-resolution data creation for all flood events...

rem Loop through all flood directories
for /d %%G in (results\floods\flood_*) do (
    echo Processing %%G...
    
    rem Check if the corrected directory exists
    if exist "%%G\corrected" (
        echo   Creating high-resolution data for %%G...
        
        rem Check for IMERG file to use as pattern
        set "IMERG_PARAM="
        
        if exist "%%G\imerg_hourly_precipitation.csv" (
            echo   Found IMERG hourly data file
            set "IMERG_PARAM=--imerg-file "%%G\imerg_hourly_precipitation.csv""
        ) else if exist "%%G\gee_imerg\hourly_precipitation.csv" (
            echo   Found IMERG hourly data in gee_imerg folder
            set "IMERG_PARAM=--imerg-file "%%G\gee_imerg\hourly_precipitation.csv""
        ) else (
            echo   No IMERG pattern file found. Will use statistical disaggregation.
        )
        
        echo   Running high-resolution creation...
        python cli.py create-high-resolution --datasets-dir "%%G" --output "%%G\highres" %IMERG_PARAM%
            
        echo   High-resolution creation complete for %%G
    ) else (
        echo   Corrected data not found in %%G. Make sure to run correct-precipitation first.
    )
)

echo All high-resolution data creation completed!
echo.
echo Results saved in each flood_* directory under the 'highres' subfolder.
echo.
endlocal
pause