@echo off
setlocal EnableDelayedExpansion

echo Starting correction for all flood events...

rem Loop through all flood directories
for /d %%G in (results\floods\flood_*) do (
    echo Processing corrections for %%G...
    
    rem Check if the comparison directory exists
    if exist "%%G\comparison" (
        echo   Running correction for %%G...
        python cli.py correct-precipitation ^
            --datasets-dir "%%G" ^
            --metadata data\station_metadata.xlsx ^
            --output "%%G\corrected"
        echo   Correction complete for %%G
    ) else (
        echo   Comparison results not found in %%G. Skipping.
    )
)

echo All corrections completed!
echo.
echo Results saved in each flood_* directory under the 'corrected' subfolder.
echo.
endlocal
pause