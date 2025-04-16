@echo off
call conda activate orise_compare

REM Run for ERA5
python cli.py download-flood-precipitation ^
  --floods-dir results/floods ^
  --metadata data/filtered_station_metadata.xlsx ^
  --dataset era5 ^
  --resolution both ^
  --adjust-timezone ^
  --nepal-convention

REM Run for IMERG
python cli.py download-flood-precipitation ^
  --floods-dir results/floods ^
  --metadata data/filtered_station_metadata.xlsx ^
  --dataset imerg ^
  --resolution both ^
  --adjust-timezone ^
  --nepal-convention

REM Run for GSMaP
python cli.py download-flood-precipitation ^
  --floods-dir results/floods ^
  --metadata data/filtered_station_metadata.xlsx ^
  --dataset gsmap ^
  --resolution both ^
  --adjust-timezone ^
  --nepal-convention

pause
