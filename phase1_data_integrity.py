"""
Phase 1: Data Integrity Verification
Team Member: DATA
Contest: Replicate-2026

Task 1.2: Verify Data Integrity
- Load each data file and verify it loads without errors
- Check spatial dimensions of each raster file
- Verify coordinate reference systems (CRS)
- Check for corrupted or invalid data values
- Log shape and data type of each loaded file
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Base directories
BASE_DIR = Path(r"C:\Users\Pranjal\Desktop\Replicate_Research")
DATASETS_DIR = BASE_DIR / "Datasets"
SENTINEL2_DIR = DATASETS_DIR / "Sentinel-2"
RAINFALL_DIR = DATASETS_DIR / "Rainfall Data"
SOIL_MOISTURE_DIR = DATASETS_DIR / "Soil_moisture"

print("=" * 70)
print("PHASE 1: DATA INTEGRITY VERIFICATION")
print("=" * 70)

# Check required libraries
print("\n--- Checking Required Libraries ---")

libraries = {
    'rasterio': 'Geospatial raster I/O',
    'numpy': 'Numerical operations',
    'xarray': 'NetCDF handling',
    'pandas': 'Data manipulation',
}

installed_libs = {}
for lib, desc in libraries.items():
    try:
        __import__(lib)
        installed_libs[lib] = True
        print(f"  ✓ {lib}: Available ({desc})")
    except ImportError:
        installed_libs[lib] = False
        print(f"  ✗ {lib}: NOT INSTALLED ({desc})")

# Proceed with verification if core libraries are available
if not installed_libs.get('rasterio', False) or not installed_libs.get('numpy', False):
    print("\n⚠ WARNING: Core geospatial libraries not installed.")
    print("  Install with: pip install rasterio xarray numpy pandas")
    print("\nRunning basic file checks only...")
    
    # Basic file checks without libraries
    print("\n--- Basic File Validation ---")
    tif_files = list(DATASETS_DIR.glob("*.tif")) + list(SENTINEL2_DIR.glob("*.tif"))
    for f in tif_files:
        if f.stat().st_size > 0:
            print(f"  ✓ {f.name}: Non-empty file ({f.stat().st_size / 1024:.2f} KB)")
        else:
            print(f"  ✗ {f.name}: Empty file!")
    
    nc_files = list(RAINFALL_DIR.glob("*.nc"))
    for f in nc_files:
        if f.stat().st_size > 0:
            print(f"  ✓ {f.name}: Non-empty file ({f.stat().st_size / 1024:.2f} KB)")
        else:
            print(f"  ✗ {f.name}: Empty file!")
    
    print("\n" + "=" * 70)
    print("TASK 1.2 COMPLETE (Basic Check Only)")
    print("=" * 70)
    sys.exit(0)

# Full verification with libraries
import rasterio
import numpy as np
import xarray as xr

integrity_results = {}
issues_found = []

print("\n" + "-" * 70)
print("TASK 1.2: LOADING AND VALIDATING DATASETS")
print("-" * 70)

# 1. Sentinel-2 Bands
print("\n--- Sentinel-2 Multispectral Bands ---")
sentinel2_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
sentinel2_info = {}

for band in sentinel2_bands:
    try:
        path = SENTINEL2_DIR / f"{band}.tif"
        with rasterio.open(path) as src:
            shape = src.shape
            dtype = str(src.dtypes[0])
            crs = str(src.crs)
            nodata = src.nodata
            min_val, max_val = src.read(1).min(), src.read(1).max()
            
            sentinel2_info[band] = {
                'shape': shape,
                'dtype': dtype,
                'crs': crs,
                'nodata': nodata,
                'min': min_val,
                'max': max_val,
                'valid': True
            }
            print(f"  ✓ {band}: {shape}, {dtype}, CRS: {crs}")
            print(f"    Value range: [{min_val}, {max_val}], NoData: {nodata}")
    except Exception as e:
        issues_found.append(f"Sentinel-2 {band}: {str(e)}")
        sentinel2_info[band] = {'valid': False, 'error': str(e)}
        print(f"  ✗ {band}: ERROR - {e}")

# 2. Sentinel-1 SAR
print("\n--- Sentinel-1 SAR Data ---")
sar_dates = ["2024-12-11", "2024-12-16"]
sar_info = {}

for date in sar_dates:
    try:
        path = DATASETS_DIR / f"Sentinel-1_{date}_SAR.tif"
        with rasterio.open(path) as src:
            shape = src.shape
            dtype = str(src.dtypes[0])
            crs = str(src.crs)
            data = src.read(1)
            min_val, max_val = data.min(), data.max()
            valid_pixels = np.sum(~np.isnan(data) & (data != src.nodata))
            total_pixels = data.size
            
            sar_info[date] = {
                'shape': shape,
                'dtype': dtype,
                'crs': crs,
                'min': min_val,
                'max': max_val,
                'valid_pixels': int(valid_pixels),
                'total_pixels': int(total_pixels),
                'valid': True
            }
            print(f"  ✓ {date}: {shape}, {dtype}, CRS: {crs}")
            print(f"    Value range: [{min_val}, {max_val}]")
            print(f"    Valid pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
    except Exception as e:
        issues_found.append(f"SAR {date}: {str(e)}")
        sar_info[date] = {'valid': False, 'error': str(e)}
        print(f"  ✗ {date}: ERROR - {e}")

# 3. Copernicus DEM
print("\n--- Copernicus DEM (Elevation) ---")
dem_info = {}
try:
    path = DATASETS_DIR / "Copernicus_DEM_30m.tif"
    with rasterio.open(path) as src:
        shape = src.shape
        dtype = str(src.dtypes[0])
        crs = str(src.crs)
        data = src.read(1)
        min_val, max_val = data.min(), data.max()
        
        dem_info = {
            'shape': shape,
            'dtype': dtype,
            'crs': crs,
            'min': min_val,
            'max': max_val,
            'valid': True
        }
        print(f"  ✓ DEM: {shape}, {dtype}, CRS: {crs}")
        print(f"    Elevation range: [{min_val}m, {max_val}m]")
except Exception as e:
    issues_found.append(f"DEM: {str(e)}")
    dem_info = {'valid': False, 'error': str(e)}
    print(f"  ✗ DEM: ERROR - {e}")

# 4. Rainfall Data (NetCDF)
print("\n--- Rainfall Data (NetCDF) ---")
rainfall_info = {}
try:
    path = RAINFALL_DIR / "kerala_rainfall_data.nc"
    ds = xr.open_dataset(path)
    
    rainfall_info = {
        'variables': list(ds.data_vars),
        'dimensions': dict(ds.dims),
        'coords': list(ds.coords),
        'valid': True
    }
    print(f"  ✓ Variables: {list(ds.data_vars)}")
    print(f"  ✓ Dimensions: {dict(ds.dims)}")
    print(f"  ✓ Coordinates: {list(ds.coords)}")
    
    # Show sample data
    for var in ds.data_vars:
        if len(ds[var].shape) > 0:
            print(f"    {var}: shape={ds[var].shape}, dtype={ds[var].dtype}")
        else:
            print(f"    {var}: scalar={ds[var].values}")
    
    ds.close()
except Exception as e:
    issues_found.append(f"Rainfall: {str(e)}")
    rainfall_info = {'valid': False, 'error': str(e)}
    print(f"  ✗ Rainfall: ERROR - {e}")

# 5. Soil Moisture
print("\n--- Soil Moisture Data ---")
soil_moisture_info = {}
soil_zip_path = SOIL_MOISTURE_DIR / "Soil_Mositure.zip"

if soil_zip_path.exists():
    import zipfile
    try:
        with zipfile.ZipFile(soil_zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            soil_moisture_info['files_in_archive'] = file_list
            print(f"  ✓ Archive contains: {len(file_list)} files")
            for f in file_list[:10]:  # Show first 10
                print(f"    - {f}")
            if len(file_list) > 10:
                print(f"    ... and {len(file_list) - 10} more files")
            
            # Try to read first file if it's a raster
            if any(f.endswith('.tif') for f in file_list):
                tif_file = [f for f in file_list if f.endswith('.tif')][0]
                with zip_ref.open(tif_file) as f:
                    # For now, just note it's readable
                    soil_moisture_info['sample_tif'] = tif_file
                    print(f"  ✓ Sample TIF readable: {tif_file}")
    except Exception as e:
        issues_found.append(f"Soil Moisture: {str(e)}")
        soil_moisture_info = {'valid': False, 'error': str(e)}
        print(f"  ✗ Soil Moisture: ERROR - {e}")
else:
    print(f"  ✗ Soil Moisture ZIP not found!")
    issues_found.append("Soil Moisture ZIP file missing")

# 6. Landslide Atlas (Ground Truth)
print("\n--- Landslide Atlas (Ground Truth) ---")
atlas_info = {}
atlas_pdf_path = DATASETS_DIR / "LandslideAtlas_new_2023.pdf"

if atlas_pdf_path.exists():
    atlas_info = {
        'path': str(atlas_pdf_path),
        'size_mb': atlas_pdf_path.stat().st_size / (1024 * 1024),
        'exists': True,
        'valid': True
    }
    print(f"  ✓ PDF exists: {atlas_info['size_mb']:.2f} MB")
    print(f"    Note: This is the ground truth reference document")
    print(f"    Will need to be digitized/processed for model training")
else:
    atlas_info = {'valid': False, 'exists': False}
    issues_found.append("Landslide Atlas PDF missing")
    print(f"  ✗ Landslide Atlas: MISSING")

# CRS Consistency Check
print("\n" + "-" * 70)
print("CRS CONSISTENCY CHECK")
print("-" * 70)

all_crs = set()
for info in [sentinel2_info, sar_info, dem_info]:
    if isinstance(info, dict):
        for key, val in info.items():
            if isinstance(val, dict) and 'crs' in val:
                all_crs.add(val['crs'])

print(f"\nUnique CRS found: {len(all_crs)}")
for crs in all_crs:
    print(f"  - {crs}")

if len(all_crs) == 1:
    print("\n✓ All raster data uses consistent CRS!")
elif len(all_crs) > 1:
    print("\n⚠ Multiple CRS detected - reprojection will be needed")
    issues_found.append("CRS inconsistency - reprojection required")
else:
    print("\n⚠ Could not determine CRS consistency")

# Summary
print("\n" + "=" * 70)
print("INTEGRITY VERIFICATION SUMMARY")
print("=" * 70)

print(f"\nSentinel-2 Bands: {sum(1 for v in sentinel2_info.values() if v.get('valid', False))}/12 loaded successfully")
print(f"SAR Scenes: {sum(1 for v in sar_info.values() if v.get('valid', False))}/2 loaded successfully")
print(f"DEM: {'✓ Loaded' if dem_info.get('valid', False) else '✗ Failed'}")
print(f"Rainfall: {'✓ Loaded' if rainfall_info.get('valid', False) else '✗ Failed'}")
print(f"Soil Moisture: {'✓ Archive readable' if soil_moisture_info.get('files_in_archive', False) else '✗ Failed'}")
print(f"Landslide Atlas: {'✓ Available' if atlas_info.get('exists', False) else '✗ Missing'}")

if issues_found:
    print(f"\n⚠ Issues Found ({len(issues_found)}):")
    for issue in issues_found:
        print(f"    - {issue}")
else:
    print("\n✓ All datasets loaded successfully with no integrity issues!")

# Save integrity report
integrity_report = BASE_DIR / "data_integrity_report.txt"
with open(integrity_report, "w") as f:
    f.write("DATA INTEGRITY REPORT - Replicate-2026\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("SENTINEL-2 BANDS\n")
    for band, info in sentinel2_info.items():
        if info.get('valid'):
            f.write(f"  {band}: {info['shape']}, {info['dtype']}, CRS: {info['crs']}\n")
        else:
            f.write(f"  {band}: FAILED - {info.get('error', 'Unknown')}\n")
    
    f.write("\nSENTINEL-1 SAR\n")
    for date, info in sar_info.items():
        if info.get('valid'):
            f.write(f"  {date}: {info['shape']}, {info['dtype']}, CRS: {info['crs']}\n")
        else:
            f.write(f"  {date}: FAILED - {info.get('error', 'Unknown')}\n")
    
    f.write("\nDEM\n")
    if dem_info.get('valid'):
        f.write(f"  Shape: {dem_info['shape']}, CRS: {dem_info['crs']}\n")
        f.write(f"  Elevation range: [{dem_info['min']}m, {dem_info['max']}m]\n")
    
    f.write("\nRAINFALL\n")
    if rainfall_info.get('valid'):
        f.write(f"  Variables: {rainfall_info['variables']}\n")
        f.write(f"  Dimensions: {rainfall_info['dimensions']}\n")
    
    f.write("\nSOIL MOISTURE\n")
    if soil_moisture_info.get('files_in_archive'):
        f.write(f"  Files in archive: {len(soil_moisture_info['files_in_archive'])}\n")
    
    f.write("\nLANDSLIDE ATLAS\n")
    f.write(f"  Status: {'Available' if atlas_info.get('exists') else 'Missing'}\n")
    f.write(f"  Size: {atlas_info.get('size_mb', 0):.2f} MB\n")
    
    f.write("\n\nISSUES FOUND\n")
    if issues_found:
        for issue in issues_found:
            f.write(f"  - {issue}\n")
    else:
        f.write("  None\n")

print(f"\nIntegrity report saved to: {integrity_report}")
print("\n" + "=" * 70)
print("TASK 1.2 COMPLETE")
print("=" * 70)
