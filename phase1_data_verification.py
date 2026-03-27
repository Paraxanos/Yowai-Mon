"""
Phase 1: Data Availability and Integrity Verification
Team Member: DATA
Contest: Replicate-2026
"""

import os
import sys
from pathlib import Path

# Base directories
BASE_DIR = Path(r"C:\Users\Pranjal\Desktop\Replicate_Research")
DATASETS_DIR = BASE_DIR / "Datasets"
SENTINEL2_DIR = DATASETS_DIR / "Sentinel-2"
RAINFALL_DIR = DATASETS_DIR / "Rainfall Data"
SOIL_MOISTURE_DIR = DATASETS_DIR / "Soil_moisture"

print("=" * 70)
print("PHASE 1: DATA AVAILABILITY VERIFICATION")
print("=" * 70)
print(f"\nBase Directory: {BASE_DIR}")
print(f"Datasets Directory: {DATASETS_DIR}")
print("\n" + "-" * 70)

# Task 1.1: Verify Data Availability
print("\nTASK 1.1: VERIFYING DATA AVAILABILITY\n")

# Expected files inventory
expected_files = {
    "Landslide Atlas (Ground Truth)": DATASETS_DIR / "LandslideAtlas_new_2023.pdf",
    "Copernicus DEM (Elevation)": DATASETS_DIR / "Copernicus_DEM_30m.tif",
    "Sentinel-1 SAR (2024-12-11)": DATASETS_DIR / "Sentinel-1_2024-12-11_SAR.tif",
    "Sentinel-1 SAR (2024-12-16)": DATASETS_DIR / "Sentinel-1_2024-12-16_SAR.tif",
    "Rainfall Data (Kerala)": RAINFALL_DIR / "kerala_rainfall_data.nc",
    "Soil Moisture": SOIL_MOISTURE_DIR / "Soil_Mositure.zip",
}

# Sentinel-2 bands
sentinel2_bands = {
    "B01 (Coastal/Aerosol)": "B01.tif",
    "B02 (Blue)": "B02.tif",
    "B03 (Green)": "B03.tif",
    "B04 (Red)": "B04.tif",
    "B05 (Vegetation Red Edge 1)": "B05.tif",
    "B06 (Vegetation Red Edge 2)": "B06.tif",
    "B07 (Vegetation Red Edge 3)": "B07.tif",
    "B08 (NIR)": "B08.tif",
    "B8A (NIR Narrow)": "B8A.tif",
    "B09 (Water Vapor)": "B09.tif",
    "B11 (SWIR 1)": "B11.tif",
    "B12 (SWIR 2)": "B12.tif",
}

file_inventory = {}
missing_files = []

print("\n--- Core Datasets ---")
for name, path in expected_files.items():
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        file_inventory[name] = {"path": str(path), "size_mb": size_mb, "exists": True}
        print(f"  ✓ {name}: {size_mb:.2f} MB")
    else:
        missing_files.append(name)
        file_inventory[name] = {"path": str(path), "exists": False}
        print(f"  ✗ {name}: MISSING")

print("\n--- Sentinel-2 Bands ---")
for name, filename in sentinel2_bands.items():
    path = SENTINEL2_DIR / filename
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        file_inventory[f"Sentinel-2 {name}"] = {"path": str(path), "size_mb": size_mb, "exists": True}
        print(f"  ✓ {name}: {size_mb:.2f} MB")
    else:
        missing_files.append(f"Sentinel-2 {name}")
        file_inventory[f"Sentinel-2 {name}"] = {"path": str(path), "exists": False}
        print(f"  ✗ {name}: MISSING")

# Summary
print("\n" + "-" * 70)
print("AVAILABILITY SUMMARY")
print("-" * 70)
total_files = len(file_inventory)
existing_files = sum(1 for f in file_inventory.values() if f["exists"])
total_size_mb = sum(f["size_mb"] for f in file_inventory.values() if f["exists"])

print(f"Total Expected Files: {total_files}")
print(f"Files Present: {existing_files}")
print(f"Files Missing: {len(missing_files)}")
print(f"Total Data Size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")

if missing_files:
    print(f"\n⚠ Missing Files:")
    for f in missing_files:
        print(f"    - {f}")
else:
    print("\n✓ All required files are present!")

# Save inventory for later use
inventory_log = BASE_DIR / "data_inventory.txt"
with open(inventory_log, "w") as f:
    f.write("DATA INVENTORY LOG - Replicate-2026\n")
    f.write("=" * 50 + "\n\n")
    for name, info in file_inventory.items():
        status = "PRESENT" if info["exists"] else "MISSING"
        size_str = f"{info['size_mb']:.2f} MB" if info["exists"] else "N/A"
        f.write(f"{name}: {status} - {size_str}\n")
        f.write(f"  Path: {info['path']}\n\n")
    f.write(f"\nTotal Size: {total_size_mb:.2f} MB\n")

print(f"\nInventory saved to: {inventory_log}")
print("\n" + "=" * 70)
print("TASK 1.1 COMPLETE")
print("=" * 70)
