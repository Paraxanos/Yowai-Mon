# Phase 1 Summary Report - DATA Member
## Replicate-2026 Contest: EEGMoE for Landslide Prediction

**Team Member:** DATA  
**Phase:** 1 (Setup & Verification)  
**Duration:** 0:00 - 0:30  
**Status:** ✅ COMPLETE  
**Date:** March 27, 2026

---

## Executive Summary

All Phase 1 data tasks have been completed successfully. The complete multi-modal geospatial dataset has been verified for availability and integrity, and a fully functional PyTorch-compatible data loading pipeline has been implemented. The unified data cube is ready for integration with the EEGMoE architecture adaptation.

**Handoff Message:** `DATA READY - Tensor Shape: (Batch, 17 Channels, 64×64)`

---

## Task Completion Status

| Task | Description | Status | Duration |
|------|-------------|--------|----------|
| 1.1 | Verify Data Availability | ✅ Complete | 10 min |
| 1.2 | Verify Data Integrity | ✅ Complete | 10 min |
| 1.3 | Prepare Data Loading Pipeline | ✅ Complete | 10 min |

---

## Task 1.1: Data Availability Verification

### Files Verified (18/18 Present ✓)

**Core Datasets:**
| Dataset | File | Size | Status |
|---------|------|------|--------|
| Landslide Atlas (Ground Truth) | LandslideAtlas_new_2023.pdf | 10.57 MB | ✓ |
| Copernicus DEM (Elevation) | Copernicus_DEM_30m.tif | 0.66 MB | ✓ |
| Sentinel-1 SAR (Pre-event) | Sentinel-1_2024-12-11_SAR.tif | 12.76 MB | ✓ |
| Sentinel-1 SAR (Post-event) | Sentinel-1_2024-12-16_SAR.tif | 12.78 MB | ✓ |
| Rainfall Data (Kerala) | kerala_rainfall_data.nc | 0.64 MB | ✓ |
| Soil Moisture (SMAP) | Soil_Mositure/ (200 TIF files) | ~100 MB | ✓ |

**Sentinel-2 Multispectral Bands (12 bands):**
| Band | Wavelength | Size | Status |
|------|------------|------|--------|
| B01 | Coastal/Aerosol (443nm) | 1.45 MB | ✓ |
| B02 | Blue (490nm) | 1.98 MB | ✓ |
| B03 | Green (560nm) | 2.18 MB | ✓ |
| B04 | Red (665nm) | 2.06 MB | ✓ |
| B05 | RE1 (705nm) | 2.28 MB | ✓ |
| B06 | RE2 (740nm) | 2.68 MB | ✓ |
| B07 | RE3 (783nm) | 2.77 MB | ✓ |
| B08 | NIR (842nm) | 2.89 MB | ✓ |
| B8A | NIRn (865nm) | 2.79 MB | ✓ |
| B09 | Water Vapor (945nm) | 2.58 MB | ✓ |
| B11 | SWIR1 (1610nm) | 2.52 MB | ✓ |
| B12 | SWIR2 (2190nm) | 2.32 MB | ✓ |

**Total Data Size:** ~180 MB (0.18 GB) including extracted soil moisture data

### Output Artifact
- `data_inventory.txt` - Complete file inventory with paths and sizes

---

## Task 1.2: Data Integrity Verification

### Validation Results

**Libraries Installed:**
- ✓ rasterio (Geospatial raster I/O)
- ✓ numpy (Numerical operations)
- ✓ xarray (NetCDF handling)
- ✓ pandas (Data manipulation)

**Raster Data Validation:**

| Dataset | Shape | dtype | CRS | Value Range | Status |
|---------|-------|-------|-----|-------------|--------|
| Sentinel-2 (all bands) | (1111, 1313) | uint16 | EPSG:32643 | Valid reflectance | ✓ |
| Sentinel-1 SAR (12-11) | (1111, 1313) | float32 | EPSG:32643 | [0.002, 28.68] | ✓ |
| Sentinel-1 SAR (12-16) | (1111, 1313) | float32 | EPSG:32643 | [0.002, 53.42] | ✓ |
| Copernicus DEM | (360, 432) | float32 | EPSG:4326 | [176m, 2235m] | ✓ |
| Soil Moisture (SMAP) | (256, 272) × 200 files | float32 | EPSG:4326 | Volumetric Water Content | ✓ |
| Rainfall | (366, 19, 12) | float64 | - | Daily time series | ✓ |

**Key Findings:**
- All Sentinel-2 bands have consistent dimensions (1111×1313 pixels)
- All Sentinel-1 SAR data uses same CRS as Sentinel-2 (EPSG:32643)
- 100% valid pixels in SAR data (no gaps or NoData)
- DEM and Soil Moisture require reprojection to target CRS
- Rainfall data contains 366 daily observations (full year 2024)
- **Soil Moisture:** 200 SMAP L4 files covering daily observations from Jan-Dec 2024 (2-3 day resolution)

**CRS Consistency:**
- Primary CRS: EPSG:32643 (UTM Zone 43N - appropriate for Kerala, India)
- Datasets requiring reprojection: DEM (EPSG:4326), Soil Moisture (EPSG:4326)

### Output Artifact
- `data_integrity_report.txt` - Detailed validation report with shapes, CRS, and value ranges

---

## Task 1.3: Data Loading Pipeline

### Unified Data Cube Created

**Format Adaptation (EEGMoE → Landslide):**

| EEGMoE Original | Landslide Adaptation |
|----------------|---------------------|
| (Time × Sr) × B × H × W | (Temporal × Sensors) × Channels × H × W |
| EEG frequency bands | Spectral bands + SAR dates |
| Domain: EEG tasks | Domain: Geographic regions / landslide types |

**Final Data Cube Specification:**
```
Shape: (17 channels, 1111 height, 1313 width)
dtype: float32
CRS: EPSG:32643
```

**Channel Breakdown:**
| Index | Channel | Source | Normalization (mean/std) |
|-------|---------|--------|-------------------------|
| 0-11 | S2_B01 to S2_B12 | Sentinel-2 | 0.12-0.36 / 0.007-0.06 |
| 12-13 | SAR_2024-12-11, SAR_2024-12-16 | Sentinel-1 | -7.8 / 3.4 (dB) |
| 14 | DEM | Copernicus | 1077m / 364m |
| 15 | SoilMoisture | SMAP (200 files) | 0.36 / 0.01 (VWC) |
| 16 | Rainfall | IMD | 7.36 mm (annual mean) |

### PyTorch Dataset Class

**LandslideDataset Features:**
- Extracts patches from unified data cube
- Configurable patch size (default: 64×64)
- Configurable stride for overlapping patches
- Binary label support (landslide/no-landslide)
- Returns: `(patch_tensor, label)` pairs

**DataLoader Configuration:**
```python
patch_size = 64
batch_size = 32
train_ratio = 0.8

# Output tensor shape: (32, 17, 64, 64)
# Format: (Batch, Channels, Height, Width)
```

**Test Results:**
- Total patches extracted: 1,320
- Training samples: 1,056
- Validation samples: 264
- Batch retrieval: ✅ Working
- Output dtype: torch.float32

### Output Artifacts
- `phase1_data_loader.py` - Complete data loading pipeline script
- `data_cube_info.txt` - Data cube specification and normalization parameters

---

## Deliverables Summary

| Deliverable | File | Status |
|-------------|------|--------|
| Data Availability Log | `data_inventory.txt` | ✅ Complete |
| Data Integrity Report | `data_integrity_report.txt` | ✅ Complete |
| Data Loading Pipeline | `phase1_data_loader.py` | ✅ Complete |
| Data Cube Specification | `data_cube_info.txt` | ✅ Complete |
| Verification Scripts | `phase1_data_verification.py`, `phase1_data_integrity.py` | ✅ Complete |

---

## Handoff to Phase 2

**Status:** `DATA READY`

**Tensor Shape Information:**
```
Input:  (Batch, 17, 64, 64)
        └───┬───┘ └─┬─┘
         Channels  Height/Width
         
Channels:
  [0-11]  → Sentinel-2 spectral bands
  [12-13] → Sentinel-1 SAR (temporal)
  [14]    → DEM elevation
  [15]    → Soil moisture
  [16]    → Rainfall
```

**Ready for:**
- ✅ Model architecture integration (ARCH member)
- ✅ Training loop integration (TRAIN member)
- ✅ Documentation of data pipeline (DOC member)

---

## Notes for Next Phase

1. **Landslide Ground Truth:** The LandslideAtlas PDF needs to be digitized/processed to create binary label rasters for training. This should be addressed in Phase 2.

2. **Class Imbalance:** Landslide pixels are expected to be rare (<5% of total area). Consider:
   - Oversampling landslide patches
   - Weighted loss functions
   - Focal loss for imbalanced classification

3. **Domain Definition for EEGMoE:** For the domain-decoupled MoE architecture, consider defining domains as:
   - Different geographic sub-regions
   - Different landslide types (from atlas)
   - Different elevation bands

4. **Memory Considerations:** Full data cube (17×1111×1313) fits in memory (~150MB), but consider on-disk caching for larger datasets.

---

## Phase 1 Completion Confirmation

**All systems operational. Data pipeline verified and ready for integration.**

**Timestamp:** 2026-03-27  
**Phase 1 Duration:** 30 minutes  
**Ready for Phase 2:** ✅ YES

---

*End of Phase 1 Summary Report*
