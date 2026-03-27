"""
Phase 1: Data Loading Pipeline for Replicate-2026
Team Member: DATA
Contest: Replicate-2026 - EEGMoE Architecture for Landslide Prediction

Task 1.3: Prepare Data Loading Pipeline
- Create a modular data loader for all input modalities
- Implement data normalization and preprocessing
- Create unified data cube from multi-modal inputs
- Prepare PyTorch-compatible dataset class

This pipeline adapts the EEGMoE 4D input format concept for geospatial data:
EEGMoE: (Time × Frequency Bands × Height × Width)
Adapted: (Time/SAR Dates × Sensor Bands × Height × Width)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.warp import calculate_default_transform, reproject, Resampling
from torch.utils.data import Dataset, DataLoader
import torch

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\Pranjal\Desktop\Replicate_Research")
DATASETS_DIR = BASE_DIR / "Datasets"
SENTINEL2_DIR = DATASETS_DIR / "Sentinel-2"
RAINFALL_DIR = DATASETS_DIR / "Rainfall Data"
SOIL_MOISTURE_DIR = DATASETS_DIR / "Soil_moisture"

# Target CRS and resolution (based on Sentinel-2 data)
TARGET_CRS = "EPSG:32643"
TARGET_RESOLUTION = 10.0  # meters

# Sentinel-2 band configuration
SENTINEL2_BANDS = {
    'B01': {'name': 'Coastal/Aerosol', 'wavelength': '443nm'},
    'B02': {'name': 'Blue', 'wavelength': '490nm'},
    'B03': {'name': 'Green', 'wavelength': '560nm'},
    'B04': {'name': 'Red', 'wavelength': '665nm'},
    'B05': {'name': 'RE1', 'wavelength': '705nm'},
    'B06': {'name': 'RE2', 'wavelength': '740nm'},
    'B07': {'name': 'RE3', 'wavelength': '783nm'},
    'B08': {'name': 'NIR', 'wavelength': '842nm'},
    'B8A': {'name': 'NIRn', 'wavelength': '865nm'},
    'B09': {'name': 'Water Vapor', 'wavelength': '945nm'},
    'B11': {'name': 'SWIR1', 'wavelength': '1610nm'},
    'B12': {'name': 'SWIR2', 'wavelength': '2190nm'},
}

# Data normalization parameters (will be computed from data)
NORMALIZATION_PARAMS = {}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_sentinel2_bands(bands: List[str] = None) -> Tuple[np.ndarray, rasterio.transform.Affine, str]:
    """
    Load Sentinel-2 multispectral bands and stack into a single array.
    
    Returns:
        stacked_data: (bands, height, width) array
        transform: GeoTransform for spatial referencing
        crs: Coordinate Reference System
    """
    if bands is None:
        bands = list(SENTINEL2_BANDS.keys())
    
    print(f"\nLoading Sentinel-2 bands: {bands}")
    
    arrays = []
    transform = None
    crs = None
    
    for band in bands:
        path = SENTINEL2_DIR / f"{band}.tif"
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            # Apply scaling factor for Sentinel-2 surface reflectance
            data = data * 0.0001  # Convert to reflectance
            arrays.append(data)
            
            if transform is None:
                transform = src.transform
                crs = src.crs
    
    stacked = np.stack(arrays, axis=0)
    print(f"  Sentinel-2 shape: {stacked.shape}")
    
    return stacked, transform, crs


def load_sentinel1_sar(dates: List[str] = None) -> Tuple[np.ndarray, rasterio.transform.Affine, str]:
    """
    Load Sentinel-1 SAR scenes for multiple dates.
    
    Returns:
        stacked_data: (dates, height, width) array
        transform: GeoTransform
        crs: Coordinate Reference System
    """
    if dates is None:
        dates = ["2024-12-11", "2024-12-16"]
    
    print(f"\nLoading Sentinel-1 SAR scenes: {dates}")
    
    arrays = []
    transform = None
    crs = None
    
    for date in dates:
        path = DATASETS_DIR / f"Sentinel-1_{date}_SAR.tif"
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            # Convert to dB for better dynamic range
            data = 10 * np.log10(np.clip(data, 1e-10, None))
            arrays.append(data)
            
            if transform is None:
                transform = src.transform
                crs = src.crs
    
    stacked = np.stack(arrays, axis=0)
    print(f"  Sentinel-1 SAR shape: {stacked.shape}")
    
    return stacked, transform, crs


def load_dem() -> Tuple[np.ndarray, rasterio.transform.Affine, str]:
    """
    Load Copernicus DEM elevation data.
    
    Returns:
        dem_data: (1, height, width) array
        transform: GeoTransform
        crs: Coordinate Reference System
    """
    print(f"\nLoading Copernicus DEM")
    
    path = DATASETS_DIR / "Copernicus_DEM_30m.tif"
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    
    # Expand to (1, H, W) for consistency
    dem = data[np.newaxis, :, :]
    print(f"  DEM shape: {dem.shape}")
    print(f"  DEM CRS: {crs} (needs reprojection to {TARGET_CRS})")
    
    return dem, transform, crs


def load_rainfall_data() -> Tuple[np.ndarray, List[str]]:
    """
    Load rainfall data from NetCDF file.
    
    Returns:
        rainfall_data: (time,) array of cumulative rainfall
        dates: List of date strings
    """
    print(f"\nLoading Rainfall data")
    
    path = RAINFALL_DIR / "kerala_rainfall_data.nc"
    ds = xr.open_dataset(path)
    
    rain_var = ds['rain']
    
    # Get temporal information
    time_dim = rain_var.dims[0]  # Should be 'time'
    time_values = ds[time_dim].values
    
    # Aggregate spatial dimensions to get time series
    spatial_dims = [d for d in rain_var.dims if d != time_dim]
    rainfall_aggregated = rain_var.mean(dim=spatial_dims).values
    
    # Convert to mm if needed (check units in metadata)
    # For now, assume raw values
    
    dates = [str(t)[:10] for t in time_values]  # Extract date portion
    
    print(f"  Rainfall shape: {rainfall_aggregated.shape}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    
    return rainfall_aggregated, dates


def load_soil_moisture(sample_date: str = "2024-12-15") -> Tuple[np.ndarray, rasterio.transform.Affine, str]:
    """
    Load soil moisture data from the extracted folder.
    Finds the closest available date to the sample_date.
    
    Returns:
        sm_data: (1, height, width) array
        transform: GeoTransform
        crs: Coordinate Reference System
    """
    print(f"\nLoading Soil Moisture data (closest to {sample_date})")
    
    from datetime import datetime
    
    # Updated path for extracted folder structure
    sm_folder = SOIL_MOISTURE_DIR / "Soil_Mositure" / "Soil_Mositure"
    
    if not sm_folder.exists():
        # Try alternate path
        sm_folder = SOIL_MOISTURE_DIR / "Soil_Mositure"
    
    if not sm_folder.exists():
        print(f"  ⚠ Soil moisture folder not found: {sm_folder}")
        return None, None, None
    
    # Find all TIF files
    tif_files = list(sm_folder.glob("*.tif"))
    
    if not tif_files:
        print("  ⚠ No TIF files found in soil moisture folder")
        return None, None, None
    
    print(f"  Found {len(tif_files)} soil moisture files")
    
    # Find closest date to sample_date
    target_date = datetime.strptime(sample_date, "%Y-%m-%d")
    
    available_dates = []
    for f in tif_files:
        # Parse date from filename like "SM_SMAP_I_20231229_20231231.tif"
        try:
            parts = f.stem.split('_')
            date_str = parts[-1]  # End date
            file_end_date = datetime.strptime(date_str, "%Y%m%d")
            available_dates.append((file_end_date, f))
        except Exception as e:
            continue
    
    if not available_dates:
        print("  ⚠ No soil moisture files with parseable dates")
        return None, None, None
    
    # Find closest
    available_dates.sort(key=lambda x: abs((x[0] - target_date).days))
    closest_file = available_dates[0][1]
    print(f"  Using: {closest_file.name}")
    
    # Load the file
    with rasterio.open(closest_file) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    
    # Handle NaN values
    data = np.nan_to_num(data, nan=-9999.0)
    
    sm = data[np.newaxis, :, :]
    print(f"  Soil Moisture shape: {sm.shape}")
    
    return sm, transform, crs


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def reproject_to_target(data: np.ndarray, src_crs: str, src_transform: rasterio.transform.Affine,
                        target_transform: rasterio.transform.Affine,
                        target_crs: str, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Reproject data to target CRS and resolution.
    """
    if src_crs == target_crs:
        # Still need to match spatial dimensions
        if data.shape[1:] == target_shape:
            return data
        # Resample to match target shape
        from scipy.ndimage import zoom
        zoom_y = target_shape[0] / data.shape[1]
        zoom_x = target_shape[1] / data.shape[2]
        n_bands = data.shape[0]
        resampled = np.zeros((n_bands, target_shape[0], target_shape[1]), dtype=np.float32)
        for i in range(n_bands):
            resampled[i] = zoom(data[i], (zoom_y, zoom_x), order=1)
        return resampled
    
    print(f"  Reprojecting from {src_crs} to {target_crs}")
    
    n_bands = data.shape[0]
    reprojected = np.zeros((n_bands, target_shape[0], target_shape[1]), dtype=np.float32)
    
    for i in range(n_bands):
        reproject(
            source=data[i],
            destination=reprojected[i],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
    
    return reprojected


def compute_normalization_params(data: np.ndarray, mask: np.ndarray = None) -> Dict:
    """
    Compute per-band mean and std for normalization.
    """
    if mask is not None:
        data = data[mask]
    
    mean = np.mean(data, axis=(1, 2), keepdims=True)
    std = np.std(data, axis=(1, 2), keepdims=True)
    std = np.clip(std, 1e-6, None)  # Avoid division by zero
    
    return {'mean': mean, 'std': std}


def normalize_data(data: np.ndarray, params: Dict) -> np.ndarray:
    """
    Apply z-score normalization using precomputed parameters.
    """
    mean = params['mean']
    std = params['std']
    return (data - mean) / std


# ============================================================================
# UNIFIED DATA CUBE
# ============================================================================

def create_unified_data_cube() -> Dict[str, np.ndarray]:
    """
    Create a unified data cube from all input modalities.
    
    EEGMoE Input Format: (T × Sr) × B × H × W
    Adapted Format: (Temporal × Sensor × Height × Width)
    
    Channels:
    - Sentinel-2: 12 spectral bands
    - Sentinel-1: 2 SAR dates (temporal information)
    - DEM: 1 elevation channel
    - Soil Moisture: 1 moisture channel
    - Rainfall: 1 aggregated channel (broadcast spatially)
    """
    print("\n" + "=" * 70)
    print("CREATING UNIFIED DATA CUBE")
    print("=" * 70)
    
    # Load all data
    s2_data, s2_transform, s2_crs = load_sentinel2_bands()
    s1_data, s1_transform, s1_crs = load_sentinel1_sar()
    dem_data, dem_transform, dem_crs = load_dem()
    soil_data, soil_transform, soil_crs = load_soil_moisture()
    rainfall_data, rainfall_dates = load_rainfall_data()
    
    # Use Sentinel-2 as reference (all data already in same CRS)
    target_shape = s2_data.shape[1:]  # (H, W)
    target_transform = s2_transform
    target_crs = s2_crs
    
    print(f"\nTarget shape: {target_shape}")
    print(f"Target CRS: {target_crs}")
    
    # Reproject DEM if needed
    if dem_crs != target_crs:
        dem_data = reproject_to_target(
            dem_data, dem_crs, dem_transform, target_transform, target_crs, target_shape
        )
    
    # Stack all channels
    # Format: (channels, height, width)
    channels = []
    channel_names = []
    
    # Sentinel-2 bands (12 channels)
    channels.append(s2_data)
    channel_names.extend([f"S2_{b}" for b in SENTINEL2_BANDS.keys()])
    
    # Sentinel-1 SAR (2 channels - temporal)
    channels.append(s1_data)
    channel_names.extend([f"SAR_{d}" for d in ["2024-12-11", "2024-12-16"]])
    
    # DEM (1 channel)
    channels.append(dem_data)
    channel_names.append("DEM")
    
    # Soil Moisture (1 channel)
    if soil_data is not None:
        # Reproject soil moisture to match target
        soil_data = reproject_to_target(
            soil_data, soil_crs, soil_transform, target_transform, target_crs, target_shape
        )
        channels.append(soil_data)
        channel_names.append("SoilMoisture")
    
    # Rainfall (1 channel - broadcast to spatial)
    # For now, use mean rainfall as a single channel
    rainfall_spatial = np.full((1,) + target_shape, np.mean(rainfall_data), dtype=np.float32)
    channels.append(rainfall_spatial)
    channel_names.append("Rainfall")
    
    # Stack all
    data_cube = np.concatenate(channels, axis=0)
    
    print(f"\n" + "-" * 70)
    print("UNIFIED DATA CUBE SUMMARY")
    print("-" * 70)
    print(f"Total channels: {data_cube.shape[0]}")
    print(f"Spatial dimensions: {data_cube.shape[1:]}")
    print(f"Data type: {data_cube.dtype}")
    print(f"\nChannel breakdown:")
    for i, name in enumerate(channel_names):
        print(f"  [{i:2d}] {name}")
    
    # Compute normalization parameters per channel
    print(f"\nComputing normalization parameters...")
    norm_params = {}
    for i, name in enumerate(channel_names):
        channel_data = data_cube[i]
        valid_mask = ~np.isnan(channel_data) & (channel_data != -9999.0)
        if np.any(valid_mask):
            norm_params[name] = {
                'mean': float(np.mean(channel_data[valid_mask])),
                'std': float(np.std(channel_data[valid_mask])),
                'min': float(np.min(channel_data[valid_mask])),
                'max': float(np.max(channel_data[valid_mask]))
            }
    
    print("\nNormalization parameters:")
    for name, params in norm_params.items():
        print(f"  {name}: mean={params['mean']:.4f}, std={params['std']:.4f}")
    
    return {
        'data_cube': data_cube,
        'channel_names': channel_names,
        'transform': target_transform,
        'crs': str(target_crs),
        'norm_params': norm_params,
        'shape': data_cube.shape
    }


# ============================================================================
# PYTORCH DATASET CLASS
# ============================================================================

class LandslideDataset(Dataset):
    """
    PyTorch Dataset for landslide prediction.
    
    Adapts EEGMoE's approach:
    - Creates patches from the unified data cube
    - Returns (patch, label) pairs for training
    """
    
    def __init__(self, data_cube: np.ndarray, patch_size: int = 64, 
                 stride: int = 32, labels: Optional[np.ndarray] = None):
        """
        Args:
            data_cube: (channels, height, width) array
            patch_size: Size of square patches to extract
            stride: Stride for patch extraction
            labels: Optional (height, width) binary labels (1=landslide, 0=no landslide)
        """
        self.data_cube = data_cube
        self.patch_size = patch_size
        self.stride = stride
        self.labels = labels
        
        # Generate patch indices
        self.patch_indices = []
        C, H, W = data_cube.shape
        
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                self.patch_indices.append((i, j))
        
        print(f"Created dataset with {len(self.patch_indices)} patches")
    
    def __len__(self):
        return len(self.patch_indices)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, Optional[int]]:
        i, j = self.patch_indices[idx]
        
        # Extract patch: (C, patch_size, patch_size)
        patch = self.data_cube[:, i:i+self.patch_size, j:j+self.patch_size]
        patch = torch.from_numpy(patch).float()
        
        # Get label if available
        label = None
        if self.labels is not None:
            label_patch = self.labels[i:i+self.patch_size, j:j+self.patch_size]
            # Binary label: 1 if any landslide pixel in patch
            label = 1 if np.any(label_patch > 0) else 0
        
        return patch, label


def create_data_loaders(data_cube: np.ndarray, labels: np.ndarray = None,
                        patch_size: int = 64, batch_size: int = 32,
                        train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    """
    # Create dataset
    dataset = LandslideDataset(data_cube, patch_size=patch_size, labels=labels)
    
    # Split into train/val
    n_samples = len(dataset)
    n_train = int(n_samples * train_ratio)
    n_val = n_samples - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nData loaders created:")
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {n_val}")
    
    return train_loader, val_loader


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 1: DATA LOADING PIPELINE")
    print("Team Member: DATA")
    print("Contest: Replicate-2026")
    print("=" * 70)
    
    # Create unified data cube
    cube_info = create_unified_data_cube()
    
    # Save cube info for Phase 2
    cube_info_path = BASE_DIR / "data_cube_info.txt"
    with open(cube_info_path, 'w') as f:
        f.write("DATA CUBE INFORMATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Shape: {cube_info['shape']}\n")
        f.write(f"Channels: {cube_info['data_cube'].shape[0]}\n")
        f.write(f"Spatial: {cube_info['data_cube'].shape[1:]}\n")
        f.write(f"CRS: {cube_info['crs']}\n\n")
        f.write("Channel Names:\n")
        for i, name in enumerate(cube_info['channel_names']):
            f.write(f"  [{i:2d}] {name}\n")
        f.write("\nNormalization Parameters:\n")
        for name, params in cube_info['norm_params'].items():
            f.write(f"  {name}: mean={params['mean']:.4f}, std={params['std']:.4f}\n")
    
    print(f"\nData cube info saved to: {cube_info_path}")
    
    # Test data loader creation
    print("\n" + "=" * 70)
    print("TESTING DATA LOADER CREATION")
    print("=" * 70)
    
    # Create dummy labels for testing (all zeros)
    dummy_labels = np.zeros(cube_info['data_cube'].shape[1:], dtype=np.uint8)
    
    train_loader, val_loader = create_data_loaders(
        cube_info['data_cube'],
        labels=dummy_labels,
        patch_size=64,
        batch_size=16
    )
    
    # Test batch retrieval
    print("\nTesting batch retrieval...")
    batch_patches, batch_labels = next(iter(train_loader))
    print(f"  Batch shape: {batch_patches.shape}")
    print(f"  Labels shape: {batch_labels.shape}")
    print(f"  Batch dtype: {batch_patches.dtype}")
    
    print("\n" + "=" * 70)
    print("TASK 1.3 COMPLETE: DATA LOADING PIPELINE READY")
    print("=" * 70)
    print("\nDeliverables:")
    print("  ✓ Unified data cube created")
    print("  ✓ Normalization parameters computed")
    print("  ✓ PyTorch Dataset class implemented")
    print("  ✓ DataLoaders created and tested")
    print(f"\nOutput tensor shape: {batch_patches.shape}")
    print(f"  Format: (Batch, Channels, Height, Width)")
    print(f"  Compatible with EEGMoE architecture adaptation")
