#!/usr/bin/env python3
r"""
2_extract_and_analyze.py (v1.0.0)

================================================================================
WHAT IS THIS?
================================================================================
This is Script 2 in the BrainGlobe pipeline. It extracts image data from Imaris
.ims files, analyzes the data to find optimal crop boundaries and identify
channel roles, then creates both full and cropped TIFF stacks.

Think of it as: "Convert my microscope data and figure out where to crop"

Run this AFTER Script 1 (organize_pipeline.py).
Run this BEFORE Script 3 (register_to_atlas.py).

================================================================================
WHAT IT DOES
================================================================================
1. Finds organized IMS files (from Script 1)
2. Extracts all channels to TIFF slices (full extraction)
3. Analyzes tissue area across Z to detect brain→cord transition
4. Analyzes channel textures to identify signal vs background channels
5. Creates cropped copies optimized for registration
6. Generates QC plots for visual verification
7. Saves comprehensive metadata for downstream scripts

================================================================================
HOW TO RUN
================================================================================
Open Anaconda Prompt, then:

    conda activate brainglobe-env
    cd Y:\2_Connectome\3_Nuclei_Detection\util_Scripts
    python 2_extract_and_analyze.py

Options:
    python 2_extract_and_analyze.py              # Process all pending
    python 2_extract_and_analyze.py --inspect    # Dry run - show status only
    python 2_extract_and_analyze.py --no-crop    # Skip crop detection
    python 2_extract_and_analyze.py --batch      # Non-interactive mode

================================================================================
OUTPUTS
================================================================================
For each brain, creates:

    349_CNT_01_02_1p625x_z4/
    ├── 1_Extracted_Full/
    │   ├── ch0/
    │   │   ├── Z0000.tif ... Z0XXX.tif
    │   ├── ch1/
    │   ├── metadata.json              ← Full extraction metadata
    │   └── QC_area_profile.png        ← Z vs area plot showing crop detection
    │
    └── 2_Cropped_For_Registration/
        ├── ch0/
        │   ├── Z0000.tif ... Z0XXX.tif  ← Renumbered from crop point
        ├── ch1/
        └── metadata.json              ← Includes crop info + channel roles

================================================================================
CROP DETECTION
================================================================================
The script automatically detects where brain tissue ends and spinal cord begins
by analyzing the cross-sectional area of tissue across Z-slices:

    Brain (large, variable area)
         ↓
    ████████████████████
   ██████████████████████
  ████████████████████████
    ██████████████████
      ██████████████
        ████████
          ████          ← Brainstem narrows
           ██           ← Spinal cord (small, constant)
           ██
           
The algorithm finds the Z-position where area drops significantly, indicating
the transition from brain to cord.

================================================================================
CHANNEL IDENTIFICATION  
================================================================================
The script analyzes each channel to determine its role:
- Signal channel (c-Fos, GFP, etc.): Sparse bright spots, high local variance
- Background channel (autofluorescence): Smoother, shows tissue structure

This info is saved to metadata so Script 3 knows which channel to register.

================================================================================
SETTINGS (edit in script if needed)
================================================================================
    CAMERA_PIXEL_SIZE = 6.5      # Andor Neo/Zyla pixel size in microns
    DEFAULT_ORIENTATION = "iar"  # inferior-anterior-right
    CROP_AREA_THRESHOLD = 0.3    # Crop where area drops below 30% of max

================================================================================
REQUIREMENTS
================================================================================
    pip install imaris-ims-file-reader tifffile numpy h5py matplotlib scipy
"""

# =============================================================================
# CHECK REQUIREMENTS
# =============================================================================
import sys

def _check_requirements():
    """Check if required packages are available."""
    missing = []
    for pkg in ['numpy', 'tifffile', 'h5py', 'matplotlib', 'scipy']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print("=" * 60)
        print("ERROR: Missing required packages:", ", ".join(missing))
        print()
        print("Run this script from Anaconda Prompt:")
        print("    conda activate brainglobe-env")
        print("    python 2_extract_and_analyze.py")
        print()
        print("Or install missing packages:")
        print(f"    pip install {' '.join(missing)}")
        print("=" * 60)
        input("\nPress Enter to close...")
        sys.exit(1)

_check_requirements()

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import io
import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import ndimage

# =============================================================================
# VERSION
# =============================================================================
SCRIPT_VERSION = "1.2.0"

# =============================================================================
# PROGRESS FEEDBACK HELPERS
# =============================================================================

def timestamp():
    """Get current time as formatted string."""
    return datetime.now().strftime("%H:%M:%S")

def format_duration(seconds):
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

def print_status(message, end='\n'):
    """Print a timestamped status message."""
    print(f"    [{timestamp()}] {message}", end=end)
    sys.stdout.flush()

def print_progress(current, total, prefix="", width=30):
    """Print a progress bar."""
    pct = current / total
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r    [{timestamp()}] {prefix}|{bar}| {current}/{total} ({pct*100:.0f}%)", end='')
    sys.stdout.flush()
    if current >= total:
        print()  # Newline when complete

# =============================================================================
# DEFAULT PATHS AND SETTINGS
# =============================================================================
DEFAULT_BRAINGLOBE_ROOT = Path(r"Y:\2_Connectome\3_Nuclei_Detection\1_Brains")

# Camera pixel size in microns (Andor Neo/Zyla sCMOS)
CAMERA_PIXEL_SIZE = 6.5

# Default orientation for brainreg (inferior-anterior-right)
DEFAULT_ORIENTATION = "iar"

# Crop detection settings
CROP_AREA_THRESHOLD = 0.3      # Crop where area drops below this fraction of max
CROP_GRADIENT_THRESHOLD = 0.1  # Also consider large area drops
MIN_BRAIN_SLICES = 100         # Minimum slices to keep (don't over-crop)

# Channel identification settings
SIGNAL_SPARSITY_THRESHOLD = 0.05  # Signal channel has fewer bright pixels

# =============================================================================
# PIPELINE FOLDER NAMES (must match Script 1)
# =============================================================================
FOLDER_RAW_IMS = "0_Raw_IMS"
FOLDER_EXTRACTED_FULL = "1_Extracted_Full"
FOLDER_CROPPED = "2_Cropped_For_Registration"


# =============================================================================
# THUMBS.DB HANDLING
# =============================================================================

def annihilate_thumbs_db(path, recursive=True):
    """Destroy all thumbs.db files."""
    path = Path(path)
    destroyed = 0
    
    def try_destroy(thumbs):
        nonlocal destroyed
        try:
            if thumbs.stat().st_size == 0:
                return
            if sys.platform == 'win32':
                import subprocess
                subprocess.run(['attrib', '-h', '-s', '-r', str(thumbs)], 
                               capture_output=True, check=False)
            thumbs.unlink()
            destroyed += 1
        except:
            pass
    
    if recursive:
        for pattern in ["Thumbs.db", "thumbs.db", "THUMBS.DB"]:
            for thumbs in path.rglob(pattern):
                try_destroy(thumbs)
    else:
        for name in ["Thumbs.db", "thumbs.db", "THUMBS.DB"]:
            thumbs = path / name
            if thumbs.exists():
                try_destroy(thumbs)
    
    return destroyed


def create_thumbs_decoy(folder):
    """Create a read-only decoy Thumbs.db to prevent Windows from creating one."""
    folder = Path(folder)
    decoy = folder / "Thumbs.db"
    if decoy.exists():
        return False
    try:
        decoy.write_bytes(b'')
        if sys.platform == 'win32':
            import subprocess
            subprocess.run(['attrib', '+h', '+r', str(decoy)], 
                           capture_output=True, check=False)
        return True
    except:
        return False


# =============================================================================
# FILENAME PARSING
# =============================================================================

def decimals_to_p(s):
    """Convert decimals to 'p' in a string."""
    return s.replace('.', 'p')


def parse_filename(filename):
    """Parse an IMS filename to extract components."""
    stem = Path(filename).stem
    pattern = r'^(\d+)_([A-Za-z]+)_(\d+)_(\d+)_(\d+\.?\d*)x_z(\d+\.?\d*)$'
    match = re.match(pattern, stem)
    
    if not match:
        return None
    
    return {
        'number': match.group(1),
        'project': match.group(2),
        'cohort': match.group(3),
        'animal': match.group(4),
        'magnification': float(match.group(5)),
        'z_step': float(match.group(6)),
        'stem': stem,
        'pipeline_folder': decimals_to_p(stem),
    }


def calculate_voxel_size_xy(magnification):
    """Calculate XY voxel size from magnification."""
    return CAMERA_PIXEL_SIZE / magnification


# =============================================================================
# DISCOVERY
# =============================================================================

def find_organized_pipelines(root_path):
    """
    Find all organized pipeline folders ready for extraction.
    Returns list of (ims_path, pipeline_folder, mouse_folder) tuples.
    """
    root_path = Path(root_path)
    pipelines = []
    
    for mouse_dir in root_path.iterdir():
        if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
            continue
        if any(skip in mouse_dir.name.lower() for skip in ['script', 'backup', 'archive']):
            continue
        
        for pipeline_dir in mouse_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue
            
            raw_folder = pipeline_dir / FOLDER_RAW_IMS
            if not raw_folder.exists():
                continue
            
            # Look for IMS file
            ims_files = list(raw_folder.glob('*.ims'))
            if len(ims_files) == 1:
                pipelines.append((ims_files[0], pipeline_dir, mouse_dir))
            elif len(ims_files) > 1:
                print(f"  Warning: Multiple IMS files in {raw_folder}, skipping")
    
    return pipelines


def check_extraction_status(pipeline_folder):
    """
    Check if extraction has been done for a pipeline.
    
    Returns:
        (status, reason)
        
        Status:
        - "needs_extraction": No extraction done yet
        - "needs_cropping": Full extracted but no crop
        - "complete": Both full and cropped exist
    """
    pipeline_folder = Path(pipeline_folder)
    
    full_folder = pipeline_folder / FOLDER_EXTRACTED_FULL
    crop_folder = pipeline_folder / FOLDER_CROPPED
    
    # Check for full extraction
    full_ch0 = full_folder / "ch0"
    has_full = (full_ch0.exists() and 
                len(list(full_ch0.glob('Z*.tif'))) > 0)
    
    # Check for cropped
    crop_ch0 = crop_folder / "ch0"
    has_crop = (crop_ch0.exists() and 
                len(list(crop_ch0.glob('Z*.tif'))) > 0)
    
    # Check for metadata
    has_full_meta = (full_folder / "metadata.json").exists()
    has_crop_meta = (crop_folder / "metadata.json").exists()
    
    if has_full and has_crop and has_crop_meta:
        return "complete", "extraction complete"
    elif has_full and has_full_meta:
        return "needs_cropping", "full extracted, needs cropping"
    else:
        return "needs_extraction", "not extracted"


# =============================================================================
# IMS FILE READING
# =============================================================================

def get_ims_info(filepath):
    """Extract metadata from IMS file."""
    import h5py
    
    result = {
        'size_x': None,
        'size_y': None,
        'size_z': None,
        'channels': [],
    }
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Get dimensions
            if 'DataSetInfo/Image' in f:
                img_info = f['DataSetInfo/Image']
                for dim in ['X', 'Y', 'Z']:
                    for attr_name in [f'{dim}', f'Size{dim}']:
                        if attr_name in img_info.attrs:
                            val = img_info.attrs[attr_name]
                            if hasattr(val, '__iter__'):
                                val = val[0]
                            if isinstance(val, bytes):
                                val = val.decode()
                            try:
                                result[f'size_{dim.lower()}'] = int(float(val))
                                break
                            except:
                                pass
            
            # Get channel info
            i = 0
            while True:
                channel_path = f'DataSetInfo/Channel {i}'
                if channel_path in f:
                    channel_info = f[channel_path]
                    name = f"Channel{i}"
                    try:
                        if 'Name' in channel_info.attrs:
                            name_val = channel_info.attrs['Name'][0]
                            if isinstance(name_val, bytes):
                                name_val = name_val.decode()
                            name = str(name_val)
                    except:
                        pass
                    result['channels'].append({'index': i, 'name': name})
                    i += 1
                else:
                    break
    except Exception as e:
        print(f"    Warning: Could not read IMS metadata: {e}")
    
    return result


# =============================================================================
# CROP DETECTION
# =============================================================================

def calculate_tissue_area(slice_2d, threshold_percentile=10):
    """
    Calculate the area of tissue in a 2D slice.
    Uses simple thresholding to find tissue pixels.
    """
    # Use percentile-based threshold to handle varying intensities
    threshold = np.percentile(slice_2d[slice_2d > 0], threshold_percentile) if np.any(slice_2d > 0) else 0
    tissue_mask = slice_2d > threshold
    return np.sum(tissue_mask)


def detect_crop_boundary(volume, orientation="iar"):
    """
    Detect where to crop based on tissue area analysis.
    
    For IAR orientation (inferior-anterior-right):
    - Z=0 is inferior (bottom of brain / top of cord)
    - We want to find where brain ends and cord begins
    
    Returns:
        (crop_start, crop_end, area_profile, analysis_info)
    """
    num_slices = volume.shape[0]
    
    # Calculate area for each Z slice with progress
    print_status(f"Computing tissue area for {num_slices} slices...")
    areas = np.zeros(num_slices)
    for z in range(num_slices):
        areas[z] = calculate_tissue_area(volume[z, :, :])
        if (z + 1) % 200 == 0 or z == num_slices - 1:
            print_progress(z + 1, num_slices, prefix="Analyzing ")
    
    print_status("Smoothing and detecting boundaries...")
    
    # Smooth the profile to reduce noise
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(areas, sigma=5)
    
    # Find max area (should be somewhere in the middle of brain)
    max_area = np.max(smoothed)
    max_idx = np.argmax(smoothed)
    
    # Threshold: where does area drop below threshold of max?
    threshold = max_area * CROP_AREA_THRESHOLD
    
    # Search from the inferior end (Z=0) for where brain starts
    # (area rises above threshold)
    crop_start = 0
    for z in range(num_slices):
        if smoothed[z] > threshold:
            # Go back a bit for safety margin
            crop_start = max(0, z - 10)
            break
    
    # Search from superior end for where brain ends
    crop_end = num_slices - 1
    for z in range(num_slices - 1, -1, -1):
        if smoothed[z] > threshold:
            crop_end = min(num_slices - 1, z + 10)
            break
    
    # Ensure we keep minimum slices
    if (crop_end - crop_start) < MIN_BRAIN_SLICES:
        # Center the minimum around max area
        half = MIN_BRAIN_SLICES // 2
        crop_start = max(0, max_idx - half)
        crop_end = min(num_slices - 1, max_idx + half)
    
    analysis_info = {
        'max_area': int(max_area),
        'max_area_z': int(max_idx),
        'threshold_area': int(threshold),
        'original_slices': num_slices,
        'cropped_slices': crop_end - crop_start + 1,
        'removed_inferior': crop_start,
        'removed_superior': num_slices - 1 - crop_end,
    }
    
    return crop_start, crop_end, areas, analysis_info


def generate_crop_qc_plot(areas, crop_start, crop_end, output_path):
    """Generate a QC plot showing the area profile and crop boundaries."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    z_indices = np.arange(len(areas))
    
    # Plot area profile
    ax.plot(z_indices, areas, 'b-', linewidth=1, alpha=0.5, label='Raw')
    
    # Smoothed version
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(areas, sigma=5)
    ax.plot(z_indices, smoothed, 'b-', linewidth=2, label='Smoothed')
    
    # Crop boundaries
    ax.axvline(crop_start, color='r', linestyle='--', linewidth=2, label=f'Crop start (Z={crop_start})')
    ax.axvline(crop_end, color='r', linestyle='--', linewidth=2, label=f'Crop end (Z={crop_end})')
    
    # Shade removed regions
    ax.axvspan(0, crop_start, alpha=0.2, color='red', label='Removed (cord)')
    ax.axvspan(crop_end, len(areas)-1, alpha=0.2, color='orange', label='Removed (superior)')
    
    # Threshold line
    threshold = np.max(smoothed) * CROP_AREA_THRESHOLD
    ax.axhline(threshold, color='gray', linestyle=':', label=f'Threshold ({CROP_AREA_THRESHOLD*100:.0f}% of max)')
    
    ax.set_xlabel('Z slice', fontsize=12)
    ax.set_ylabel('Tissue area (pixels)', fontsize=12)
    ax.set_title('Crop Detection: Tissue Area vs Z Position', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add text annotation
    kept = crop_end - crop_start + 1
    total = len(areas)
    ax.text(0.02, 0.98, f'Keeping {kept}/{total} slices ({100*kept/total:.1f}%)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# CHANNEL IDENTIFICATION
# =============================================================================

def analyze_channel_characteristics(volume, sample_slices=10):
    """
    Analyze a channel to determine if it's signal or background.
    
    Returns dict with metrics:
    - sparsity: fraction of very bright pixels (signal channels are sparse)
    - local_variance: measure of texture (signal channels are speckly)
    - histogram_bimodality: signal channels tend to be bimodal
    """
    num_slices = volume.shape[0]
    
    # Sample evenly spaced slices
    slice_indices = np.linspace(0, num_slices-1, sample_slices, dtype=int)
    
    sparsities = []
    local_variances = []
    
    for z in slice_indices:
        slice_2d = volume[z, :, :].astype(float)
        
        # Skip mostly empty slices
        if np.mean(slice_2d) < 1:
            continue
        
        # Sparsity: fraction of pixels > 90th percentile
        p90 = np.percentile(slice_2d, 90)
        p99 = np.percentile(slice_2d, 99)
        if p99 > 0:
            bright_fraction = np.sum(slice_2d > p90) / slice_2d.size
            very_bright_fraction = np.sum(slice_2d > p99) / slice_2d.size
            sparsities.append(very_bright_fraction / (bright_fraction + 1e-10))
        
        # Local variance (using small kernel)
        local_mean = ndimage.uniform_filter(slice_2d, size=5)
        local_sqr_mean = ndimage.uniform_filter(slice_2d**2, size=5)
        local_var = local_sqr_mean - local_mean**2
        local_variances.append(np.mean(local_var) / (np.mean(slice_2d)**2 + 1e-10))
    
    return {
        'sparsity': np.mean(sparsities) if sparsities else 0,
        'local_variance': np.mean(local_variances) if local_variances else 0,
    }


def identify_channel_roles(channel_metrics):
    """
    Given metrics for each channel, identify which is signal vs background.
    
    Returns dict mapping channel index to role.
    """
    if len(channel_metrics) == 1:
        return {0: 'signal'}  # Only one channel, assume it's what we want
    
    if len(channel_metrics) == 2:
        # Compare the two channels
        ch0_score = channel_metrics[0]['sparsity'] + channel_metrics[0]['local_variance']
        ch1_score = channel_metrics[1]['sparsity'] + channel_metrics[1]['local_variance']
        
        # Higher score = more likely signal (sparse + speckly)
        if ch0_score > ch1_score:
            return {0: 'signal', 1: 'background'}
        else:
            return {0: 'background', 1: 'signal'}
    
    # More than 2 channels - rank by combined score
    scores = []
    for i, metrics in enumerate(channel_metrics):
        score = metrics['sparsity'] + metrics['local_variance']
        scores.append((i, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    roles = {}
    roles[scores[0][0]] = 'signal'
    for i, _ in scores[1:]:
        roles[i] = 'background'
    
    return roles


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def extract_and_analyze(ims_path, pipeline_folder, skip_crop=False):
    """
    Main extraction function.
    
    Intelligently handles existing data:
    - If full TIFFs exist: Load from TIFFs (fast), analyze, create crops
    - If no TIFFs: Extract from IMS (slow), analyze, create crops
    """
    import tifffile
    
    ims_path = Path(ims_path)
    pipeline_folder = Path(pipeline_folder)
    
    parsed = parse_filename(ims_path.name)
    
    # Setup output folders
    full_folder = pipeline_folder / FOLDER_EXTRACTED_FULL
    crop_folder = pipeline_folder / FOLDER_CROPPED
    
    full_folder.mkdir(parents=True, exist_ok=True)
    crop_folder.mkdir(parents=True, exist_ok=True)
    
    # Get file info
    file_size_gb = ims_path.stat().st_size / (1024**3)
    ims_info = get_ims_info(str(ims_path))
    num_channels = len(ims_info['channels'])
    
    # Calculate voxel sizes
    voxel_xy = calculate_voxel_size_xy(parsed['magnification'])
    voxel_z = parsed['z_step']
    
    print(f"    File: {file_size_gb:.1f} GB")
    print(f"    Dimensions: {ims_info['size_x']}x{ims_info['size_y']}x{ims_info['size_z']}")
    print(f"    Voxels (µm): X={voxel_xy:.2f}, Y={voxel_xy:.2f}, Z={voxel_z:.2f}")
    print(f"    Channels: {num_channels}")
    sys.stdout.flush()
    
    # =========================================================================
    # CHECK IF FULL EXTRACTION ALREADY EXISTS
    # =========================================================================
    full_exists = all(
        (full_folder / f"ch{ch_idx}").exists() and 
        len(list((full_folder / f"ch{ch_idx}").glob('Z*.tif'))) > 0
        for ch_idx in range(num_channels)
    )
    
    volumes = []
    channel_metrics = []
    total_start = time.time()
    num_slices = 0
    
    if full_exists:
        # =====================================================================
        # LOAD FROM EXISTING TIFFS (FAST PATH)
        # =====================================================================
        print(f"\n    --- Loading Existing Extraction (started {timestamp()}) ---")
        print_status("Full extraction already exists, loading from TIFFs...")
        
        for ch_idx in range(num_channels):
            ch_name = ims_info['channels'][ch_idx]['name'] if ch_idx < len(ims_info['channels']) else f"Channel{ch_idx}"
            print(f"\n    Channel {ch_idx}/{num_channels-1} ({ch_name}):")
            
            ch_folder = full_folder / f"ch{ch_idx}"
            tif_files = sorted(ch_folder.glob('Z*.tif'))
            num_slices = len(tif_files)
            
            # Load volume from TIFFs
            print_status(f"Loading {num_slices} existing TIFF slices...")
            start_time = time.time()
            
            # Read first slice to get dimensions
            first_slice = tifffile.imread(str(tif_files[0]))
            volume = np.zeros((num_slices, first_slice.shape[0], first_slice.shape[1]), 
                            dtype=first_slice.dtype)
            volume[0] = first_slice
            
            for z, tif_path in enumerate(tif_files[1:], 1):
                volume[z] = tifffile.imread(str(tif_path))
                if (z + 1) % 200 == 0 or z == num_slices - 1:
                    print_progress(z + 1, num_slices, prefix="Loading ")
            
            load_time = time.time() - start_time
            print_status(f"Loaded in {format_duration(load_time)} | Shape: {volume.shape}")
            
            # Analyze channel
            print_status("Analyzing channel characteristics...", end='')
            analysis_start = time.time()
            metrics = analyze_channel_characteristics(volume)
            channel_metrics.append(metrics)
            print(f" done ({format_duration(time.time() - analysis_start)})")
            print_status(f"Sparsity={metrics['sparsity']:.4f}, LocalVariance={metrics['local_variance']:.4f}")
            
            volumes.append(volume)
        
        load_time = time.time() - total_start
        print(f"\n    Loaded existing extraction in {format_duration(load_time)}")
        
    else:
        # =====================================================================
        # EXTRACT FROM IMS (SLOW PATH)
        # =====================================================================
        from imaris_ims_file_reader.ims import ims
        
        # Open IMS file (suppress warnings)
        print_status("Opening IMS file...", end='')
        open_start = time.time()
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            ims_data = ims(str(ims_path), ResolutionLevelLock=0, squeeze_output=False)
        finally:
            sys.stderr = old_stderr
        print(f" done ({format_duration(time.time() - open_start)})")
        
        print(f"\n    --- Full Extraction (started {timestamp()}) ---")
        
        for ch_idx in range(num_channels):
            ch_name = ims_info['channels'][ch_idx]['name'] if ch_idx < len(ims_info['channels']) else f"Channel{ch_idx}"
            print(f"\n    Channel {ch_idx}/{num_channels-1} ({ch_name}):")
            
            # Create channel folder
            ch_folder = full_folder / f"ch{ch_idx}"
            ch_folder.mkdir(exist_ok=True)
            create_thumbs_decoy(ch_folder)
            
            # Load channel from IMS
            print_status(f"Loading from IMS (this may take several minutes)...", end='')
            start_time = time.time()
            
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                volume = np.array(ims_data[0, ch_idx, :, :, :])
            finally:
                sys.stderr = old_stderr
            
            volume = np.squeeze(volume)
            load_time = time.time() - start_time
            num_slices = volume.shape[0]
            print(f" done ({format_duration(load_time)})")
            print_status(f"Loaded {volume.shape[0]} x {volume.shape[1]} x {volume.shape[2]} voxels")
            
            # Save slices with progress bar
            print_status(f"Saving {num_slices} TIFF slices...")
            start_time = time.time()
            
            for z in range(num_slices):
                slice_path = ch_folder / f"Z{z:04d}.tif"
                tifffile.imwrite(str(slice_path), volume[z, :, :])
                if (z + 1) % 50 == 0 or z == num_slices - 1:
                    print_progress(z + 1, num_slices, prefix="Saving ")
            
            save_time = time.time() - start_time
            print_status(f"Saved in {format_duration(save_time)} ({num_slices/save_time:.1f} slices/sec)")
            
            # Analyze channel
            print_status("Analyzing channel characteristics...", end='')
            analysis_start = time.time()
            metrics = analyze_channel_characteristics(volume)
            channel_metrics.append(metrics)
            print(f" done ({format_duration(time.time() - analysis_start)})")
            print_status(f"Sparsity={metrics['sparsity']:.4f}, LocalVariance={metrics['local_variance']:.4f}")
            
            volumes.append(volume)
            annihilate_thumbs_db(ch_folder, recursive=False)
        
        extraction_time = time.time() - total_start
        print(f"\n    Full extraction completed in {format_duration(extraction_time)}")
    
    # =========================================================================
    # IDENTIFY CHANNEL ROLES
    # =========================================================================
    print(f"\n    --- Channel Analysis ---")
    channel_roles = identify_channel_roles(channel_metrics)
    
    for ch_idx, role in channel_roles.items():
        ch_name = ims_info['channels'][ch_idx]['name'] if ch_idx < len(ims_info['channels']) else f"Channel{ch_idx}"
        print(f"    Channel {ch_idx} ({ch_name}): {role.upper()}")
    
    # Find signal and background channel indices
    signal_channel = [k for k, v in channel_roles.items() if v == 'signal'][0]
    background_channel = [k for k, v in channel_roles.items() if v == 'background'][0] if num_channels > 1 else None
    
    # =========================================================================
    # DETECT CROP BOUNDARY
    # =========================================================================
    crop_start = 0
    crop_end = num_slices - 1
    crop_analysis = None
    
    if not skip_crop:
        print(f"\n    --- Crop Detection (started {timestamp()}) ---")
        
        # Use background channel for crop detection (clearer tissue boundaries)
        analysis_channel = background_channel if background_channel is not None else signal_channel
        print_status(f"Analyzing channel {analysis_channel} for tissue boundaries...")
        
        crop_detect_start = time.time()
        crop_start, crop_end, areas, crop_analysis = detect_crop_boundary(volumes[analysis_channel])
        crop_detect_time = time.time() - crop_detect_start
        
        print_status(f"Analysis complete ({format_duration(crop_detect_time)})")
        print_status(f"Detected crop: Z={crop_start} to Z={crop_end}")
        print_status(f"Keeping {crop_end - crop_start + 1}/{num_slices} slices " +
              f"({100*(crop_end-crop_start+1)/num_slices:.1f}%)")
        
        # Generate QC plot
        print_status("Generating QC plot...", end='')
        qc_path = full_folder / "QC_area_profile.png"
        generate_crop_qc_plot(areas, crop_start, crop_end, qc_path)
        print(" done")
        print_status(f"Saved: QC_area_profile.png")
    
    # =========================================================================
    # CREATE CROPPED COPIES
    # =========================================================================
    print(f"\n    --- Creating Cropped Copies (started {timestamp()}) ---")
    
    cropped_slices = crop_end - crop_start + 1
    crop_copy_start = time.time()
    
    for ch_idx in range(num_channels):
        ch_name = ims_info['channels'][ch_idx]['name'] if ch_idx < len(ims_info['channels']) else f"Channel{ch_idx}"
        print_status(f"Copying channel {ch_idx} ({ch_name}): {cropped_slices} slices...")
        
        ch_folder = crop_folder / f"ch{ch_idx}"
        ch_folder.mkdir(exist_ok=True)
        create_thumbs_decoy(ch_folder)
        
        # Copy and renumber slices with progress
        for new_z, old_z in enumerate(range(crop_start, crop_end + 1)):
            src = full_folder / f"ch{ch_idx}" / f"Z{old_z:04d}.tif"
            dst = ch_folder / f"Z{new_z:04d}.tif"
            shutil.copy2(str(src), str(dst))
            if (new_z + 1) % 100 == 0 or new_z == cropped_slices - 1:
                print_progress(new_z + 1, cropped_slices, prefix="Copying ")
        
        annihilate_thumbs_db(ch_folder, recursive=False)
    
    crop_copy_time = time.time() - crop_copy_start
    print_status(f"Cropped copies complete ({format_duration(crop_copy_time)})")
    
    # =========================================================================
    # SAVE METADATA
    # =========================================================================
    print(f"\n    --- Saving Metadata ---")
    print_status("Writing metadata files...")
    
    total_time = time.time() - total_start
    
    # Full extraction metadata
    full_metadata = {
        'script_version': SCRIPT_VERSION,
        'extraction_type': 'full',
        'processed_date': datetime.now().isoformat(),
        'source_file': ims_path.name,
        'pipeline_folder': parsed['pipeline_folder'],
        'magnification': parsed['magnification'],
        'z_step': parsed['z_step'],
        'dimensions': {
            'x': int(ims_info['size_x']) if ims_info['size_x'] else None,
            'y': int(ims_info['size_y']) if ims_info['size_y'] else None,
            'z': num_slices,
        },
        'voxel_size_um': {
            'x': round(voxel_xy, 4),
            'y': round(voxel_xy, 4),
            'z': round(voxel_z, 4),
        },
        'channels': {
            'count': num_channels,
            'names': {ch['index']: ch['name'] for ch in ims_info['channels']},
            'roles': channel_roles,
            'signal_channel': signal_channel,
            'background_channel': background_channel,
            'metrics': {i: m for i, m in enumerate(channel_metrics)},
        },
        'crop_analysis': crop_analysis,
        'orientation': DEFAULT_ORIENTATION,
    }
    
    with open(full_folder / "metadata.json", 'w') as f:
        json.dump(full_metadata, f, indent=2)
    
    # Cropped metadata
    reg_ch = background_channel if background_channel is not None else signal_channel
    crop_metadata = {
        'script_version': SCRIPT_VERSION,
        'extraction_type': 'cropped',
        'processed_date': datetime.now().isoformat(),
        'source_file': ims_path.name,
        'pipeline_folder': parsed['pipeline_folder'],
        'magnification': parsed['magnification'],
        'z_step': parsed['z_step'],
        'dimensions': {
            'x': int(ims_info['size_x']) if ims_info['size_x'] else None,
            'y': int(ims_info['size_y']) if ims_info['size_y'] else None,
            'z': cropped_slices,
        },
        'voxel_size_um': {
            'x': round(voxel_xy, 4),
            'y': round(voxel_xy, 4),
            'z': round(voxel_z, 4),
        },
        'channels': {
            'count': num_channels,
            'names': {ch['index']: ch['name'] for ch in ims_info['channels']},
            'roles': channel_roles,
            'signal_channel': signal_channel,
            'background_channel': background_channel,
        },
        'crop': {
            'original_slices': num_slices,
            'crop_start': crop_start,
            'crop_end': crop_end,
            'cropped_slices': cropped_slices,
            'z_offset': crop_start,  # Add this to cropped Z to get original Z
        },
        'orientation': DEFAULT_ORIENTATION,
        'brainreg_command': (
            f"brainreg \"{crop_folder / f'ch{reg_ch}'}\" "
            f"\"{{output}}\" "
            f"-v {voxel_z:.2f} {voxel_xy:.2f} {voxel_xy:.2f} "
            f"--orientation {DEFAULT_ORIENTATION} --atlas allen_mouse_10um"
        ),
    }
    
    with open(crop_folder / "metadata.json", 'w') as f:
        json.dump(crop_metadata, f, indent=2)
    
    print_status("Metadata saved to both folders")
    
    # Final summary
    print(f"\n    {'='*50}")
    print(f"    EXTRACTION COMPLETE")
    print(f"    {'='*50}")
    print(f"    Total time: {format_duration(total_time)}")
    print(f"    Full extraction: {num_slices} slices × {num_channels} channels")
    print(f"    Cropped for registration: {cropped_slices} slices")
    print(f"    Signal channel: {signal_channel}")
    print(f"    Background channel: {background_channel}")
    print(f"    {'='*50}")
    
    # Cleanup
    for v in volumes:
        del v
    
    annihilate_thumbs_db(pipeline_folder)
    
    return full_metadata, crop_metadata


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract IMS files and analyze for registration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Version: {SCRIPT_VERSION}

This script extracts IMS files to TIFF, detects crop boundaries,
and identifies channel roles. Run AFTER Script 1.

Examples:
  python 2_extract_and_analyze.py
  python 2_extract_and_analyze.py --inspect
  python 2_extract_and_analyze.py --no-crop
        """
    )
    
    parser.add_argument('path', nargs='?', default=None,
                        help=f'Path to scan (default: {DEFAULT_BRAINGLOBE_ROOT})')
    parser.add_argument('--inspect', '-i', action='store_true',
                        help='Dry run - show what would be processed')
    parser.add_argument('--no-crop', action='store_true',
                        help='Skip automatic crop detection')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Non-interactive mode - process all pending')
    
    args = parser.parse_args()
    
    root_path = Path(args.path) if args.path else DEFAULT_BRAINGLOBE_ROOT
    
    if not root_path.exists():
        print(f"Error: Path not found: {root_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("BrainGlobe IMS Extractor & Analyzer")
    print(f"Version: {SCRIPT_VERSION}")
    print("=" * 60)
    
    # Find organized pipelines
    print(f"\nScanning: {root_path}")
    pipelines = find_organized_pipelines(root_path)
    
    if not pipelines:
        print("\nNo organized pipelines found.")
        print("Run Script 1 (organize_pipeline.py) first!")
        return
    
    # Check status of each
    needs_work = []
    complete = []
    
    for ims_path, pipeline_folder, mouse_folder in pipelines:
        status, reason = check_extraction_status(pipeline_folder)
        if status in ["needs_extraction", "needs_cropping"]:
            needs_work.append((ims_path, pipeline_folder, mouse_folder, status, reason))
        else:
            complete.append((ims_path, pipeline_folder, mouse_folder))
    
    # Report
    print(f"\nFound {len(pipelines)} organized pipeline(s):\n")
    
    for ims_path, pipeline_folder, mouse_folder in complete:
        print(f"  ✓ {mouse_folder.name}/{pipeline_folder.name}")
    
    for ims_path, pipeline_folder, mouse_folder, status, reason in needs_work:
        print(f"  ○ {mouse_folder.name}/{pipeline_folder.name} - {reason}")
    
    if not needs_work:
        print("\nAll extractions complete! Ready for Script 3.")
        return
    
    if args.inspect:
        print(f"\n{len(needs_work)} pipeline(s) would be processed.")
        return
    
    # Confirm
    if not args.batch:
        response = input(f"\nExtract {len(needs_work)} pipeline(s)? [Enter to continue, 'q' to quit]: ").strip()
        if response.lower() == 'q':
            print("Cancelled.")
            return
    
    # Process
    print("\n" + "=" * 60)
    print("Processing...")
    print("=" * 60)
    
    success = 0
    failed = 0
    
    for i, (ims_path, pipeline_folder, mouse_folder, status, reason) in enumerate(needs_work):
        print(f"\n[{i+1}/{len(needs_work)}] {mouse_folder.name}/{pipeline_folder.name}")
        print("-" * 50)
        
        try:
            extract_and_analyze(ims_path, pipeline_folder, skip_crop=args.no_crop)
            success += 1
            print(f"\n    ✓ Complete")
        except Exception as e:
            failed += 1
            print(f"\n    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Complete: {success} succeeded, {failed} failed")
    print("=" * 60)
    
    if success > 0:
        print("\n✓ Ready for Script 3 (register_to_atlas.py)!")


if __name__ == '__main__':
    main()
    
    if len(sys.argv) == 1:
        print()
        input("Press Enter to close...")
