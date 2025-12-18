# SCI-Connectome-Tools

Pipeline for processing lightsheet microscopy data (LaVision UltraMicroscope) for brain-wide analysis using the BrainGlobe ecosystem.

## Overview

This pipeline converts Imaris `.ims` files from cleared brain samples into atlas-registered data for cell detection and regional analysis. It automatically handles spinal cord cropping and channel identification.

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `1_organize_pipeline.py` | Sets up folder structure, moves files into place |
| `2_extract_and_analyze.py` | Extracts TIFFs, detects crop boundaries, identifies channels |
| `3_register_to_atlas.py` | Registers to Allen Mouse Brain Atlas via brainreg |

## Quick Start

```bash
# Activate environment
conda activate brainglobe-env

# Navigate to scripts
cd Y:\2_Connectome\3_Nuclei_Detection\util_Scripts

# Run in order:
python 1_organize_pipeline.py
python 2_extract_and_analyze.py
python 3_register_to_atlas.py
```

## File Naming Convention

IMS files must be named: `NUMBER_PROJECT_COHORT_ANIMAL_MAGx_zSTEP.ims`

Example: `349_CNT_01_02_1.625x_z4.ims`

- `349` - Brain/sample number
- `CNT` - Project code (CNT=control, SCI=spinal cord injury, etc.)
- `01` - Cohort number
- `02` - Animal number within cohort
- `1.625x` - Magnification (used to calculate XY voxel size)
- `z4` - Z-step in microns

## Output Structure

```
1_Brains/
└── 349_CNT_01_02/
    └── 349_CNT_01_02_1p625x_z4/          # Pipeline folder
        ├── 0_Raw_IMS/                     # Original IMS file
        ├── 1_Extracted_Full/              # Full TIFF extraction
        │   ├── ch0/, ch1/
        │   ├── metadata.json
        │   └── QC_area_profile.png        # Crop detection visualization
        ├── 2_Cropped_For_Registration/    # Cropped for atlas registration
        │   ├── ch0/, ch1/
        │   └── metadata.json
        ├── 3_Registered_Atlas/            # brainreg output
        ├── 4_Cell_Candidates/             # cellfinder detection (future)
        ├── 5_Classified_Cells/            # cellfinder classification (future)
        └── 6_Region_Analysis/             # Regional counts (future)
```

## Key Features

### Automatic Crop Detection
Analyzes tissue cross-sectional area across Z-slices to detect where brain ends and spinal cord begins. Keeps brain tissue, removes cord that would interfere with atlas registration.

### Channel Identification
Automatically identifies signal vs background channels based on:
- **Sparsity**: Signal channels have sparse bright spots (e.g., c-Fos+ nuclei)
- **Local variance**: Signal channels are "speckly", background is smooth

### Smart Re-processing
Script 2 detects existing extractions and loads from TIFFs instead of re-extracting from IMS (much faster for re-runs).

## Requirements

```bash
conda activate brainglobe-env
pip install imaris-ims-file-reader tifffile numpy h5py scipy matplotlib
```

BrainGlobe tools: `brainreg`, `cellfinder`, `brainglobe-segmentation`

## Configuration

Edit the `DEFAULT_BRAINGLOBE_ROOT` in each script to match your setup:

```python
DEFAULT_BRAINGLOBE_ROOT = Path(r"Y:\2_Connectome\3_Nuclei_Detection\1_Brains")
```

Camera pixel size (for voxel calculations):
```python
CAMERA_PIXEL_SIZE = 6.5  # Andor Neo/Zyla sCMOS in microns
```

## Troubleshooting

### "doesn't match pattern" error
Rename your IMS file to match: `NUMBER_PROJECT_COHORT_ANIMAL_MAGx_zSTEP.ims`

### Napari can't drag-drop folders
There's probably a `thumbs.db` file. The scripts try to remove these, but Windows is persistent. Delete manually if needed.

### Registration fails
Check the QC_area_profile.png - if the crop detection looks wrong, you may need to adjust `CROP_AREA_THRESHOLD` in Script 2.

## License

Internal lab use.
