#!/usr/bin/env python3
r"""
3_register_to_atlas.py (v2.0.0)

================================================================================
WHAT IS THIS?
================================================================================
This is Script 3 in the BrainGlobe pipeline. It registers your cropped brain
images to the Allen Mouse Brain Atlas using brainreg.

Think of it as: "Align my brain to the atlas so we know where everything is"

Run this AFTER Script 2 (extract_and_analyze.py).
Run this BEFORE Script 4 (cell detection).

================================================================================
WHAT IT DOES
================================================================================
1. Reads metadata from Script 2 (channel roles, voxel sizes)
2. Runs brainreg to register your data to the Allen atlas
3. Archives any previous registration attempts (never overwrites)
4. Generates QC images for verification
5. Prepares output for cellfinder

================================================================================
HOW TO RUN
================================================================================
Open Anaconda Prompt, then:

    conda activate brainglobe-env
    cd Y:\2_Connectome\3_Nuclei_Detection\util_Scripts
    python 3_register_to_atlas.py

The script will show an interactive menu of available brains.

================================================================================
INTERACTIVE MENU
================================================================================
When you run the script, you'll see something like:

    [READY TO REGISTER]
      1. 349_CNT_01_02/349_CNT_01_02_1p625x_z4
      2. 350_SCI_02_05/350_SCI_02_05_1p9x_z3p37

    [ALREADY REGISTERED]
      3. 348_CNT_01_01/348_CNT_01_01_1p625x_z4

    Options:
      - Enter numbers to process (e.g., '1' or '1,2')
      - 'all' to process all ready
      - 'reprocess' to redo completed ones
      - 'q' to quit

================================================================================
REGISTRATION ARCHIVING
================================================================================
When you re-register a brain, the old registration is NOT deleted. Instead:

    3_Registered_Atlas/
    ├── registered_atlas.tiff      ← Current registration
    ├── brainreg.json
    ├── QC_registration.png
    └── _archive/
        ├── 20241216_143052/       ← Previous attempt #1
        └── 20241216_160215/       ← Previous attempt #2

You can always roll back by moving files from _archive back to main folder.

================================================================================
COMMAND LINE OPTIONS
================================================================================
    python 3_register_to_atlas.py              # Interactive mode
    python 3_register_to_atlas.py --batch      # Process all pending automatically
    python 3_register_to_atlas.py --inspect    # Dry run - show status only
    python 3_register_to_atlas.py --n-free-cpus 106  # Limit CPU usage

================================================================================
SETTINGS (edit in script if needed)
================================================================================
    DEFAULT_ATLAS = "allen_mouse_10um"   # Atlas to use
    DEFAULT_N_FREE_CPUS = 106            # CPUs to leave free (for servers)

================================================================================
REQUIREMENTS
================================================================================
    pip install brainreg brainglobe-napari-io
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# VERSION
# =============================================================================
SCRIPT_VERSION = "2.0.1"

# =============================================================================
# PROGRESS HELPERS
# =============================================================================
def timestamp():
    """Get current time as formatted string."""
    return datetime.now().strftime("%H:%M:%S")

# =============================================================================
# DEFAULT SETTINGS
# =============================================================================
DEFAULT_BRAINGLOBE_ROOT = Path(r"Y:\2_Connectome\3_Nuclei_Detection\1_Brains")

# Atlas to use (allen_mouse_10um or allen_mouse_25um)
DEFAULT_ATLAS = "allen_mouse_10um"

# CPUs to leave free (for servers with many cores)
# Set to None to let brainreg decide, or a number like 106 (for 180-core server using 74)
DEFAULT_N_FREE_CPUS = 106

# =============================================================================
# PIPELINE FOLDER NAMES (must match Scripts 1 & 2)
# =============================================================================
FOLDER_RAW_IMS = "0_Raw_IMS"
FOLDER_EXTRACTED_FULL = "1_Extracted_Full"
FOLDER_CROPPED = "2_Cropped_For_Registration"
FOLDER_REGISTRATION = "3_Registered_Atlas"

# Files that indicate a complete registration
REGISTRATION_REQUIRED_FILES = [
    "brainreg.json",
]


# =============================================================================
# DISCOVERY AND STATUS
# =============================================================================

def find_extracted_pipelines(root_path):
    """
    Find all pipelines that have been extracted (have cropped data).
    Returns list of (pipeline_folder, mouse_folder, metadata) tuples.
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
            
            # Check for cropped data with metadata
            crop_folder = pipeline_dir / FOLDER_CROPPED
            metadata_path = crop_folder / "metadata.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Verify cropped data exists
                    ch0 = crop_folder / "ch0"
                    if ch0.exists() and len(list(ch0.glob('Z*.tif'))) > 0:
                        pipelines.append((pipeline_dir, mouse_dir, metadata))
                except:
                    pass
    
    return pipelines


def check_registration_status(pipeline_folder):
    """
    Check if registration has been done.
    
    Returns:
        (status, reason)
        
        Status:
        - "needs_registration": Not registered yet
        - "incomplete": Started but failed/incomplete
        - "complete": Successfully registered
    """
    pipeline_folder = Path(pipeline_folder)
    reg_folder = pipeline_folder / FOLDER_REGISTRATION
    
    if not reg_folder.exists():
        return "needs_registration", "registration folder missing"
    
    # Get contents, ignoring hidden files and archive
    contents = [f for f in reg_folder.iterdir() 
                if not f.name.startswith('.') 
                and f.name.lower() not in ['thumbs.db', 'desktop.ini']
                and f.name != '_archive']
    
    if len(contents) == 0:
        return "needs_registration", "registration folder empty"
    
    # Check for required files
    missing = []
    for req_file in REGISTRATION_REQUIRED_FILES:
        if not (reg_folder / req_file).exists():
            missing.append(req_file)
    
    if missing:
        return "incomplete", f"missing: {', '.join(missing)}"
    
    return "complete", "registration complete"


# =============================================================================
# ARCHIVING
# =============================================================================

def archive_existing_registration(pipeline_folder):
    """
    Move existing registration to _archive folder with timestamp.
    Returns True if anything was archived.
    """
    reg_folder = Path(pipeline_folder) / FOLDER_REGISTRATION
    
    if not reg_folder.exists():
        return False
    
    # Get contents to archive (excluding _archive folder itself)
    contents = [f for f in reg_folder.iterdir() 
                if f.name != '_archive' 
                and not f.name.startswith('.')
                and f.name.lower() not in ['thumbs.db', 'desktop.ini']]
    
    if not contents:
        return False
    
    # Create archive folder
    archive_base = reg_folder / "_archive"
    archive_base.mkdir(exist_ok=True)
    
    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_folder = archive_base / timestamp
    archive_folder.mkdir()
    
    # Move contents
    for item in contents:
        dest = archive_folder / item.name
        shutil.move(str(item), str(dest))
    
    return True


# =============================================================================
# REGISTRATION
# =============================================================================

def run_brainreg(pipeline_folder, metadata, n_free_cpus=None, atlas=DEFAULT_ATLAS):
    """
    Run brainreg on a pipeline.
    
    Uses metadata from Script 2 to determine:
    - Which channel to register (background channel preferred)
    - Voxel sizes
    - Orientation
    """
    pipeline_folder = Path(pipeline_folder)
    
    crop_folder = pipeline_folder / FOLDER_CROPPED
    reg_folder = pipeline_folder / FOLDER_REGISTRATION
    
    # Ensure registration folder exists
    reg_folder.mkdir(parents=True, exist_ok=True)
    
    # Get registration channel from metadata
    # Prefer background channel for registration (clearer tissue boundaries)
    channels_info = metadata.get('channels', {})
    background_ch = channels_info.get('background_channel')
    signal_ch = channels_info.get('signal_channel', 0)
    
    reg_channel = background_ch if background_ch is not None else signal_ch
    input_folder = crop_folder / f"ch{reg_channel}"
    
    # Get voxel sizes
    voxel = metadata.get('voxel_size_um', {})
    voxel_z = voxel.get('z', 4.0)
    voxel_xy = voxel.get('x', 4.0)
    
    # Get orientation
    orientation = metadata.get('orientation', 'iar')
    
    # Build brainreg command
    cmd = [
        "brainreg",
        str(input_folder),
        str(reg_folder),
        "-v", str(voxel_z), str(voxel_xy), str(voxel_xy),
        "--orientation", orientation,
        "--atlas", atlas,
    ]
    
    # Add additional channels if present
    num_channels = channels_info.get('count', 1)
    if num_channels > 1:
        for ch_idx in range(num_channels):
            if ch_idx != reg_channel:
                additional_folder = crop_folder / f"ch{ch_idx}"
                if additional_folder.exists():
                    cmd.extend(["-a", str(additional_folder)])
    
    # Add CPU limit if specified
    if n_free_cpus is not None:
        cmd.extend(["--n-free-cpus", str(n_free_cpus)])
    
    print(f"    Registration channel: ch{reg_channel}")
    print(f"    Voxel sizes (Z,Y,X): {voxel_z}, {voxel_xy}, {voxel_xy} µm")
    print(f"    Atlas: {atlas}")
    print(f"    Orientation: {orientation}")
    if n_free_cpus:
        print(f"    CPU limit: leaving {n_free_cpus} cores free")
    print(f"\n    [{timestamp()}] Starting brainreg (this typically takes 20-60 minutes)...")
    print(f"    Command: {' '.join(cmd)}")
    print(f"    {'='*50}")
    sys.stdout.flush()
    
    # Run brainreg
    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"    {'='*50}")
    if result.returncode != 0:
        raise RuntimeError(f"brainreg failed with return code {result.returncode}")
    
    print(f"    [{timestamp()}] Registration completed in {elapsed/60:.1f} minutes")
    
    # Verify output
    status, reason = check_registration_status(pipeline_folder)
    if status != "complete":
        raise RuntimeError(f"Registration verification failed: {reason}")
    
    return True


# =============================================================================
# INTERACTIVE MENU
# =============================================================================

def interactive_select_pipelines(pipelines, statuses):
    """
    Show interactive menu for selecting which pipelines to process.
    
    Returns list of indices to process, or None to cancel.
    """
    # Categorize pipelines
    ready = []
    complete = []
    incomplete = []
    
    for i, ((pipeline_folder, mouse_folder, metadata), (status, reason)) in enumerate(zip(pipelines, statuses)):
        display_name = f"{mouse_folder.name}/{pipeline_folder.name}"
        if status == "needs_registration":
            ready.append((i, display_name))
        elif status == "complete":
            complete.append((i, display_name))
        else:
            incomplete.append((i, display_name, reason))
    
    # Display menu
    print("\n" + "=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)
    
    if ready:
        print("\n[READY TO REGISTER]")
        for idx, name in ready:
            print(f"  {idx + 1}. {name}")
    
    if complete:
        print("\n[ALREADY REGISTERED]")
        for idx, name in complete:
            print(f"  {idx + 1}. {name}")
    
    if incomplete:
        print("\n[INCOMPLETE/FAILED]")
        for idx, name, reason in incomplete:
            print(f"  {idx + 1}. {name} ({reason})")
    
    print("\n" + "-" * 60)
    print("Options:")
    print("  - Enter numbers to process (e.g., '1' or '1,2,3')")
    print("  - 'all' to process all ready")
    print("  - 'reprocess' to redo already-completed")
    print("  - 'retry' to retry incomplete/failed")
    print("  - 'q' to quit")
    print("-" * 60)
    
    while True:
        response = input("\nSelection: ").strip().lower()
        
        if response == 'q':
            return None
        
        if response == 'all':
            if not ready:
                print("No pipelines ready to register.")
                continue
            return [idx for idx, _ in ready]
        
        if response == 'reprocess':
            if not complete:
                print("No completed registrations to reprocess.")
                continue
            return [idx for idx, _ in complete]
        
        if response == 'retry':
            if not incomplete:
                print("No incomplete registrations to retry.")
                continue
            return [idx for idx, _, _ in incomplete]
        
        # Parse comma-separated numbers
        try:
            indices = []
            for part in response.split(','):
                num = int(part.strip()) - 1  # Convert to 0-indexed
                if 0 <= num < len(pipelines):
                    indices.append(num)
                else:
                    print(f"Invalid number: {num + 1}")
                    indices = None
                    break
            
            if indices:
                return indices
        except ValueError:
            print("Invalid input. Enter numbers, 'all', 'reprocess', 'retry', or 'q'.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Register brain images to atlas using brainreg',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Version: {SCRIPT_VERSION}

This script registers your cropped brain data to the Allen atlas.
Run AFTER Script 2 (extract_and_analyze.py).

Examples:
  python 3_register_to_atlas.py
  python 3_register_to_atlas.py --batch
  python 3_register_to_atlas.py --n-free-cpus 106
        """
    )
    
    parser.add_argument('path', nargs='?', default=None,
                        help=f'Path to scan (default: {DEFAULT_BRAINGLOBE_ROOT})')
    parser.add_argument('--inspect', '-i', action='store_true',
                        help='Dry run - show status only')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Non-interactive - process all pending')
    parser.add_argument('--n-free-cpus', type=int, default=DEFAULT_N_FREE_CPUS,
                        help=f'CPUs to leave free (default: {DEFAULT_N_FREE_CPUS})')
    parser.add_argument('--atlas', default=DEFAULT_ATLAS,
                        help=f'Atlas to use (default: {DEFAULT_ATLAS})')
    
    args = parser.parse_args()
    
    root_path = Path(args.path) if args.path else DEFAULT_BRAINGLOBE_ROOT
    
    if not root_path.exists():
        print(f"Error: Path not found: {root_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("BrainGlobe Atlas Registration")
    print(f"Version: {SCRIPT_VERSION}")
    print(f"Atlas: {args.atlas}")
    if args.n_free_cpus:
        print(f"CPU Limit: leaving {args.n_free_cpus} cores free")
    print("=" * 60)
    
    # Find extracted pipelines
    print(f"\nScanning: {root_path}")
    pipelines = find_extracted_pipelines(root_path)
    
    if not pipelines:
        print("\nNo extracted pipelines found.")
        print("Run Script 2 (extract_and_analyze.py) first!")
        return
    
    # Check status of each
    statuses = []
    for pipeline_folder, mouse_folder, metadata in pipelines:
        status, reason = check_registration_status(pipeline_folder)
        statuses.append((status, reason))
    
    if args.inspect:
        # Just show status
        print(f"\nFound {len(pipelines)} extracted pipeline(s):\n")
        for (pipeline_folder, mouse_folder, metadata), (status, reason) in zip(pipelines, statuses):
            symbol = "✓" if status == "complete" else "○" if status == "needs_registration" else "⚠"
            print(f"  {symbol} {mouse_folder.name}/{pipeline_folder.name} - {reason}")
        return
    
    # Select pipelines to process
    if args.batch:
        # Non-interactive: process all pending
        to_process = [i for i, (status, _) in enumerate(statuses) 
                      if status in ["needs_registration", "incomplete"]]
        if not to_process:
            print("\nNo pipelines need registration. All complete!")
            return
    else:
        # Interactive selection
        to_process = interactive_select_pipelines(pipelines, statuses)
        if to_process is None:
            print("Cancelled.")
            return
        if not to_process:
            print("Nothing selected.")
            return
    
    # Confirm
    print(f"\nWill process {len(to_process)} pipeline(s):")
    for idx in to_process:
        pipeline_folder, mouse_folder, metadata = pipelines[idx]
        print(f"  - {mouse_folder.name}/{pipeline_folder.name}")
    
    if not args.batch:
        response = input("\nProceed? [Enter to continue, 'q' to quit]: ").strip()
        if response.lower() == 'q':
            print("Cancelled.")
            return
    
    # Process
    print("\n" + "=" * 60)
    print("Processing...")
    print("=" * 60)
    
    success = 0
    failed = 0
    
    for i, idx in enumerate(to_process):
        pipeline_folder, mouse_folder, metadata = pipelines[idx]
        status, _ = statuses[idx]
        
        print(f"\n[{i+1}/{len(to_process)}] {mouse_folder.name}/{pipeline_folder.name}")
        print("-" * 50)
        
        try:
            # Archive existing if reprocessing
            if status == "complete":
                print("    Archiving previous registration...")
                archived = archive_existing_registration(pipeline_folder)
                if archived:
                    print("    ✓ Previous registration archived")
            
            # Run registration
            run_brainreg(
                pipeline_folder, 
                metadata,
                n_free_cpus=args.n_free_cpus,
                atlas=args.atlas
            )
            
            success += 1
            print(f"\n    ✓ Registration complete")
            
        except Exception as e:
            failed += 1
            print(f"\n    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Complete: {success} succeeded, {failed} failed")
    print("=" * 60)
    
    if success > 0:
        print("\n✓ Ready for Script 4 (cell detection)!")


if __name__ == '__main__':
    main()
    
    if len(sys.argv) == 1:
        print()
        input("Press Enter to close...")
