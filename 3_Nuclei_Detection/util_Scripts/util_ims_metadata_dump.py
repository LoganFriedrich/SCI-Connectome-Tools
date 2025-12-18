#!/usr/bin/env python3
"""
IMS Metadata Dumper

Dumps ALL metadata from an IMS file to help find voxel size information.
"""

import h5py
import sys
from pathlib import Path


def dump_attrs(name, obj, output_file=None):
    """Recursively dump all attributes from HDF5 objects."""
    if len(obj.attrs) > 0:
        line = f"\n{'='*60}\n{name}\n{'='*60}"
        print(line)
        if output_file:
            output_file.write(line + "\n")
        
        for key, val in obj.attrs.items():
            # Decode bytes if needed
            if isinstance(val, bytes):
                val = val.decode('utf-8', errors='replace')
            elif hasattr(val, '__iter__') and len(val) > 0:
                if isinstance(val[0], bytes):
                    val = [v.decode('utf-8', errors='replace') for v in val]
            
            line = f"  {key}: {val}"
            print(line)
            if output_file:
                output_file.write(line + "\n")


def dump_ims_metadata(filepath):
    """Dump all metadata from an IMS file."""
    filepath = Path(filepath)
    output_path = filepath.with_suffix('.metadata.txt')
    
    print(f"\nDumping metadata from: {filepath}")
    print(f"Output will be saved to: {output_path}")
    print("\n" + "="*60)
    print("FULL IMS METADATA DUMP")
    print("="*60)
    
    with open(output_path, 'w') as out_file:
        out_file.write(f"Metadata dump for: {filepath}\n")
        out_file.write("="*60 + "\n\n")
        
        with h5py.File(filepath, 'r') as f:
            # First, show the structure
            print("\n--- HDF5 Structure ---")
            out_file.write("--- HDF5 Structure ---\n")
            
            def print_structure(name, obj):
                indent = "  " * name.count('/')
                obj_type = "Group" if isinstance(obj, h5py.Group) else "Dataset"
                if isinstance(obj, h5py.Dataset):
                    line = f"{indent}{name} [{obj_type}] shape={obj.shape} dtype={obj.dtype}"
                else:
                    line = f"{indent}{name} [{obj_type}]"
                print(line)
                out_file.write(line + "\n")
            
            f.visititems(print_structure)
            
            # Now dump all attributes
            print("\n\n--- ALL ATTRIBUTES ---")
            out_file.write("\n\n--- ALL ATTRIBUTES ---\n")
            
            # Root attributes
            if len(f.attrs) > 0:
                print("\n[ROOT]")
                out_file.write("\n[ROOT]\n")
                for key, val in f.attrs.items():
                    if isinstance(val, bytes):
                        val = val.decode('utf-8', errors='replace')
                    line = f"  {key}: {val}"
                    print(line)
                    out_file.write(line + "\n")
            
            # Visit all groups and datasets
            f.visititems(lambda name, obj: dump_attrs(name, obj, out_file))
            
            # Special focus on DataSetInfo
            print("\n\n" + "="*60)
            print("LOOKING FOR VOXEL/CALIBRATION INFO...")
            print("="*60)
            out_file.write("\n\n" + "="*60 + "\n")
            out_file.write("LOOKING FOR VOXEL/CALIBRATION INFO...\n")
            out_file.write("="*60 + "\n")
            
            # Keywords that might indicate voxel/calibration info
            keywords = ['voxel', 'pixel', 'resolution', 'size', 'spacing', 
                       'scale', 'calibration', 'micron', 'um', 'extent', 
                       'physical', 'dimension', 'step', 'interval']
            
            found_items = []
            
            def search_for_keywords(name, obj):
                for key, val in obj.attrs.items():
                    key_lower = key.lower()
                    for kw in keywords:
                        if kw in key_lower:
                            if isinstance(val, bytes):
                                val = val.decode('utf-8', errors='replace')
                            elif hasattr(val, '__iter__') and len(val) > 0:
                                if isinstance(val[0], bytes):
                                    val = [v.decode('utf-8', errors='replace') for v in val]
                            found_items.append((name, key, val))
                            break
            
            f.visititems(search_for_keywords)
            
            if found_items:
                print("\nPotentially relevant attributes found:")
                out_file.write("\nPotentially relevant attributes found:\n")
                for path, key, val in found_items:
                    line = f"  {path} -> {key}: {val}"
                    print(line)
                    out_file.write(line + "\n")
            else:
                print("\nNo obvious voxel-related attributes found.")
                out_file.write("\nNo obvious voxel-related attributes found.\n")
            
            # Also try the imaris-ims-file-reader
            print("\n\n" + "="*60)
            print("TRYING imaris-ims-file-reader...")
            print("="*60)
            out_file.write("\n\n" + "="*60 + "\n")
            out_file.write("TRYING imaris-ims-file-reader...\n")
            out_file.write("="*60 + "\n")
            
    try:
        from imaris_ims_file_reader.ims import ims
        ims_data = ims(str(filepath), ResolutionLevelLock=0)
        
        # Check all attributes of the ims object
        attrs_to_check = ['resolution', 'metaData', 'shape', 'chunks', 
                          'Channels', 'TimePoints', 'ResolutionLevels']
        
        for attr in attrs_to_check:
            if hasattr(ims_data, attr):
                val = getattr(ims_data, attr)
                line = f"  ims.{attr}: {val}"
                print(line)
                with open(output_path, 'a') as out_file:
                    out_file.write(line + "\n")
        
        # Check if there's detailed metadata
        if hasattr(ims_data, 'metaData'):
            print("\n  Detailed metaData:")
            with open(output_path, 'a') as out_file:
                out_file.write("\n  Detailed metaData:\n")
            md = ims_data.metaData
            if hasattr(md, 'keys'):
                for key in list(md.keys())[:50]:  # First 50 keys
                    line = f"    {key}: {md[key]}"
                    print(line)
                    with open(output_path, 'a') as out_file:
                        out_file.write(line + "\n")
                        
    except Exception as e:
        line = f"  Error: {e}"
        print(line)
        with open(output_path, 'a') as out_file:
            out_file.write(line + "\n")
    
    print(f"\n\nFull dump saved to: {output_path}")
    print("Please share the relevant parts (especially anything with voxel/resolution info)")
    
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Interactive mode
        print("IMS Metadata Dumper")
        print("-" * 40)
        filepath = input("Paste path to .ims file:\n> ").strip().strip('"').strip("'")
    else:
        filepath = sys.argv[1]
    
    dump_ims_metadata(filepath)
    input("\nPress Enter to exit...")