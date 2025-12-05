"""
Dataset Integrity Checker for D-Fire Smoke Detection Dataset
Validates images, labels, and data quality for YOLO training.

Requirements:
    pip install Pillow tqdm
"""

import os
import glob
import logging
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATASET_ROOT = r"d:\pet-project\smoke-detection\dataset\d-fire"
IMG_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')

def check_integrity(folder_name: str) -> Optional[Dict]:
    """
    Check dataset integrity for a given folder (train/test/val).
    
    Args:
        folder_name: Name of the folder to check (train, test, val)
        
    Returns:
        Dictionary with statistics or None if folder not found
    """
    folder_path = os.path.join(DATASET_ROOT, folder_name)
    if not os.path.exists(folder_path):
        logger.warning(f"Folder not found: {folder_name}")
        return None

    logger.info(f"Checking folder: {folder_name}")
    
    # Find all images recursively
    images_list = []
    for ext in IMG_FORMATS:
        pattern = os.path.join(folder_path, '**', f'*{ext}')
        images_list.extend(glob.glob(pattern, recursive=True))
    
    if not images_list:
        logger.error(f"No images found in {folder_name}")
        return None

    stats = {
        'total_images': len(images_list),
        'corrupt_images': 0,
        'missing_labels': 0,
        'empty_labels': 0,
        'valid_objects': 0,
        'class_counts': {},
        'errors': []
    }

    logger.info(f"Found {len(images_list)} images. Scanning...")

    for img_path in tqdm(images_list, desc=f"Processing {folder_name}"):
        img_path_obj = Path(img_path)
        
        # Validate image integrity
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            stats['corrupt_images'] += 1
            stats['errors'].append(f"Corrupt image: {img_path_obj.name}")
            continue

        # Find corresponding label file
        label_path = _find_label_file(img_path_obj)

        # Process label
        if label_path and label_path.exists():
            _process_label(label_path, stats)
        else:
            stats['missing_labels'] += 1

    return stats


def _find_label_file(img_path: Path) -> Optional[Path]:
    """
    Find label file for given image.
    Checks both same directory and images/labels structure.
    
    Args:
        img_path: Path to image file
        
    Returns:
        Path to label file or None if not found
    """
    # Case 1: Same directory
    potential_path = img_path.with_suffix('.txt')
    if potential_path.exists():
        return potential_path
    
    # Case 2: images/ labels/ parallel structure
    parts = list(img_path.parts)
    if 'images' in parts:
        idx = parts.index('images')
        parts[idx] = 'labels'
        potential_path_2 = Path(*parts).with_suffix('.txt')
        if potential_path_2.exists():
            return potential_path_2
    
    return None


def _process_label(label_path: Path, stats: Dict) -> None:
    """
    Process and validate label file.
    
    Args:
        label_path: Path to label file
        stats: Statistics dictionary to update
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            stats['empty_labels'] += 1
            return
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    
                    # Validate coordinates are normalized (0-1)
                    if any(c < 0 or c > 1 for c in coords):
                        stats['errors'].append(
                            f"Invalid coordinates in {label_path.name}"
                        )
                    
                    stats['class_counts'][cls_id] = \
                        stats['class_counts'].get(cls_id, 0) + 1
                    stats['valid_objects'] += 1
                except ValueError as e:
                    stats['errors'].append(
                        f"Parse error in {label_path.name}: {e}"
                    )
    except Exception as e:
        stats['errors'].append(f"Error reading {label_path.name}: {e}")


def print_report(stats: Optional[Dict], name: str) -> None:
    """
    Print formatted statistics report.
    
    Args:
        stats: Statistics dictionary
        name: Name of the dataset split
    """
    if not stats:
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DATASET REPORT: {name.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Total images:         {stats['total_images']}")
    logger.info(f"Corrupt images:       {stats['corrupt_images']}")
    logger.info(f"Empty labels:         {stats['empty_labels']} (background)")
    logger.info(f"Missing labels:       {stats['missing_labels']}")
    logger.info(f"Total objects:        {stats['valid_objects']}")
    logger.info(f"Class distribution:   {stats['class_counts']}")
    
    if stats['errors']:
        logger.warning(f"\nFound {len(stats['errors'])} issues:")
        for err in stats['errors'][:10]:
            logger.warning(f"  - {err}")
        if len(stats['errors']) > 10:
            logger.warning(f"  ... and {len(stats['errors']) - 10} more")
    
    logger.info(f"{'='*60}\n")

def main() -> None:
    """Main entry point for dataset validation."""
    logger.info("Starting D-Fire dataset integrity check...")
    
    # Check all dataset splits
    train_stats = check_integrity('train')
    test_stats = check_integrity('test')
    val_stats = check_integrity('val')

    # Print reports
    print_report(train_stats, 'train')
    print_report(test_stats, 'test')
    print_report(val_stats, 'val')

    logger.info("Dataset validation complete!")


if __name__ == "__main__":
    main()