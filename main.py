"""
OMR Answer Sheet Recognition System - Main Entry
================================================

Usage:
    python main.py --image <path>              Single image
    python main.py --folder <path>             Batch process
    python main.py --config <config.json>      Custom config
"""

import argparse
import sys
import os
from typing import Optional

from config import OMRConfig, create_default_config, create_high_resolution_config
from controller import OMRController


def print_banner() -> None:
    banner = """
=====================================================================
                    OMR System v1.0
---------------------------------------------------------------------
  Features:
    [+] 9-digit ID recognition
    [+] 80 questions extraction (T/F mode)
    [+] Batch processing + CSV export
    [+] Visualization for human review
=====================================================================
    """
    print(banner)


def process_single_image(
    image_path: str, 
    config: OMRConfig
) -> None:
    print(f"\n[Mode] Single Image Processing")
    print(f"   Path: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return
    
    try:
        controller = OMRController(config=config)
        result = controller.process_single_image(image_path)
        print_result_summary(result)
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()


def process_folder(
    folder_path: str,
    config: OMRConfig,
    output_csv: str = "results.csv"
) -> None:
    print(f"\n[Mode] Batch Processing")
    print(f"   Folder: {folder_path}")
    print(f"   Output: {output_csv}")
    
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return
    
    try:
        controller = OMRController(config=config)
        success = controller.process_folder(folder_path, output_csv)
        
        if success:
            print("\n[OK] Batch processing complete!")
            csv_size = os.path.getsize(output_csv) if os.path.exists(output_csv) else 0
            print(f"   Output: {output_csv} ({csv_size/1024:.1f} KB)")
        else:
            print("\n[WARN] Batch processing incomplete")
            
    except Exception as e:
        print(f"[ERROR] Batch failed: {str(e)}")
        import traceback
        traceback.print_exc()


def print_result_summary(result: dict) -> None:
    print(f"\n{'-'*60}")
    print("[Result Summary]")
    print(f"{'-'*60}")
    
    print(f"\n  Student ID: {result['student_id']}")
    
    answers = result.get('answers', {})
    t_count = sum(1 for tf_str in answers.values() if 'T' in tf_str)
    e_count = sum(1 for tf_str in answers.values() if 'E' in tf_str)
    
    print(f"  T count: {t_count}/80")
    print(f"  E count (needs review): {e_count}/80")
    
    if t_count > 0 and t_count <= 10:
        print(f"\n  Sample answers (first 5):")
        for q_num in sorted(answers.keys())[:5]:
            tf_str = answers[q_num]
            print(f"    Q{q_num:3d}: {tf_str}")
    
    status_icons = {
        'Success': '[OK]',
        'Error': '[FAIL]'
    }
    icon = status_icons.get(result['status'], '[?]')
    print(f"\n  Status: {icon} {result['status']}")
    
    if result.get('error_message'):
        print(f"  Note: {result['error_message']}")
    
    print(f"\n  Time: {result['processing_time']}s")
    print(f"{'-'*60}\n")


def run_calibration_wizard() -> None:
    print("\n" + "="*70)
    print("[Calibration Wizard]")
    print("="*70)
    
    wizard_text = """
This wizard helps you measure and configure coordinate parameters.

[Preparation]
1. Prepare a blank answer sheet scan/photo
2. Open it in Windows Paint (mspaint)

[Measurement Steps]

Step 1: Measure ID area start position
   - Hover mouse over [Digit 0 of Position 1] fill box
   - Record (X, Y) from status bar
   - ID_START_X = _____
   - ID_START_Y = _____

Step 2: Measure Answer area start position  
   - Hover mouse over [Option A of Question 1] fill box
   - Record (X, Y) from status bar
   - ANS_START_X = _____
   - ANS_START_Y = _____

Step 3: Calculate Q_DX (horizontal gap between questions)
   - Measure X of [Question 2 Option A]
   - Q_DX = X_Q2A - X_Q1A

Step 4: Calculate BLOCK_DY (vertical gap between rows)
   - Measure Y of [Question 9 Option A]
   - BLOCK_DY = Y_Q9A - Y_Q1A

[Save Config]
Edit config.py with your measured values.
"""
    print(wizard_text)


def generate_sample_config(output_path: str = "omr_config_template.json") -> None:
    config = create_default_config()
    
    try:
        config.save_to_json(output_path)
        print(f"\n[OK] Config template saved: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed: {str(e)}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OMR Answer Sheet Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --image scan.jpg                 Single image
  %(prog)s --folder ./images/               Batch process
  %(prog)s --calibrate                      Calibration wizard
        """
    )
    
    parser.add_argument('--image', '-i', type=str, help='Single image path')
    parser.add_argument('--folder', '-f', type=str, help='Folder with images')
    parser.add_argument('--output', '-o', type=str, default='results.csv', help='Output CSV')
    parser.add_argument('--config', '-c', type=str, help='Custom JSON config')
    parser.add_argument('--high-res', action='store_true', help='High resolution config')
    parser.add_argument('--calibrate', action='store_true', help='Calibration wizard')
    parser.add_argument('--generate-config', action='store_true', help='Generate config template')
    
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    
    print_banner()
    
    if args.calibrate:
        run_calibration_wizard()
        return 0
    
    if args.generate_config:
        generate_sample_config()
        return 0
    
    config: OMRConfig
    if args.config:
        try:
            config = OMRConfig.load_custom_config(args.config)
            print(f"[OK] Custom config loaded: {args.config}")
        except Exception as e:
            print(f"[WARN] Config load failed: {str(e)}")
            config = create_default_config()
    elif args.high_res:
        config = create_high_resolution_config()
        print("[OK] High resolution config")
    else:
        config = create_default_config()
        print("[OK] Default config")
    
    if args.image:
        process_single_image(
            image_path=args.image,
            config=config
        )
    elif args.folder:
        process_folder(
            folder_path=args.folder,
            config=config,
            output_csv=args.output
        )
    else:
        print("\nPlease specify mode:")
        print("  --image <path>     Single image")
        print("  --folder <path>    Batch processing")
        print("  --calibrate       Calibration wizard")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)