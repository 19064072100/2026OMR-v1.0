"""
OMRController - OMR 系统核心调度模块
负责协调所有模块完成完整的答题卡识别流程
实现单张图片处理和批量文件夹处理功能
"""

import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[WARN] pandas not installed, CSV export will be disabled")

from config import OMRConfig, create_default_config
from preprocessor import ImagePreProcessor, load_image_safe
from extractor import OMRExtractor
from enhanced_visualizer import EnhancedVisualizer


class OMRController:
    """
    OMR 系统主控制器
    
    职责：
    1. 初始化和配置所有子模块
    2. 协调整个处理流水线（Pipeline）
    3. 处理单张图像或批量处理整个文件夹
    4. 结果汇总与导出
    
    处理流水线：
    ┌─────────────┐
    │ 读取图像     │ → load_image_safe()
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ 图像预处理   │ → get_binary_image()
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ 信息提取     │ → OMRExtractor.extract_student_id()
    │ (考号+答案)  │ → OMRExtractor.extract_answers_tf_mode()
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ 结果存储     │ → 汇总到列表 / 导出CSV
    └─────────────┘
    """
    
    def __init__(self, config: Optional[OMRConfig] = None) -> None:
        """
        初始化 OMR 控制器
        
        Args:
            config: OMR 配置实例，如果为 None 则使用默认配置
        """
        self.config = config if config else create_default_config()
        
        self.pre_processor = ImagePreProcessor(self.config)
        self.extractor = OMRExtractor(self.config)
        self.visualizer = EnhancedVisualizer(self.config)
        
        self.results: List[Dict[str, Any]] = []
        
        self._stats = {
            'total_images': 0,
            'success_count': 0,
            'error_count': 0,
            'start_time': None,
            'end_time': None
        }
        
        print("=" * 60)
        print("[OMR] Answer Sheet Recognition System Initialized")
        print(f"  Standard Size: {self.config.STD_WIDTH} x {self.config.STD_HEIGHT}")
        print(f"  TF Mode: {'Enabled' if self.config.TF_MODE else 'Disabled'}")
        print("=" * 60)
    
    def process_single_image(self, img_path: str) -> Dict[str, Any]:
        """
        处理单个答题卡图像（完整流水线）
        
        Args:
            img_path: 图像文件路径
            
        Returns:
            包含以下字段的字典：
            - student_id: 学生考号 (str)
            - answers: 答案字典 {题号: "TFTF"格式字符串} (dict)
            - status: 处理状态 (str)
            - processing_time: 处理耗时（秒）(float)
        """
        start_time = time.time()
        result: Dict[str, Any] = {
            'image_path': img_path,
            'student_id': '',
            'answers': {},
            'status': 'Error',
            'processing_time': 0.0
        }
        
        self._stats['total_images'] += 1
        
        print(f"\n{'='*60}")
        print(f"[IMAGE] Processing: {os.path.basename(img_path)}")
        print(f"{'='*60}")
        
        try:
            print("  [1/3] Loading image...")
            img, error = load_image_safe(img_path)
            
            if img is None:
                raise ValueError(f"Failed to load image: {error}")
            
            print(f"       [OK] Image size: {img.shape[1]}x{img.shape[0]}")
            
            print("  [2/3] Preprocessing...")
            aligned_img = img.copy()
            self.pre_processor._last_aligned_image = aligned_img
            binary_img = self.pre_processor.get_binary_image(aligned_img)
            print(f"       [OK] Binary image: {binary_img.shape[1]}x{binary_img.shape[0]}")
            
            print("  [3/3] Extracting data...")
            
            student_id = self.extractor.extract_student_id(binary_img)
            result['student_id'] = student_id
            print(f"       [OK] Student ID: {student_id}")
            
            answers = self.extractor.extract_answers_tf_mode(binary_img)
            result['answers'] = answers
            
            t_count = sum(1 for tf_str in answers.values() if 'T' in tf_str)
            e_count = sum(1 for tf_str in answers.values() if 'E' in tf_str)
            print(f"       [OK] T/F Mode - T count: {t_count}/80, E count: {e_count}/80")
            
            result['status'] = 'Success'
            self._stats['success_count'] += 1
            
        except ValueError as e:
            result['error_message'] = str(e)
            self._stats['error_count'] += 1
            print(f"       [ERROR] {str(e)}")
            
        except Exception as e:
            result['error_message'] = f"Unexpected error: {str(e)}"
            self._stats['error_count'] += 1
            print(f"       [ERROR] Processing failed: {str(e)}")
        
        result['processing_time'] = round(time.time() - start_time, 3)
        print(f"\n  [TIME] Processing time: {result['processing_time']}s")
        
        self.results.append(result)
        return result
    
    def process_folder(
        self, 
        folder_path: str, 
        output_csv: Optional[str] = None
    ) -> bool:
        """
        批量处理文件夹中的所有答题卡图像
        
        Args:
            folder_path: 文件夹路径
            output_csv: 输出 CSV 文件路径
            
        Returns:
            是否处理成功
        """
        self._stats['start_time'] = datetime.now()
        self.results.clear()
        
        if not os.path.exists(folder_path):
            print(f"[ERROR] Folder not found: {folder_path}")
            return False
        
        actual_csv_path = output_csv if output_csv else self.config.OUTPUT_CSV_PATH
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        if not image_files:
            print(f"[WARN] No image files found in: {folder_path}")
            return False
        
        print(f"\n{'#'*60}")
        print(f"[START] Batch Processing")
        print(f"   Folder: {folder_path}")
        print(f"   Found {len(image_files)} images")
        print(f"{'#'*60}\n")
        
        success_count = 0
        
        vis_output_dir = os.path.join(self.config.OUTPUT_DIR, 'visualized')
        os.makedirs(vis_output_dir, exist_ok=True)
        
        for i, filename in enumerate(image_files, 1):
            img_path = os.path.join(folder_path, filename)
            
            print(f"\n[{i}/{len(image_files)}]", end=" ")
            result = self.process_single_image(img_path)
            
            if result['status'] == 'Success':
                success_count += 1
            
            self._generate_visualization(result, vis_output_dir, filename)
        
        self._stats['end_time'] = datetime.now()
        
        if PANDAS_AVAILABLE and self.results:
            self.export_to_csv(actual_csv_path)
        
        self._print_summary(success_count, len(image_files))
        
        return success_count > 0
    
    def export_to_csv(self, output_path: str) -> bool:
        """
        将处理结果导出为 CSV 文件
        
        CSV 格式：Student_ID,Q1,Q2,...,Q80,Processing_Time
        """
        if not PANDAS_AVAILABLE:
            print("[ERROR] pandas required for CSV export")
            return False
        
        try:
            rows = []
            
            for result in self.results:
                row = {
                    'Student_ID': result['student_id'],
                    'Processing_Time': result['processing_time'],
                    'Image_Path': result['image_path']
                }
                
                answers = result.get('answers', {})
                for q_num in range(1, 81):
                    answer_val = answers.get(q_num, '')
                    row[f'Q{q_num}'] = answer_val if isinstance(answer_val, str) else ''
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            columns_order = ['Student_ID']
            columns_order.extend([f'Q{i}' for i in range(1, 81)])
            columns_order.extend(['Processing_Time', 'Image_Path'])
            
            df = df.reindex(columns=columns_order)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"\n[OK] Results exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] CSV export failed: {str(e)}")
            return False
    
    def _generate_visualization(
        self, 
        result: Dict[str, Any], 
        output_dir: str, 
        original_filename: str
    ) -> None:
        """
        为每张答题卡生成可视化复核图
        """
        aligned_img = self.pre_processor.get_last_aligned_image()
        if aligned_img is None:
            return
        
        base_name = os.path.splitext(original_filename)[0]
        
        viz = EnhancedVisualizer(self.config)
        grid_path = os.path.join(output_dir, f'{base_name}_grid.jpg')
        viz.visualize_full_grid(
            aligned_img,
            result.get('answers', {}),
            grid_path,
            show_unanswered=True,
            show_question_numbers=True,
            tf_mode=True
        )
        
        print(f"       + Visualization saved to: {output_dir}/")
    
    def _print_summary(self, success_count: int, total_count: int) -> None:
        """打印处理统计摘要"""
        duration = None
        if self._stats['start_time'] and self._stats['end_time']:
            duration = (self._stats['end_time'] - self._stats['start_time']).total_seconds()
        
        print(f"\n{'='*60}")
        print("[Summary] Processing Complete")
        print(f"{'='*60}")
        print(f"  Total images: {total_count}")
        print(f"  Success: {success_count} ({success_count/total_count*100:.1f}%)")
        print(f"  Errors: {self._stats['error_count']}")
        
        if duration:
            print(f"  Total time: {duration:.2f}s")
            if total_count > 0:
                print(f"  Avg time: {duration/total_count:.2f}s/image")
        
        vis_dir = os.path.join(self.config.OUTPUT_DIR, 'visualized')
        csv_path = self.config.OUTPUT_CSV_PATH
        print(f"\n[Output Files]")
        print(f"  CSV results: {csv_path}")
        print(f"  Visualized: {vis_dir}/")
        
        print(f"{'='*60}\n")
    
    def get_results(self) -> List[Dict[str, Any]]:
        return self.results.copy()
    
    def get_statistics(self) -> Dict[str, any]:
        return self._stats.copy()


if __name__ == "__main__":
    print("OMRController Test")
    print("=" * 60)
    
    controller = OMRController()
    print("\n[OK] OMRController module loaded successfully!")