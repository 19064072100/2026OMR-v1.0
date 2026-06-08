"""
OMRExtractor - OMR 信息提取模块
负责从二值化图像中提取准考证号和答案信息
包含完整的 80 题遍历逻辑和像素密度计算
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from numpy.typing import NDArray

from config import OMRConfig


class OMRExtractor:
    """
    OMR 信息提取器
    
    核心功能：
    1. 提取 12 位准考证号（12列 × 10行网格）
    2. 提取 80 道选择题答案（10行 × 8列布局，每题4选项）
    
    算法原理：
    - 通过计算每个填涂框内的黑色像素密度来判断是否被填涂
    - 密度 > FILL_THRESHOLD 则认为该位置被填涂
    """
    
    def __init__(self, config: OMRConfig) -> None:
        self.config = config
        
        self._stats = {
            'total_cells_processed': 0,
            'avg_density': 0.0,
            'id_extraction_time': 0.0,
            'answers_extraction_time': 0.0
        }
    
    def extract_student_id(self, binary_img: NDArray) -> str:
        """
        提取准考证号（12位数字）
        
        网格结构：12列 × 10行
        - 每列代表准考证号的一位（共12位）
        - 每行代表数字 0-9（从上到下）
        
        Args:
            binary_img: 二值化图像（白色背景=255，黑色填涂=0）
            
        Returns:
            12位字符串格式的准考证号
        """
        if binary_img is None or binary_img.size == 0:
            raise ValueError("输入二值图像为空或无效")
        
        student_id = ""
        
        try:
            for col in range(9):
                max_density = -1.0
                best_digit = 0
                
                for row in range(10):
                    x, y, w, h = self.config.get_id_grid_coords(col, row)
                    
                    cell = self._safe_crop(binary_img, x, y, w, h)
                    
                    if cell is None or cell.size == 0:
                        continue
                    
                    density = self.calculate_density(cell)
                    
                    if density > max_density:
                        max_density = density
                        best_digit = row
                
                student_id += str(best_digit)
            
            if len(student_id) != 9:
                print(f"       ! ID length warning: {len(student_id)} (expected 9)")
            
            return student_id
            
        except Exception as e:
            print(f"       ! ID extraction failed: {str(e)}")
            return "000000000"
    
    def extract_answers(self, binary_img: NDArray) -> Dict[int, List[str]]:
        """
        提取80道选择题答案
        
        布局结构：10行 × 8列 = 80题
        - 每行包含8道题（横向排列）
        - 每题4个选项 A/B/C/D（纵向排列）
        
        Args:
            binary_img: 二值化图像
            
        Returns:
            字典格式：{题号: [选项列表]}
        """
        if binary_img is None or binary_img.size == 0:
            raise ValueError("输入二值图像为空或无效")
        
        answers: Dict[int, List[str]] = {}
        
        try:
            for row in range(10):
                for col in range(8):
                    question_num = row * 8 + col + 1
                    
                    if not (1 <= question_num <= 80):
                        print(f"[WARN] 题号越界: {question_num}")
                        continue
                    
                    q_x = self.config.ANS_START_X + col * self.config.Q_DX
                    base_y = self.config.ANS_START_Y + row * self.config.BLOCK_DY
                    
                    selected_options: List[str] = []
                    
                    for opt_idx in range(4):
                        opt_y = base_y + opt_idx * self.config.OPT_DY
                        
                        coords = self.config.get_answer_cell_coords(question_num, opt_idx)
                        x, y, w, h = coords
                        
                        cell = self._safe_crop(binary_img, x, y, w, h)
                        
                        if cell is None or cell.size == 0:
                            continue
                        
                        density = self.calculate_density(cell)
                        
                        if density >= self.config.FILL_THRESHOLD:
                            option_char = self.config.OPTION_MAP[opt_idx]
                            selected_options.append(option_char)
                    
                    answers[question_num] = selected_options
            
            expected_count = 80
            actual_count = len(answers)
            
            if actual_count != expected_count:
                print(f"[WARN] 答案提取不完整: {actual_count}/{expected_count} 题")
            
            return answers
            
        except Exception as e:
            print(f"[ERROR] 答案提取失败: {str(e)}")
            return {}
    
    def extract_answers_tf_mode(self, binary_img: NDArray) -> Dict[int, str]:
        """
        提取80道选择题答案（T/F模式）
        
        布局结构：10行 × 8列 = 80题
        - 每行包含8道题（横向排列）
        - 每题4个选项 A/B/C/D（纵向排列）
        - 每个选项下方有T和F两个选择框
        
        输出格式：每个题目输出4个字符的字符串（如 "TFTF"）
        - 第1个字符：A选项的T/F状态
        - 第2个字符：B选项的T/F状态
        - 第3个字符：C选项的T/F状态
        - 第4个字符：D选项的T/F状态
        
        判断逻辑：
        - 比较T框和F框的填涂密度
        - 密度差值小于阈值（0.15）时输出'E'表示模糊
        - 否则输出密度较大的那个（T或F）
        
        Args:
            binary_img: 二值化图像
            
        Returns:
            字典格式：{题号: "TFTF"格式字符串，每个字符为T/F/E之一}
        """
        if binary_img is None or binary_img.size == 0:
            raise ValueError("输入二值图像为空或无效")
        
        answers: Dict[int, str] = {}
        
        # 密度差值阈值，小于此值认为是模糊状态，输出'E'
        diff_threshold = 0.25
        
        try:
            for row in range(10):
                for col in range(8):
                    question_num = row * 8 + col + 1
                    
                    if not (1 <= question_num <= 80):
                        print(f"[WARN] 题号越界: {question_num}")
                        continue
                    
                    tf_result = []
                    
                    for opt_idx in range(4):
                        # 检测T框
                        t_coords = self.config.get_tf_cell_coords(question_num, opt_idx, is_true=True)
                        t_x, t_y, t_w, t_h = t_coords
                        t_cell = self._safe_crop(binary_img, t_x, t_y, t_w, t_h)
                        t_density = self.calculate_density(t_cell) if t_cell is not None else 0.0
                        
                        # 检测F框
                        f_coords = self.config.get_tf_cell_coords(question_num, opt_idx, is_true=False)
                        f_x, f_y, f_w, f_h = f_coords
                        f_cell = self._safe_crop(binary_img, f_x, f_y, f_w, f_h)
                        f_density = self.calculate_density(f_cell) if f_cell is not None else 0.0
                        
                        # 计算密度差值
                        density_diff = abs(t_density - f_density)
                        
                        # 判断逻辑：比较T和F的密度
                        if density_diff < diff_threshold:
                            # 密度接近，输出'E'表示模糊/错误
                            tf_result.append('E')
                        elif t_density > f_density:
                            # T密度更大，输出'T'
                            tf_result.append('T')
                        else:
                            # F密度更大，输出'F'
                            tf_result.append('F')
                    
                    answers[question_num] = ''.join(tf_result)
            
            expected_count = 80
            actual_count = len(answers)
            
            if actual_count != expected_count:
                print(f"[WARN] 答案提取不完整: {actual_count}/{expected_count} 题")
            
            return answers
            
        except Exception as e:
            print(f"[ERROR] T/F模式答案提取失败: {str(e)}")
            return {}
    
    def calculate_density(self, cell: NDArray) -> float:
        if cell is None or cell.size == 0:
            return 0.0
        
        try:
            total_pixels = cell.shape[0] * cell.shape[1]
            
            if total_pixels == 0:
                return 0.0
            
            black_pixels = np.count_nonzero(cell == 0)
            density = black_pixels / total_pixels
            
            self._stats['total_cells_processed'] += 1
            
            return float(density)
            
        except Exception as e:
            print(f"[WARN] 密度计算异常: {str(e)}")
            return 0.0
    
    def _safe_crop(
        self, 
        img: NDArray, 
        x: int, 
        y: int, 
        w: int, 
        h: int
    ) -> Optional[NDArray]:
        try:
            img_h, img_w = img.shape[:2]
            
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            cropped = img[y1:y2, x1:x2]
            
            return cropped
            
        except Exception as e:
            print(f"[WARN] 图像截取失败 ({x},{y},{w},{h}): {str(e)}")
            return None
    
    def get_statistics(self) -> Dict[str, any]:
        return self._stats.copy()
    
    def reset_statistics(self) -> None:
        self._stats = {
            'total_cells_processed': 0,
            'avg_density': 0.0,
            'id_extraction_time': 0.0,
            'answers_extraction_time': 0.0
        }


class ExtractorVisualizer:
    def __init__(self, config: OMRConfig) -> None:
        self.config = config
    
    def visualize_id_extraction(
        self, 
        original_img: NDArray, 
        student_id: str,
        output_path: Optional[str] = None
    ) -> NDArray:
        vis_img = original_img.copy()
        
        for col, digit in enumerate(student_id):
            row = int(digit)
            x, y, w, h = self.config.get_id_grid_coords(col, row)
            
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.putText(
                vis_img, str(digit),
                (x + w//2 - 5, y + h//2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        
        if output_path:
            cv2.imencode('.jpg', vis_img)[1].tofile(output_path)
        
        return vis_img
    
    def visualize_answers(
        self,
        original_img: NDArray,
        answers: Dict[int, List[str]],
        output_path: Optional[str] = None
    ) -> NDArray:
        vis_img = original_img.copy()
        
        colors = {
            'A': (0, 0, 255),      # 红
            'B': (0, 165, 255),    # 橙
            'C': (0, 255, 0),      # 绿
            'D': (255, 0, 0),      # 蓝
        }
        
        for q_num, options in answers.items():
            for opt in options:
                opt_idx = list(self.config.OPTION_MAP.values()).index(opt)
                x, y, w, h = self.config.get_answer_cell_coords(q_num, opt_idx)
                
                color = colors.get(opt, (128, 128, 128))
                
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
                
                cv2.putText(
                    vis_img, opt,
                    (x + w//2 - 5, y + h//2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )
        
        if output_path:
            cv2.imencode('.jpg', vis_img)[1].tofile(output_path)
        
        return vis_img


if __name__ == "__main__":
    from config import create_default_config
    
    config = create_default_config()
    extractor = OMRExtractor(config)
    
    print("OMRExtractor 测试 (新答题卡格式)")
    print("=" * 60)
    print("\n【配置参数】")
    print(f"  填涂阈值: {config.FILL_THRESHOLD}")
    print(f"  选项映射: {config.OPTION_MAP}")
    
    print("\n【准考证号区网格测试】")
    print("  显示前3列的坐标:")
    for col in range(3):
        for row in [0, 5, 9]:
            coords = config.get_id_grid_coords(col, row)
            print(f"    列{col} 行{row} (数字{row}): ({coords[0]:4d}, {coords[1]:4d})")
    
    print("\n【答题区关键点位测试】")
    test_points = [
        (1, 0, "Q1-A"),
        (1, 3, "Q1-D"),
        (8, 0, "Q8-A"),
        (9, 0, "Q9-A"),
        (40, 0, "Q40-A"),
        (41, 0, "Q41-A"),
        (79, 3, "Q79-D"),
        (80, 3, "Q80-D")
    ]
    
    for q, opt, label in test_points:
        coords = config.get_answer_cell_coords(q, opt)
        print(f"  {label:8s}: ({coords[0]:4d}, {coords[1]:4d})")
    
    print("\n[OK] OMRExtractor 模块加载成功！")