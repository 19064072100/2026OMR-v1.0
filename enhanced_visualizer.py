"""
OMR 增强版可视化工具 - 显示完整网格视图
适配新格式：80题(10行×8列) + 4选项(A/B/C/D)

改进：每个选项框都按类型显示不同颜色，整体更清晰美观
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
from numpy.typing import NDArray

from config import OMRConfig


class EnhancedVisualizer:
    """
    增强版可视化器
    
    功能：
    1. 显示所有320个选项框（80题 × 4选项）
    2. 每个选项框按类型显示不同颜色（A/B/C/D 各一种颜色）
    3. 已填涂的选项：高亮实心 + 粗边框 + 字母标注
    4. 未填涂的选项：同色系半透明/细边框
    5. 标注题号和选项字母
    6. 支持T/F模式显示
    """
    
    def __init__(self, config: OMRConfig) -> None:
        self.config = config
        
        # 选项颜色映射（BGR格式 - OpenCV使用BGR）
        self.option_colors = {
            'A': (0, 0, 255),      # 红
            'B': (0, 127, 255),    # 橙红
            'C': (0, 200, 200),    # 黄绿
            'D': (0, 255, 0),      # 绿
        }
        
        # 未填涂时的淡色版本（用于显示网格结构）
        self.option_colors_light = {
            'A': (100, 80, 180),   # 浅红
            'B': (80, 120, 180),   # 浅橙
            'C': (80, 160, 150),   # 浅黄绿
            'D': (80, 190, 110),   # 浅绿
        }
        
        # T/F颜色
        self.tf_colors = {
            'T': (0, 255, 0),      # 绿色表示T
            'F': (0, 0, 255),      # 红色表示F
            'E': (0, 255, 255),    # 青色表示模糊/错误(Equal)
        }
    
    def visualize_full_grid(
        self,
        original_img: NDArray,
        answers: Dict[int, List[str]],
        output_path: Optional[str] = None,
        show_unanswered: bool = True,
        show_question_numbers: bool = True,
        tf_mode: bool = False
    ) -> NDArray:
        vis_img = original_img.copy()
        
        option_chars = ['A', 'B', 'C', 'D']  # 直接定义选项字符，不再依赖OPTION_MAP
    
        for q_num in range(1, 81):
            for opt_idx in range(4):
                opt_char = option_chars[opt_idx]
                
                # T/F模式下不再绘制选项框，只绘制T/F框
                if tf_mode:
                    # T/F模式：答案是字符串格式（如 "TFTF"）
                    tf_str = answers.get(q_num, 'FFFF')
                    if opt_idx < len(tf_str):
                        tf_val = tf_str[opt_idx]
                    else:
                        tf_val = 'F'
                    
                    # 绘制T/F框
                    t_coords = self.config.get_tf_cell_coords(q_num, opt_idx, is_true=True)
                    t_x, t_y, t_w, t_h = t_coords
                    f_coords = self.config.get_tf_cell_coords(q_num, opt_idx, is_true=False)
                    f_x, f_y, f_w, f_h = f_coords
                    
                    if tf_val == 'T':
                        # T被选中：绿色实心填充
                        overlay = vis_img.copy()
                        cv2.rectangle(overlay, (t_x, t_y), (t_x+t_w, t_y+t_h), self.tf_colors['T'], -1)
                        cv2.addWeighted(overlay, 0.4, vis_img, 0.6, 0, vis_img)
                        cv2.rectangle(vis_img, (t_x, t_y), (t_x+t_w, t_y+t_h), self.tf_colors['T'], 2)
                        cv2.putText(vis_img, 'T', (t_x + t_w//2 - 6, t_y + t_h//2 + 6),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # F框显示为红色边框
                        cv2.rectangle(vis_img, (f_x, f_y), (f_x+f_w, f_y+f_h), self.tf_colors['F'], 1)
                        cv2.putText(vis_img, 'F', (f_x + f_w//2 - 6, f_y + f_h//2 + 6),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.tf_colors['F'], 1)
                    elif tf_val == 'E':
                        # E表示模糊：T/F框都显示青色边框，在T/F框中间标注'E'
                        cv2.rectangle(vis_img, (t_x, t_y), (t_x+t_w, t_y+t_h), self.tf_colors['E'], 2)
                        cv2.putText(vis_img, 'E', (t_x + t_w//2 - 6, t_y + t_h//2 + 6),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.tf_colors['E'], 2)
                        
                        cv2.rectangle(vis_img, (f_x, f_y), (f_x+f_w, f_y+f_h), self.tf_colors['E'], 2)
                        cv2.putText(vis_img, 'E', (f_x + f_w//2 - 6, f_y + f_h//2 + 6),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.tf_colors['E'], 2)
                    else:
                        # F被选中：红色实心填充
                        overlay = vis_img.copy()
                        cv2.rectangle(overlay, (f_x, f_y), (f_x+f_w, f_y+f_h), self.tf_colors['F'], -1)
                        cv2.addWeighted(overlay, 0.4, vis_img, 0.6, 0, vis_img)
                        cv2.rectangle(vis_img, (f_x, f_y), (f_x+f_w, f_y+f_h), self.tf_colors['F'], 2)
                        cv2.putText(vis_img, 'F', (f_x + f_w//2 - 6, f_y + f_h//2 + 6),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # T框显示为绿色边框
                        cv2.rectangle(vis_img, (t_x, t_y), (t_x+t_w, t_y+t_h), self.tf_colors['T'], 1)
                        cv2.putText(vis_img, 'T', (t_x + t_w//2 - 6, t_y + t_h//2 + 6),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.tf_colors['T'], 1)
                    
                else:
                    # 传统模式
                    is_answered = False
                    options_for_q = answers.get(q_num, [])
                    
                    if opt_char in options_for_q:
                        is_answered = True
                        color = self.option_colors[opt_char]
                        
                        # 已填涂：鲜艳实心填充(30%透明度) + 粗边框(3px)
                        overlay = vis_img.copy()
                        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
                        cv2.addWeighted(overlay, 0.35, vis_img, 0.65, 0, vis_img)
                        cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 3)
                        
                        # 标注选项字母（白色粗字）
                        cv2.putText(
                            vis_img, opt_char,
                            (x + w//2 - 8, y + h//2 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                        )
                        
                    elif show_unanswered:
                        # 未填涂：同色系浅色细边框(2px)，让网格结构清晰可见
                        light_color = self.option_colors_light[opt_char]
                        cv2.rectangle(vis_img, (x, y), (x+w, y+h), light_color, 2)
            
            # 标注题号（每行第一列上方）
            if show_question_numbers and q_num % 8 == 1:
                x, y, w, h = self.config.get_answer_cell_coords(q_num, 0)
                cv2.putText(
                    vis_img, f"Q{q_num}",
                    (x, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                    cv2.LINE_AA
                )
        
        # 图例和统计信息
        if tf_mode:
            self._draw_legend_tf(vis_img)
        else:
            self._draw_legend(vis_img)
        
        if tf_mode:
            # T/F模式统计
            t_count = sum(1 for tf_str in answers.values() if 'T' in tf_str)
        else:
            t_count = sum(1 for opts in answers.values() if opts)
        
        total_questions = 80
        stats_text = f"Answered: {t_count}/{total_questions} ({t_count/total_questions*100:.1f}%)"
        cv2.putText(
            vis_img, stats_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2,
            cv2.LINE_AA
        )
        
        if output_path:
            cv2.imencode('.jpg', vis_img)[1].tofile(output_path)
            print(f"[OK] Enhanced visualization saved: {output_path}")
        
        return vis_img
    
    def _draw_legend_tf(self, img: NDArray) -> None:
        """绘制T/F模式的图例"""
        start_x = 15
        start_y = img.shape[0] - 145
        
        # 半透明背景
        overlay = img.copy()
        cv2.rectangle(overlay, 
                     (start_x-8, start_y-28), 
                     (210, img.shape[0]-12), 
                     (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.rectangle(img, 
                     (start_x-8, start_y-28), 
                     (210, img.shape[0]-12), 
                     (180, 180, 180), 1)
        
        cv2.putText(img, "Legend (TF Mode):", (start_x, start_y-8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # T/F/E图例
        cv2.rectangle(img, (start_x, start_y+6), (start_x+24, start_y+26), self.tf_colors['T'], -1)
        cv2.putText(img, "T", (start_x+7, start_y+22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.rectangle(img, (start_x+35, start_y+6), (start_x+59, start_y+26), self.tf_colors['F'], -1)
        cv2.putText(img, "F", (start_x+42, start_y+22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.rectangle(img, (start_x+70, start_y+6), (start_x+94, start_y+26), self.tf_colors['E'], -1)
        cv2.putText(img, "E", (start_x+77, start_y+22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # 选项字母
        y_pos = start_y + 40
        cv2.putText(img, "Options:", (start_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        for i, (opt, color) in enumerate(self.option_colors.items()):
            cv2.rectangle(img, (start_x + i*28, y_pos+10), (start_x + i*28 + 20, y_pos+26), color, 2)
            cv2.putText(img, opt, (start_x + i*28 + 5, y_pos+24),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        y_pos = start_y + 70
        cv2.putText(img, "Format: A-T B-F C-T D-F", (start_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    
    def _draw_legend(self, img: NDArray) -> None:
        start_x = 15
        start_y = img.shape[0] - 145
        
        # 半透明背景
        overlay = img.copy()
        cv2.rectangle(overlay, 
                     (start_x-8, start_y-28), 
                     (210, img.shape[0]-12), 
                     (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.rectangle(img, 
                     (start_x-8, start_y-28), 
                     (210, img.shape[0]-12), 
                     (180, 180, 180), 1)
        
        cv2.putText(img, "Legend:", (start_x, start_y-8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        for i, (opt, color) in enumerate(self.option_colors.items()):
            y_pos = start_y + 18 + i * 24
            
            # 已填涂样式（实心）
            cv2.rectangle(img, (start_x, y_pos-14), (start_x+24, y_pos+6), color, -1)
            cv2.putText(img, opt, (start_x+7, y_pos-1),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # 未填涂样式（空心）
            light_color = self.option_colors_light[opt]
            cv2.rectangle(img, (start_x+35, y_pos-14), (start_x+59, y_pos+6), light_color, 2)
            cv2.putText(img, "□", (start_x+40, y_pos-1),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, light_color, 1, cv2.LINE_AA)
        
        # 说明文字
        y_pos = start_y + 18 + 4 * 24 + 5
        cv2.putText(img, "■ = Filled    □ = Blank", (start_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    
    def visualize_by_blocks(
        self,
        original_img: NDArray,
        answers: Dict[int, List[str]],
        output_path: Optional[str] = None
    ) -> NDArray:
        vis_img = original_img.copy()
        
        row_colors = [
            (255, 50, 50),     # 第1行 - 红
            (50, 255, 50),     # 第2行 - 绿
            (50, 50, 255),     # 第3行 - 蓝
            (255, 255, 50),    # 第4行 - 青
            (255, 50, 255),    # 第5行 - 品红
            (50, 255, 255),    # 第6行 - 黄
            (200, 100, 255),   # 第7行 - 紫
            (255, 180, 50),    # 第8行 - 橙
            (100, 200, 255),   # 第9行 - 天蓝
            (180, 255, 130),   # 第10行 - 黄绿
        ]
        
        for macro_row in range(10):
            block_y_start = self.config.ANS_START_Y + macro_row * self.config.BLOCK_DY - 12
            block_y_end = block_y_start + self.config.BLOCK_DY + 3 * self.config.OPT_DY + 24
            
            block_x_start = self.config.ANS_START_X - 12
            block_x_end = block_x_start + 8 * self.config.Q_DX + 24
            
            overlay = vis_img.copy()
            cv2.rectangle(overlay, 
                         (block_x_start, block_y_start), 
                         (block_x_end, block_y_end),
                         row_colors[macro_row], -1)
            cv2.addWeighted(overlay, 0.06, vis_img, 0.94, 0, vis_img)
            
            cv2.rectangle(vis_img,
                         (block_x_start, block_y_start),
                         (block_x_end, block_y_end),
                         row_colors[macro_row], 2)
            
            cv2.putText(vis_img, f"Row {macro_row+1}",
                       (block_x_start, block_y_start - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, row_colors[macro_row], 2, cv2.LINE_AA)
        
        vis_img = self.visualize_full_grid(vis_img, answers, show_question_numbers=False)
        
        if output_path:
            cv2.imencode('.jpg', vis_img)[1].tofile(output_path)
            print(f"[OK] Block visualization saved: {output_path}")
        
        return vis_img
    
    def create_comparison_view(
        self,
        original_img: NDArray,
        aligned_img: NDArray,
        binary_img: NDArray,
        answers: Dict[int, List[str]],
        output_path: Optional[str] = None
    ) -> NDArray:
        if len(binary_img.shape) == 2:
            binary_color = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        else:
            binary_color = binary_img.copy()
        
        binary_annotated = self.visualize_full_grid(
            binary_color, answers, show_unanswered=False
        )
        
        h, w = 800, 600
        
        orig_resized = cv2.resize(original_img, (w, h))
        aligned_resized = cv2.resize(aligned_img, (w, h))
        binary_resized = cv2.resize(binary_annotated, (w, h))
        
        comparison = np.hstack([orig_resized, aligned_resized, binary_resized])
        
        cv2.putText(comparison, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(comparison, "Aligned", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(comparison, "Binary + Results", (2*w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        if output_path:
            cv2.imencode('.jpg', comparison)[1].tofile(output_path)
            print(f"[OK] Comparison view saved: {output_path}")
        
        return comparison


if __name__ == "__main__":
    from config import create_default_config
    
    config = create_default_config()
    visualizer = EnhancedVisualizer(config)
    
    print("✅ EnhancedVisualizer 模块加载成功 (新答题卡格式)")
    print("使用方法:")
    print("  from enhanced_visualizer import EnhancedVisualizer")
    print("  viz = EnhancedVisualizer(config)")
    print("  result = viz.visualize_full_grid(image, answers, 'output.jpg')")