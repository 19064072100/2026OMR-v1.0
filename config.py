"""
OMRConfig - OMR 答题卡识别系统配置模块
集中管理所有坐标参数、阈值常量和路径配置
支持从 JSON 文件加载自定义配置
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class OMRConfig:
    STD_WIDTH: int = 4379           # 实际图像宽度（自动检测）
    STD_HEIGHT: int = 6167          # 实际图像高度（自动检测）
    FILL_THRESHOLD: float = 0.425   # 填涂判定阈值
    
    ENABLE_IMAGE_PREPROCESSING: bool = False
    
    # ==================== 准考证号区参数 (9列 × 10行) ====================
    ID_START_X: int = 3021          # 准考证号第1位-数字0的X坐标
    ID_START_Y: int = 552           # 准考证号第1位-数字0的Y坐标
    ID_COL_DX: int = 120            # 列间距（水平方向，9位之间）
    ID_ROW_DY: int = 83            # 行间距（垂直方向，0-9之间）
    ID_CELL_WIDTH: int = 75        # 单个填涂框宽度
    ID_CELL_HEIGHT: int = 35        # 单个填涂框高度
    
    # ==================== 答题区参数 (80题) ====================
    ANS_START_X: int = 392          # 第1题-A选项T框的X坐标（直接定义T框位置）
    ANS_START_Y: int = 1722       # 第1题-A选项T框的Y坐标
    
    Q_DX: int = 480               # 题与题之间的水平间距
    OPT_DY: int = 82               # 选项之间的垂直间距（A→B→C→D）
    BLOCK_DX: int = 480            # 列间距（等同于Q_DX）
    BLOCK_DY: int = 413             # 行间距（每行题目之间的垂直距离）
    
    OPT_CELL_WIDTH: int = 70       # T/F框宽度
    OPT_CELL_HEIGHT: int = 42       # T/F框高度
    
    # ==================== T/F 检测参数 ====================
    TF_DX: int = 125                 # T框和F框之间的水平间距（T在左，F在右）
    TF_DY: int = 0                  # T框和F框之间的垂直间距（同一行，通常为0）
    TF_MODE: bool = True            # 是否启用T/F模式
    
    # 布局说明：
    # 每个题目(80题)有4个选项位置(A/B/C/D)，纵向排列
    # 每个选项位置只有T和F两个框，水平排列（T在左，F在右）
    # 坐标计算：
    #   T框X = ANS_START_X + col*Q_DX
    #   F框X = T框X + TF_DX
    #   Y = ANS_START_Y + row*BLOCK_DY + opt_idx*OPT_DY
    
    EXCEL_PATH: str = "input/student_mapping.xlsx"
    OUTPUT_DIR: str = "output"
    OUTPUT_CSV_PATH: str = "output/results.csv"
    
    CANNY_THRESHOLD1: int = 50
    CANNY_THRESHOLD2: int = 150
    MORPH_KERNEL_SIZE: int = 5
    BINARY_THRESH: int = 127
    ADAPTIVE_THRESH_BLOCK_SIZE: int = 11
    ADAPTIVE_THRESH_C: int = 2
    
    MIN_CONTOUR_AREA: float = 50000.0
    MAX_CONTOUR_AREA_RATIO: float = 0.95
    
    def __post_init__(self) -> None:
        self._validate_config()
    
    def _validate_config(self) -> None:
        assert 0 < self.FILL_THRESHOLD <= 1.0, \
            f"FILL_THRESHOLD 必须在 (0, 1] 范围内，当前值: {self.FILL_THRESHOLD}"
        
        assert self.STD_WIDTH > 0 and self.STD_HEIGHT > 0, \
            f"标准尺寸必须为正数: {self.STD_WIDTH}x{self.STD_HEIGHT}"
        
        assert self.ID_START_X >= 0 and self.ID_START_Y >= 0, \
            f"准考证号区起始坐标不能为负数"
        
        assert self.ANS_START_X >= 0 and self.ANS_START_Y >= 0, \
            f"答题区起始坐标不能为负数"
    
    @classmethod
    def load_custom_config(cls, path: str) -> 'OMRConfig':
        config_path = Path(path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data: Dict[str, Any] = json.load(f)
            
            return cls(**config_data)
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"配置文件 JSON 格式错误: {path}",
                e.doc,
                e.pos
            )
    
    def save_to_json(self, path: str) -> None:
        import dataclasses
        
        config_dict = dataclasses.asdict(self)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
    
    def get_id_grid_coords(self, col: int, row: int) -> tuple:
        """
        计算准考证号区指定单元格的坐标
        
        Args:
            col: 列索引 (0-8，代表9位准考证号)
            row: 行索引 (0-9，代表数字0-9)
            
        Returns:
            (x, y, width, height) 元组
        """
        if not (0 <= col < 9 and 0 <= row < 10):
            raise ValueError(f"准考证号区坐标越界: col={col}, row={row}")
        
        x = self.ID_START_X + col * self.ID_COL_DX
        y = self.ID_START_Y + row * self.ID_ROW_DY
        
        return (x, y, self.ID_CELL_WIDTH, self.ID_CELL_HEIGHT)
    
    def get_answer_cell_coords(self, question_num: int, option_idx: int) -> tuple:
        """
        计算80题答题区中指定题目选项的坐标
        
        布局结构：10行 × 8列 = 80题
        - 每行包含8道题（横向排列）
        - 每题4个选项（纵向排列：A→B→C→D）
        
        Args:
            question_num: 题号 (1-80)
            option_idx: 选项索引 (0-3，对应 A/B/C/D)
            
        Returns:
            (x, y, width, height) 元组
        """
        if not (1 <= question_num <= 80):
            raise ValueError(f"题号必须在 1-80 范围内，当前值: {question_num}")
        
        if not (0 <= option_idx < 4):
            raise ValueError(f"选项索引必须在 0-3 范围内，当前值: {option_idx}")
        
        q_idx = question_num - 1
        
        row = q_idx // 8             # 所在行 (0-9)
        col = q_idx % 8              # 所在列 (0-7)
        
        q_x = self.ANS_START_X + col * self.Q_DX
        opt_y = self.ANS_START_Y + row * self.BLOCK_DY + option_idx * self.OPT_DY
        
        return (q_x, opt_y, self.OPT_CELL_WIDTH, self.OPT_CELL_HEIGHT)
    
    def get_tf_cell_coords(self, question_num: int, option_idx: int, is_true: bool) -> tuple:
        """
        计算T/F选择框的坐标
        
        布局结构：每个题目有4个选项位置(A/B/C/D)，每个位置只有T和F两个选择框
        - 每行8题，共10行 = 80题
        - 每题4个选项位置(A→B→C→D)，纵向排列
        - 每个选项位置有T和F两个框，水平排列（T在左，F在右）
        
        Args:
            question_num: 题号 (1-80)
            option_idx: 选项索引 (0-3，对应 A/B/C/D)
            is_true: True表示T框，False表示F框
            
        Returns:
            (x, y, width, height) 元组
        """
        if not (1 <= question_num <= 80):
            raise ValueError(f"题号必须在 1-80 范围内，当前值: {question_num}")
        
        if not (0 <= option_idx < 4):
            raise ValueError(f"选项索引必须在 0-3 范围内，当前值: {option_idx}")
        
        q_idx = question_num - 1
        row = q_idx // 8          # 所在行 (0-9)
        col = q_idx % 8           # 所在列 (0-7)
        
        # T框起始位置（直接从ANS_START_X开始，不再加选项框宽度）
        t_x = self.ANS_START_X + col * self.Q_DX
        t_y = self.ANS_START_Y + row * self.BLOCK_DY + option_idx * self.OPT_DY
        
        if is_true:
            return (t_x, t_y, self.OPT_CELL_WIDTH, self.OPT_CELL_HEIGHT)
        else:
            return (t_x + self.TF_DX, t_y, self.OPT_CELL_WIDTH, self.OPT_CELL_HEIGHT)
    
    def get_debug_info(self) -> str:
        info_lines = [
            "=" * 60,
            "OMR 系统配置参数 (新答题卡)",
            "=" * 60,
            f"\n【全局参数】",
            f"  标准化尺寸: {self.STD_WIDTH} x {self.STD_HEIGHT}",
            f"  填涂阈值: {self.FILL_THRESHOLD}",
            
            f"\n【准考证号区】(9列×10行)",
            f"  起始位置: ({self.ID_START_X}, {self.ID_START_Y})",
            f"  列间距: {self.ID_COL_DX}px | 行间距: {self.ID_ROW_DY}px",
            f"  单元格尺寸: {self.ID_CELL_WIDTH}x{self.ID_CELL_HEIGHT}",
            
            f"\n【答题区】(80题 - 10行×8列布局)",
            f"  起始位置: ({self.ANS_START_X}, {self.ANS_START_Y})",
            f"  题目水平间距(Q_DX): {self.Q_DX}px",
            f"  选项垂直间距(OPT_DY): {self.OPT_DY}px",
            f"  行间距(BLOCK_DY): {self.BLOCK_DY}px",
            
            f"\n【选项映射】",
            f"  {self.OPTION_MAP}",
            
            f"\n【T/F模式】",
            f"  启用: {'是' if self.TF_MODE else '否'}",
            f"  T/F间距(TF_DX): {self.TF_DX}px",
            
            f"\n【数据文件】",
            f"  学生名单: {self.EXCEL_PATH}",
            f"  输出CSV: {self.OUTPUT_CSV_PATH}",
            
            "\n" + "=" * 60,
            "坐标测量指南（使用 Windows 画图工具）：",
            "=" * 60,
            "1. 打开答题卡图片",
            "2. 测量准考证号左上角第一个框(数字0) → 设置 ID_START_X/Y",
            "3. 测量第1题[A]选项 → 设置 ANS_START_X/Y", 
            "4. 测量第2题[A] → 计算 Q_DX = X2 - X1",
            "5. 测量第9题[A] → 计算 BLOCK_DY = Y9 - Y1",
            "6. 测量T/F框间距 → 设置 TF_DX",
            "=" * 60,
        ]
        
        return "\n".join(info_lines)


def create_default_config() -> OMRConfig:
    return OMRConfig()


def create_high_resolution_config() -> OMRConfig:
    base = OMRConfig()
    
    scale_factor = 1.5
    
    return OMRConfig(
        STD_WIDTH=int(base.STD_WIDTH * scale_factor),
        STD_HEIGHT=int(base.STD_HEIGHT * scale_factor),
        FILL_THRESHOLD=base.FILL_THRESHOLD,
        ID_START_X=int(base.ID_START_X * scale_factor),
        ID_START_Y=int(base.ID_START_Y * scale_factor),
        ID_COL_DX=int(base.ID_COL_DX * scale_factor),
        ID_ROW_DY=int(base.ID_ROW_DY * scale_factor),
        ID_CELL_WIDTH=int(base.ID_CELL_WIDTH * scale_factor),
        ID_CELL_HEIGHT=int(base.ID_CELL_HEIGHT * scale_factor),
        ANS_START_X=int(base.ANS_START_X * scale_factor),
        ANS_START_Y=int(base.ANS_START_Y * scale_factor),
        Q_DX=int(base.Q_DX * scale_factor),
        OPT_DY=int(base.OPT_DY * scale_factor),
        BLOCK_DX=int(base.BLOCK_DX * scale_factor),
        BLOCK_DY=int(base.BLOCK_DY * scale_factor),
        OPT_CELL_WIDTH=int(base.OPT_CELL_WIDTH * scale_factor),
        OPT_CELL_HEIGHT=int(base.OPT_CELL_HEIGHT * scale_factor),
        EXCEL_PATH=base.EXCEL_PATH,
        OUTPUT_CSV_PATH=base.OUTPUT_CSV_PATH,
        OPTION_MAP=base.OPTION_MAP.copy()
    )


if __name__ == "__main__":
    config = create_default_config()
    print(config.get_debug_info())
    
    print("\n\n===== 坐标计算测试 =====")
    
    print("\n【准考证号区测试】")
    for col in range(3):
        coords = config.get_id_grid_coords(col, 0)
        print(f"  第{col+1}位(数字0): x={coords[0]}, y={coords[1]}")
    
    print("\n【答题区关键点位测试】")
    test_questions = [1, 2, 8, 9, 40, 41, 79, 80]
    for q in test_questions:
        coords_a = config.get_answer_cell_coords(q, 0)
        coords_d = config.get_answer_cell_coords(q, 3)
        print(f"  Q{q:2d}: A({coords_a[0]:4d},{coords_a[1]:4d}) ~ D({coords_d[0]:4d},{coords_d[1]:4d})")