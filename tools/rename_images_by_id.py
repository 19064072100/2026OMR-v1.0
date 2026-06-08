"""
Image Renaming Tool - 根据OMR识别结果重命名图片
==================================================

功能：从CSV结果文件读取学号，将原始图片重命名为对应学号

使用方法：
    python rename_images_by_id.py --csv results.csv --input ./input --output ./output/renamed
    python rename_images_by_id.py --csv results.csv  # 使用默认路径
"""

import argparse
import os
import shutil
from typing import Optional, Dict

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[ERROR] pandas not installed! Please install: pip install pandas")
    exit(1)


def load_csv_results(csv_path: str) -> Dict[str, str]:
    """
    从CSV文件加载学号和图片路径映射
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        字典: {学号: 原始图片路径}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        if 'Student_ID' not in df.columns:
            raise ValueError("CSV文件缺少 'Student_ID' 列")
        
        if 'Image_Path' not in df.columns:
            raise ValueError("CSV文件缺少 'Image_Path' 列")
        
        mapping = {}
        for _, row in df.iterrows():
            student_id = str(row['Student_ID']).strip()
            image_path = str(row['Image_Path']).strip()
            
            if student_id and image_path:
                mapping[student_id] = image_path
        
        print(f"[OK] 从CSV加载了 {len(mapping)} 条记录")
        return mapping
        
    except Exception as e:
        raise RuntimeError(f"读取CSV失败: {str(e)}")


def rename_images(
    mapping: Dict[str, str],
    input_dir: Optional[str] = None,
    output_dir: str = "./output/renamed",
    copy_only: bool = False
) -> int:
    """
    根据学号重命名图片
    
    Args:
        mapping: 学号到图片路径的映射
        input_dir: 输入目录（如果为空，使用CSV中的绝对路径）
        output_dir: 输出目录
        copy_only: 是否只复制不移动
        
    Returns:
        成功重命名的图片数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failed_count = 0
    
    print(f"\n{'='*60}")
    print(f"[START] 图片重命名")
    print(f"{'='*60}")
    print(f"  输入目录: {'(使用CSV路径)' if input_dir is None else input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  模式: {'复制' if copy_only else '移动'}")
    print(f"  待处理: {len(mapping)} 张图片")
    print(f"{'='*60}")
    
    for student_id, orig_path in mapping.items():
        # 获取原始文件名和扩展名
        if input_dir:
            # 使用输入目录中的文件
            filename = os.path.basename(orig_path)
            source_path = os.path.join(input_dir, filename)
        else:
            # 使用CSV中的完整路径
            source_path = orig_path
        
        # 检查源文件是否存在
        if not os.path.exists(source_path):
            print(f"[SKIP] 文件不存在: {source_path}")
            failed_count += 1
            continue
        
        # 获取文件扩展名
        ext = os.path.splitext(source_path)[1].lower()
        
        # 生成新文件名
        new_filename = f"{student_id}{ext}"
        new_path = os.path.join(output_dir, new_filename)
        
        # 处理重复文件名
        counter = 1
        while os.path.exists(new_path):
            new_filename = f"{student_id}_{counter}{ext}"
            new_path = os.path.join(output_dir, new_filename)
            counter += 1
        
        try:
            if copy_only:
                shutil.copy2(source_path, new_path)
                print(f"[COPY] {os.path.basename(source_path)} -> {new_filename}")
            else:
                shutil.move(source_path, new_path)
                print(f"[MOVE] {os.path.basename(source_path)} -> {new_filename}")
            
            success_count += 1
            
        except Exception as e:
            print(f"[ERROR] 处理失败 {os.path.basename(source_path)}: {str(e)}")
            failed_count += 1
    
    print(f"\n{'='*60}")
    print(f"[DONE] 处理完成")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"{'='*60}")
    
    return success_count


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据OMR识别结果重命名图片",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--csv', '-c', 
        type=str, 
        required=True,
        help='CSV结果文件路径'
    )
    
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        default=None,
        help='图片输入目录（可选，不指定则使用CSV中的路径）'
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        default='./output/renamed',
        help='重命名后图片的输出目录'
    )
    
    parser.add_argument(
        '--copy', '-cp', 
        action='store_true',
        help='仅复制不移动（保留原文件）'
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    print("""
╔══════════════════════════════════════════════════════════╗
║              Image Renaming Tool v1.0                    ║
║          根据OMR识别结果重命名图片为学号                   ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # 加载CSV映射
        mapping = load_csv_results(args.csv)
        
        if not mapping:
            print("[WARN] CSV中没有有效数据")
            return
        
        # 执行重命名
        success_count = rename_images(
            mapping=mapping,
            input_dir=args.input,
            output_dir=args.output,
            copy_only=args.copy
        )
        
        if success_count > 0:
            print(f"\n[OK] 成功重命名 {success_count} 张图片到 {args.output}")
        else:
            print("\n[WARN] 没有成功重命名任何图片")
            
    except Exception as e:
        print(f"\n[FATAL] 错误: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()