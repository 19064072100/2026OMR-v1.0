"""
ImagePreProcessor - OMR 图像预处理模块
负责文档对齐、透视变换、二值化等图像预处理操作
包含完整的异常处理和降级机制

关键改进：
- ENABLE_IMAGE_PREPROCESSING=False 时直接返回原图，不做任何变换
- 确保坐标参数与原始图像完全对应
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray

from config import OMRConfig


class ImagePreProcessor:
    """
    图像预处理器
    
    处理流程（仅在 ENABLE_IMAGE_PREPROCESSING=True 时执行）：
    1. 边缘检测与轮廓提取
    2. 文档四边形定位（透视变换）
    3. 图像标准化（统一尺寸）
    4. 灰度转换与二值化
    
    ENABLE_IMAGE_PREPROCESSING=False 时：
    - align_document() 直接返回原图副本
    - 保持原始尺寸和坐标精确性
    """
    
    def __init__(self, config: OMRConfig) -> None:
        self.config = config
        self._last_aligned_image: Optional[NDArray] = None
        self._used_perspective_transform: bool = False
    
    def align_document(self, img: NDArray) -> NDArray:
        """
        对齐文档（主入口方法）
        
        核心逻辑：
        - 如果 ENABLE_IMAGE_PREPROCESSING=False → 直接返回原图（不变形）
        - 如果 True → 尝试透视变换，失败则强制缩放
        
        Args:
            img: 输入的 BGR 彩色图像 (从 cv2.imread 获取)
            
        Returns:
            图像（原图或处理后的图）
        """
        if img is None or img.size == 0:
            raise ValueError("输入图像为空或无效")
        
        if not self.config.ENABLE_IMAGE_PREPROCESSING:
            # ⭐ 关键：禁用变换时直接返回原图副本
            # 这样坐标参数可以直接在原图上使用，无需考虑缩放比例
            print(f"   [OK] 使用原始图像 (尺寸: {img.shape[1]}×{img.shape[0]})")
            self._last_aligned_image = img.copy()
            self._used_perspective_transform = False
            return img.copy()
        
        try:
            aligned_img = self._try_perspective_transform(img)
            
            if aligned_img is not None:
                self._last_aligned_image = aligned_img
                self._used_perspective_transform = True
                return aligned_img
            
            print("[WARN] 未检测到文档边界，使用强制缩放模式")
            aligned_img = self._force_resize(img)
            self._last_aligned_image = aligned_img
            self._used_perspective_transform = False
            return aligned_img
            
        except Exception as e:
            print(f"[ERROR] 文档对齐过程出错: {str(e)}")
            print("→ 降级使用强制缩放")
            return self._force_resize(img)
    
    def _try_perspective_transform(self, img: NDArray) -> Optional[NDArray]:
        """尝试进行透视变换"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(
                blurred,
                self.config.CANNY_THRESHOLD1,
                self.config.CANNY_THRESHOLD2
            )
            
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (self.config.MORPH_KERNEL_SIZE, self.config.MORPH_KERNEL_SIZE)
            )
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(
                closed.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return None
            
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            doc_contour = contours_sorted[0]
            
            area = cv2.contourArea(doc_contour)
            img_area = img.shape[0] * img.shape[1]
            
            if area < self.config.MIN_CONTOUR_AREA:
                return None
            
            if area > img_area * self.config.MAX_CONTOUR_AREA_RATIO:
                return None
            
            peri = cv2.arcLength(doc_contour, True)
            approx = cv2.approxPolyDP(doc_contour, 0.02 * peri, True)
            
            if len(approx) != 4:
                return None
            
            warped = self.four_point_transform(img, approx.reshape(4, 2))
            
            resized = cv2.resize(
                warped,
                (self.config.STD_WIDTH, self.config.STD_HEIGHT),
                interpolation=cv2.INTER_LINEAR
            )
            
            return resized
            
        except Exception as e:
            return None
    
    def _force_resize(self, img: NDArray) -> NDArray:
        """强制缩放（降级方案）"""
        resized = cv2.resize(
            img,
            (self.config.STD_WIDTH, self.config.STD_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )
        return resized
    
    @staticmethod
    def order_points(pts: NDArray) -> NDArray:
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    @staticmethod
    def four_point_transform(image: NDArray, pts: NDArray) -> NDArray:
        rect = ImagePreProcessor.order_points(pts)
        (tl, tr, br, bl) = rect
        
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = int(max(width_a, width_b))
        
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = int(max(height_a, height_b))
        
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        
        warped = cv2.warpPerspective(
            image, M, (max_width, max_height),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return warped
    
    def get_binary_image(self, aligned_img: NDArray) -> NDArray:
        """
        将对齐后的彩色图像转换为二值图
        """
        if aligned_img is None or aligned_img.size == 0:
            raise ValueError("输入图像为空或无效")
        
        try:
            gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
            
            _, binary_otsu = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            return binary_otsu
            
        except Exception as e:
            print(f"[ERROR] 二值化处理失败: {str(e)}")
            try:
                gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(
                    gray,
                    self.config.BINARY_THRESH,
                    255,
                    cv2.THRESH_BINARY
                )
                return binary
            except:
                raise RuntimeError("所有二值化方法均失败")
    
    def get_last_aligned_image(self) -> Optional[NDArray]:
        """获取最近一次对齐后的图像（用于调试）"""
        return self._last_aligned_image
    
    def was_perspective_transform_used(self) -> bool:
        """检查上次处理是否使用了透视变换"""
        return self._used_perspective_transform


def load_image_safe(path: str) -> Tuple[Optional[NDArray], Optional[str]]:
    """
    安全加载图像文件
    """
    try:
        if not path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            return None, f"不支持的图像格式: {path}"
        
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return None, f"无法读取图像文件（可能损坏或路径错误）: {path}"
        
        return img, None
        
    except Exception as e:
        return None, f"加载图像时发生异常: {str(e)}"


def save_debug_image(img: NDArray, output_path: str) -> bool:
    """保存调试图像"""
    try:
        cv2.imwrite(output_path, img)
        return True
    except Exception as e:
        print(f"[ERROR] 保存调试图像失败: {output_path}, 错误: {str(e)}")
        return False


if __name__ == "__main__":
    from config import create_default_config
    
    config = create_default_config()
    preprocessor = ImagePreProcessor(config)
    
    print("ImagePreProcessor 测试")
    print("=" * 50)
    print(f"配置信息:")
    print(f"  标准尺寸: {config.STD_WIDTH} x {config.STD_HEIGHT}")
    print(f"  启用预处理: {config.ENABLE_IMAGE_PREPROCESSING}")
    print("\n[OK] 模块加载成功！")