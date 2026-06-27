此仓库设计用于2026年青海省全国生物竞赛机读扫描用（已经实际验证使用），使用python语言，核心为opencv图像识别；前置所需文件为答题卡扫描文件（jpg格式），设置有中间文件便于迅速核查识别效果；编写者：李壮壮
准备文件
input文件夹：答题卡扫描件（jpg，600dpi），统一放于同一文件夹内
output：存放中间文件，可直接识别扫描偏离位
注意：扫描准确率唯一取决于放置答题卡时，是否准确放置于扫描仪中，在output中发现明显错误时，只需重新扫描文件

This repository is designed for the machine-readable scanning process used in the 2026 National Biology Competition in Qinghai Province (validated in actual use). It is written in Python, with OpenCV image recognition as its core. The prerequisite file is the scanned answer sheet (JPG format), and intermediate files are set up to facilitate rapid verification of the recognition results. Author: Li Zhuangzhuang.

Preparation of Files
input folder: Scanned answer sheets (JPG, 600dpi), all placed within the same folder.

output folder: Stores intermediate files, allowing for the direct identification of scanning alignment offsets.

Note
The accuracy of the scan depends entirely on whether the answer sheet was placed accurately into the scanner. If obvious errors are discovered in the output files, simply re-scan the corresponding document.
