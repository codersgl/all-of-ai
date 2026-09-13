"""仓库路径锚点：全部相对仓库根计算，因此从哪个目录运行脚本都一致。"""

from pathlib import Path

# utils/paths.py -> parents[1] 就是仓库根
ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "DataSets"
RESULTS_DIR = ROOT / "results"
