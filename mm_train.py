import os
import sys
import time
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.logging import print_log


sys.path.append(os.path.join(os.path.dirname(__file__), 'mmdetection'))


# ========= 用户设置部分 =========
MODEL_TYPE = 'mask_rcnn'  # 可选: 'mask_rcnn' 或 'sparse_rcnn'
CONFIG_DIR = os.path.join('mmdetection', 'configs')
WORK_DIR = 'work_dirs'

def get_config_path(model_type):
    if model_type == 'mask_rcnn':
        return os.path.join(CONFIG_DIR, 'mask_rcnn', 'mask-rcnn_r50_fpn_1x_voc.py')
    elif model_type == 'sparse_rcnn':
        return os.path.join(CONFIG_DIR, 'sparse_rcnn', 'sparse-rcnn_r50_fpn_voc.py')
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def main():
    config_path = get_config_path(MODEL_TYPE)
    abs_config_path = os.path.abspath(config_path)
    print_log(f"配置文件绝对路径: {abs_config_path}", 'current')

    if not os.path.exists(abs_config_path):
        raise FileNotFoundError(f"配置文件不存在: {abs_config_path}")

    cfg = Config.fromfile(abs_config_path)

    if not hasattr(cfg, 'work_dir'):
        cfg.work_dir = os.path.join(WORK_DIR, f"{MODEL_TYPE}_voc")

    os.makedirs(cfg.work_dir, exist_ok=True)

    print_log(f"\n使用配置文件: {abs_config_path}", 'current')
    print_log(f"输出目录: {cfg.work_dir}", 'current')

    runner = Runner.from_cfg(cfg)

    start_time = time.time()
    runner.train()
    print_log(f"\n训练完成，总耗时: {time.time() - start_time:.2f} 秒", 'current')

if __name__ == '__main__':
    main()
