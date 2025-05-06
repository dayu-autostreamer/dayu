# utils/logger.py
import logging
import sys
from datetime import datetime
import os

def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    创建一个logger实例
    Args:
        name: logger的名称
        log_dir: 日志文件保存目录
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 如果logger已经有handler，就不添加新的handler
    if logger.handlers:
        return logger
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 添加控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 添加文件handler
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{current_time}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# 创建一个默认的logger
default_logger = setup_logger('default')