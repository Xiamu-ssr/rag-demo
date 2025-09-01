"""日志配置模块"""

import sys
from loguru import logger
from app.core.config import settings


def setup_logging():
    """配置应用日志"""
    # 移除默认处理器
    logger.remove()
    
    # 控制台日志格式
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    # 添加控制台处理器
    logger.add(
        sys.stdout,
        format=console_format,
        level="DEBUG" if settings.debug else "INFO",
        colorize=True
    )
    
    # 添加文件处理器
    logger.add(
        "logs/app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    # 错误日志单独文件
    logger.add(
        "logs/error.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    return logger


# 创建日志目录
import os
os.makedirs("logs", exist_ok=True)

# 初始化日志
app_logger = setup_logging()