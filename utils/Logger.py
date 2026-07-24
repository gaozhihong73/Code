import logging
import sys
from datetime import datetime
from pathlib import Path


class Logger:
    """双输出日志器，同时写入文件和控制台

    使用示例:
        logger = Logger("logs/app.log")
        logger.info("开始训练")
    """

    def __init__(
        self,
        log_filename: str,
        file_level: str = "DEBUG",
        console_level: str = "INFO",
        unique: bool = False,
        max_bytes: int | None = None,
        backup_count: int = 5,
        encoding: str = "utf-8",
    ):
        self.file_level = self._get_level(file_level)
        self.console_level = self._get_level(console_level)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding

        # 处理唯一文件名
        if unique:
            file_path = Path(log_filename)
            timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            self.log_filename = str(
                file_path.parent / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            )
        else:
            self.log_filename = log_filename

        # 确保日志目录存在
        Path(self.log_filename).parent.mkdir(parents=True, exist_ok=True)

        # 创建 logger 实例
        self.logger = logging.getLogger(
            f"Logger_{Path(self.log_filename).resolve()}_{id(self)}"
        )
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 文件 handler
        if self.max_bytes:
            from logging.handlers import RotatingFileHandler

            self.file_handler = RotatingFileHandler(
                self.log_filename,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding=self.encoding,
            )
        else:
            self.file_handler = logging.FileHandler(
                self.log_filename, encoding=self.encoding, mode="a"
            )
        self.file_handler.setLevel(self.file_level)
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

        # 控制台 handler
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(self.console_level)
        self.console_handler.setFormatter(formatter)
        self.logger.addHandler(self.console_handler)

    @staticmethod
    def _get_level(level_str: str) -> int:
        """将字符串级别转换为 logging 常量，未知级别降级为 INFO"""
        level = getattr(logging, level_str.upper(), None)
        if isinstance(level, int):
            return level
        return logging.INFO

    def _log(self, level: int, msg, *args, stacklevel: int = 2, **kwargs) -> None:
        """统一日志记录入口"""
        try:
            self.logger.log(level, msg, *args, stacklevel=stacklevel, **kwargs)
        except Exception as e:
            print(f"记录日志失败: {e}")

    def debug(self, message: str) -> None:
        """记录 DEBUG 级别信息"""
        self._log(logging.DEBUG, str(message), stacklevel=2)

    def info(self, msg, *args, **kwargs) -> None:
        """记录 INFO 级别信息，支持原生 logging 的 args/kwargs"""
        self._log(logging.INFO, msg, *args, stacklevel=2, **kwargs)

    def warning(self, message: str) -> None:
        """记录 WARNING 级别信息"""
        self._log(logging.WARNING, str(message), stacklevel=2)

    def error(self, msg, *args, **kwargs) -> None:
        """记录 ERROR 级别信息，支持 exc_info=True 等原生参数"""
        self._log(logging.ERROR, msg, *args, stacklevel=2, **kwargs)

    def critical(self, message: str) -> None:
        """记录 CRITICAL 级别信息"""
        self._log(logging.CRITICAL, str(message), stacklevel=2)

    def set_file_level(self, level: str) -> None:
        """动态调整文件日志级别"""
        new_level = self._get_level(level)
        self.file_handler.setLevel(new_level)
        self.file_level = new_level
        self.info(f"文件日志级别已调整为: {level}")

    def set_console_level(self, level: str) -> None:
        """动态调整控制台日志级别"""
        new_level = self._get_level(level)
        self.console_handler.setLevel(new_level)
        self.console_level = new_level
        self.info(f"控制台日志级别已调整为: {level}")

    def flush(self) -> None:
        """强制刷新所有处理器的缓冲区"""
        for handler in self.logger.handlers:
            handler.flush()
