"""数据处理模块 - 基于 deep-searcher loader 实现"""

from agent.data_processor.file_loader import FileLoader, get_file_loader
from agent.data_processor.text_splitter import TextSplitter, get_text_splitter

# 保持向后兼容
from agent.data_processor.file_manager import FileManager
from agent.data_processor.chunker import chunk_documents

__all__ = [
    "FileLoader",
    "get_file_loader",
    "TextSplitter",
    "get_text_splitter",
    "FileManager",  # 向后兼容
    "chunk_documents",  # 向后兼容
]


