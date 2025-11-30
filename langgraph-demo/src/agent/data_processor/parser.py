"""文档解析器，支持PDF/DOCX/PPT等多种格式（已弃用，请使用FileManager）"""

import logging
from typing import Dict, Any
from pathlib import Path

from agent.data_processor.file_manager import FileManager

logger = logging.getLogger(__name__)


class DocumentParser:
    """文档解析器（兼容性包装类，实际使用FileManager）
    
    支持多种格式：
    - PDF: 使用pdfplumber（更好的中文支持）
    - DOCX/DOC: 使用python-docx和docx2python
    - PPTX/PPT: 使用python-pptx
    - Excel: 使用pandas/openpyxl
    - TXT: 自动检测编码
    """
    
    @staticmethod
    def parse_document(file_path: str) -> Dict[str, Any]:
        """根据文件扩展名自动选择解析器（使用FileManager）
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析后的文档内容
        """
        return FileManager.parse_file(file_path)


# 便捷函数
def parse_document(file_path: str) -> Dict[str, Any]:
    """解析文档的便捷函数"""
    return DocumentParser.parse_document(file_path)


