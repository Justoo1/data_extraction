import os
import tempfile
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextContainer
import logging

logger = logging.getLogger(__name__)

class PDFService:
    """Service for interacting with PDF documents"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """
        Extract text from a PDF file using PDFMiner
        Returns a dictionary with page numbers as keys and text content as values
        """
        try:
            page_texts = {}
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        page_texts[i + 1] = text  # 1-indexed pages for user-friendliness
            
            return page_texts
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    @staticmethod
    def extract_text_by_pages(pdf_path, pages):
        """
        Extract text from specific pages in a PDF
        pages: List of page numbers (1-indexed)
        """
        try:
            result = {}
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                total_pages = len(reader.pages)
                
                for page_num in pages:
                    # Convert to 0-indexed for PyPDF2
                    idx = page_num - 1
                    
                    if 0 <= idx < total_pages:
                        text = reader.pages[idx].extract_text()
                        result[page_num] = text
                    else:
                        logger.warning(f"Page {page_num} is out of range (total pages: {total_pages})")
            
            return result
        except Exception as e:
            logger.error(f"Error extracting specific pages from PDF: {str(e)}")
            raise
    
    @staticmethod
    def extract_tables_from_pdf(pdf_path, pages=None):
        """
        Extract tables from a PDF using a combination of libraries
        This is a placeholder - actual implementation would depend on the structure of your PDFs
        Consider using libraries like tabula-py or camelot-py for table extraction
        """
        try:
            # This is just a skeleton - actual table extraction is more complex
            # and would require a specialized library based on your specific PDFs
            return {"tables_extracted": "Placeholder for table extraction"}
        except Exception as e:
            logger.error(f"Error extracting tables from PDF: {str(e)}")
            raise
    
    @staticmethod
    def get_pdf_metadata(pdf_path):
        """Extract metadata from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                info = reader.metadata
                
                # Count pages
                num_pages = len(reader.pages)
                
                metadata = {
                    'title': info.title if info.title else None,
                    'author': info.author if info.author else None,
                    'subject': info.subject if info.subject else None,
                    'creator': info.creator if info.creator else None,
                    'producer': info.producer if info.producer else None,
                    'page_count': num_pages
                }
                
                return metadata
        except Exception as e:
            logger.error(f"Error getting PDF metadata: {str(e)}")
            raise