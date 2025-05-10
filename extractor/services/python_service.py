import os
import logging
import tempfile
from typing import Dict, List, Optional, Tuple, Union
import fitz  # PyMuPDF
import pdfplumber
import camelot
import cv2
import numpy as np
import pytesseract
from PIL import Image
import pandas as pd
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ExtractionMethod(Enum):
    """Available extraction methods for different PDF structures"""
    PDFPLUMBER = "pdfplumber"
    PYMUPDF = "pymupdf"
    CAMELOT = "camelot"
    OCR = "ocr"
    HYBRID = "hybrid"

@dataclass
class TableStructure:
    """Represents a detected table structure"""
    page_number: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    data: pd.DataFrame
    confidence: float
    extraction_method: ExtractionMethod

@dataclass
class DomainData:
    """Represents data extracted for a specific SENDIG domain"""
    domain_code: str
    pages: List[int]
    tables: List[TableStructure]
    text_blocks: List[str]
    extraction_confidence: float

class EnhancedPDFService:
    """Advanced PDF service with multiple extraction strategies"""
    
    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr
        # Configure tesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
    def extract_text_with_layout(self, pdf_path: str) -> Dict[int, Dict]:
        """Extract text with layout information from PDF"""
        results = {}
        
        # Using pdfplumber for better layout detection
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_data = {
                    'text': page.extract_text(),
                    'tables': self._extract_tables_pdfplumber(page),
                    'words': page.extract_words(),
                    'layout': self._analyze_layout(page)
                }
                results[page_num] = page_data
        
        # Fallback to PyMuPDF for pages with issues
        for page_num in results:
            if not results[page_num]['text'] or not results[page_num]['tables']:
                results[page_num].update(self._extract_with_pymupdf(pdf_path, page_num))
        
        return results
    
    def _extract_tables_pdfplumber(self, page) -> List[pd.DataFrame]:
        """Extract tables using pdfplumber"""
        tables = []
        try:
            # Extract tables with different settings for robustness
            detected_tables = page.extract_tables()
            for table_data in detected_tables:
                if table_data:
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    tables.append(df)
            
            # Try with different table settings if no tables found
            if not tables:
                settings = {
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 3,
                    "join_tolerance": 3
                }
                detected_tables = page.extract_tables(**settings)
                for table_data in detected_tables:
                    if table_data:
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        tables.append(df)
        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed: {str(e)}")
        
        return tables
    
    def _extract_with_pymupdf(self, pdf_path: str, page_num: int) -> Dict:
        """Extract data using PyMuPDF as fallback"""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]
            
            # Extract text with layout
            text_instances = page.get_text("dict")
            
            # Extract tables
            tables = page.find_tables()
            table_dfs = []
            for table in tables:
                df = table.to_pandas()
                table_dfs.append(df)
            
            return {
                'text': page.get_text(),
                'tables': table_dfs,
                'layout': text_instances
            }
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {str(e)}")
            return {'text': '', 'tables': [], 'layout': {}}
    
    def _analyze_layout(self, page) -> Dict:
        """Analyze page layout to understand structure"""
        words = page.extract_words()
        
        # Group words by lines based on y-coordinate
        lines = {}
        for word in words:
            y = round(word['top'])
            if y not in lines:
                lines[y] = []
            lines[y].append(word)
        
        # Sort lines by y-coordinate
        sorted_lines = dict(sorted(lines.items()))
        
        # Detect columns based on x-coordinates
        columns = self._detect_columns(words)
        
        return {
            'lines': sorted_lines,
            'columns': columns,
            'word_count': len(words),
            'estimated_structure': self._estimate_structure_type(lines, columns)
        }
    
    def _detect_columns(self, words: List[Dict]) -> List[Tuple[float, float]]:
        """Detect column boundaries in the page"""
        if not words:
            return []
        
        # Get all x-coordinates of word starts
        x_coords = [word['x0'] for word in words]
        x_coords = sorted(set(x_coords))
        
        # Find significant gaps that indicate column boundaries
        gaps = []
        for i in range(1, len(x_coords)):
            gap = x_coords[i] - x_coords[i-1]
            gaps.append((x_coords[i-1], x_coords[i], gap))
        
        # Sort by gap size and identify column boundaries
        gaps.sort(key=lambda x: x[2], reverse=True)
        
        # Use a threshold to identify significant gaps
        if gaps:
            median_gap = sorted([g[2] for g in gaps])[len(gaps)//2]
            threshold = median_gap * 2
            
            columns = []
            for gap in gaps:
                if gap[2] > threshold:
                    columns.append((gap[0], gap[1]))
        
        return columns
    
    def _estimate_structure_type(self, lines: Dict, columns: List) -> str:
        """Estimate the structure type of the content"""
        if not lines:
            return "empty"
        
        # Check if content is primarily tabular
        consistent_columns = len(columns) > 0
        line_lengths = [len(line_words) for line_words in lines.values()]
        avg_words_per_line = sum(line_lengths) / len(line_lengths) if line_lengths else 0
        
        if consistent_columns and avg_words_per_line > 3:
            return "tabular"
        elif avg_words_per_line > 10:
            return "paragraph"
        else:
            return "mixed"
    
    def extract_tables_camelot(self, pdf_path: str, pages: Optional[List[int]] = None) -> List[TableStructure]:
        """Extract tables using camelot for more complex table structures"""
        tables = []
        
        try:
            # Camelot uses 1-based page numbers
            page_range = ','.join(map(str, pages)) if pages else 'all'
            
            # Stream method for simple tables
            table_list = camelot.read_pdf(pdf_path, pages=page_range, flavor='stream')
            
            for i, table in enumerate(table_list):
                table_struct = TableStructure(
                    page_number=table.page,
                    bbox=(0, 0, 0, 0),  # Camelot doesn't provide precise bbox
                    data=table.df,
                    confidence=table.parsing_report.get('accuracy', 0.0),
                    extraction_method=ExtractionMethod.CAMELOT
                )
                tables.append(table_struct)
            
            # Lattice method for tables with visible grid lines
            if not tables or all(t.confidence < 0.5 for t in tables):
                table_list = camelot.read_pdf(pdf_path, pages=page_range, flavor='lattice')
                for i, table in enumerate(table_list):
                    table_struct = TableStructure(
                        page_number=table.page,
                        bbox=table._bbox,
                        data=table.df,
                        confidence=table.parsing_report.get('accuracy', 0.0),
                        extraction_method=ExtractionMethod.CAMELOT
                    )
                    tables.append(table_struct)
        
        except Exception as e:
            logger.error(f"Camelot extraction failed: {str(e)}")
        
        return tables
    
    def extract_with_ocr(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> Dict[int, str]:
        """Extract text using OCR for scanned PDFs or images"""
        results = {}
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                if page_numbers and (page_num + 1) not in page_numbers:
                    continue
                
                page = doc[page_num]
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better OCR
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))
                
                # Preprocess image for better OCR
                processed_image = self._preprocess_image_for_ocr(image)
                
                # Perform OCR
                text = pytesseract.image_to_string(processed_image, lang='eng')
                results[page_num + 1] = text
        
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
        
        return results
    
    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Convert back to PIL Image
        return Image.fromarray(denoised)
    
    def extract_domain_data(self, pdf_path: str, domain_info: Dict[str, Dict]) -> Dict[str, DomainData]:
        """Extract data for specific SENDIG domains using hybrid approach"""
        domain_results = {}
        
        # First pass: Extract all content with layout analysis
        page_data = self.extract_text_with_layout(pdf_path)
        
        # Extract tables with multiple methods
        camelot_tables = self.extract_tables_camelot(pdf_path)
        
        # Perform OCR on complex pages if enabled
        if self.use_ocr:
            ocr_results = self.extract_with_ocr(pdf_path)
        
        # Process each domain
        for domain_code, domain_details in domain_info.items():
            domain_pages = domain_details.get('pages', [])
            keywords = domain_details.get('keywords', [domain_code])
            
            domain_data = DomainData(
                domain_code=domain_code,
                pages=domain_pages,
                tables=[],
                text_blocks=[],
                extraction_confidence=0.0
            )
            
            # Extract data from specified pages
            for page_num in domain_pages:
                if page_num in page_data:
                    page_info = page_data[page_num]
                    
                    # Filter relevant text blocks
                    text = page_info.get('text', '')
                    relevant_blocks = self._extract_relevant_text_blocks(text, keywords)
                    domain_data.text_blocks.extend(relevant_blocks)
                    
                    # Find tables related to this domain
                    domain_tables = self._find_domain_tables(
                        page_info.get('tables', []),
                        camelot_tables,
                        page_num,
                        keywords
                    )
                    domain_data.tables.extend(domain_tables)
            
            # Calculate extraction confidence
            domain_data.extraction_confidence = self._calculate_domain_confidence(domain_data)
            domain_results[domain_code] = domain_data
        
        return domain_results
    
    def _extract_relevant_text_blocks(self, text: str, keywords: List[str]) -> List[str]:
        """Extract text blocks relevant to specific keywords"""
        blocks = []
        lines = text.split('\n')
        current_block = []
        
        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in keywords):
                if current_block:
                    blocks.append('\n'.join(current_block))
                current_block = [line]
            elif current_block:
                current_block.append(line)
                # End block if empty line or significant content change
                if not line.strip() or len(current_block) > 20:
                    blocks.append('\n'.join(current_block))
                    current_block = []
        
        if current_block:
            blocks.append('\n'.join(current_block))
        
        return blocks
    
    def _find_domain_tables(self, 
                           pdfplumber_tables: List[pd.DataFrame],
                           camelot_tables: List[TableStructure],
                           page_num: int,
                           keywords: List[str]) -> List[TableStructure]:
        """Find tables related to a specific domain"""
        domain_tables = []
        
        # Check pdfplumber tables
        for i, table in enumerate(pdfplumber_tables):
            if self._is_domain_table(table, keywords):
                table_struct = TableStructure(
                    page_number=page_num,
                    bbox=(0, 0, 0, 0),
                    data=table,
                    confidence=0.8,  # Default confidence for pdfplumber
                    extraction_method=ExtractionMethod.PDFPLUMBER
                )
                domain_tables.append(table_struct)
        
        # Check camelot tables for this page
        for table in camelot_tables:
            if table.page_number == page_num and self._is_domain_table(table.data, keywords):
                domain_tables.append(table)
        
        return domain_tables
    
    def _is_domain_table(self, table: pd.DataFrame, keywords: List[str]) -> bool:
        """Check if a table belongs to a specific domain"""
        # Convert table to string for searching
        table_str = table.to_string().lower()
        
        # Check for keywords in table headers or content
        for keyword in keywords:
            if keyword.lower() in table_str:
                return True
        
        # Additional heuristics based on column names or patterns
        if not table.empty:
            headers = [str(col).lower() for col in table.columns]
            for keyword in keywords:
                if any(keyword.lower() in header for header in headers):
                    return True
        
        return False
    
    def _calculate_domain_confidence(self, domain_data: DomainData) -> float:
        """Calculate confidence score for extracted domain data"""
        scores = []
        
        # Text block confidence (presence of relevant data)
        if domain_data.text_blocks:
            text_score = min(len(domain_data.text_blocks) / 5.0, 1.0)
            scores.append(text_score)
        
        # Table confidence (average of all table confidences)
        if domain_data.tables:
            table_scores = [t.confidence for t in domain_data.tables]
            avg_table_score = sum(table_scores) / len(table_scores)
            scores.append(avg_table_score)
        
        # Page coverage (data found across expected pages)
        if domain_data.pages:
            pages_with_data = set()
            for table in domain_data.tables:
                pages_with_data.add(table.page_number)
            
            coverage_score = len(pages_with_data) / len(domain_data.pages)
            scores.append(coverage_score)
        
        return sum(scores) / len(scores) if scores else 0.0

# Usage example
if __name__ == "__main__":
    service = EnhancedPDFService(use_ocr=True)
    
    # Define domain information
    domain_info = {
        'DM': {
            'pages': [1, 2],
            'keywords': ['Demographics', 'Subject', 'Age', 'Sex', 'Species']
        },
        'BW': {
            'pages': [5, 6],
            'keywords': ['Body Weight', 'BW', 'Weight', 'Measurement']
        },
        'LB': {
            'pages': [10, 11, 12],
            'keywords': ['Laboratory', 'Lab Results', 'Clinical Chemistry', 'Hematology']
        }
    }
    
    # Extract domain data
    results = service.extract_domain_data('toxicology_study.pdf', domain_info)
    
    # Process results
    for domain_code, domain_data in results.items():
        print(f"\nDomain: {domain_code}")
        print(f"Confidence: {domain_data.extraction_confidence:.2f}")
        print(f"Tables found: {len(domain_data.tables)}")
        print(f"Text blocks: {len(domain_data.text_blocks)}")