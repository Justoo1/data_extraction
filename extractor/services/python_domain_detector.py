import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .python_service import EnhancedPDFService, TableStructure, ExtractionMethod
from .llm_service import LLMService
from ..models import SENDIGDomain, DetectedDomain, PDFDocument

logger = logging.getLogger(__name__)

@dataclass
class DomainDetectionResult:
    """Results from domain detection process"""
    domain_code: str
    confidence: float
    pages: List[int]
    evidence: List[str]
    tables: List[TableStructure]
    detection_method: str

class EnhancedDomainDetector:
    """
    Advanced domain detector using multiple detection strategies:
    1. Pattern matching for known structures
    2. Machine learning similarity matching
    3. LLM-based detection for complex cases
    4. Table structure analysis
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self.pdf_service = EnhancedPDFService(use_ocr=True)
        
        # Initialize domain patterns
        self.domain_patterns = self._initialize_domain_patterns()
        
        # Initialize vectorizer for similarity matching
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            stop_words='english'
        )
        
        # Domain keyword mappings
        self.domain_keywords = self._get_domain_keywords()
    
    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for each SENDIG domain"""
        patterns = {
            'DM': [
                r'demographics\s+table',
                r'subject\s+demographics',
                r'animal\s+demographics',
                r'demographic\s+data',
                r'species.*sex.*age',
                r'subject\s+id.*species.*strain',
            ],
            'BW': [
                r'body\s+weight\s+table',
                r'weight\s+measurements',
                r'terminal\s+body\s+weights',
                r'weekly\s+body\s+weights',
                r'bw\s+group.*mean.*sd',
            ],
            'LB': [
                r'laboratory\s+results',
                r'clinical\s+chemistry',
                r'hematology\s+results',
                r'lb\s+domain',
                r'lab\s+parameter.*value.*unit',
            ],
            'MA': [
                r'macroscopic\s+findings',
                r'gross\s+pathology',
                r'necropsy\s+findings',
                r'organ\s+weights',
                r'ma\s+domain',
            ],
            'MI': [
                r'microscopic\s+findings',
                r'histopathology',
                r'tissue\s+examination',
                r'pathology\s+results',
                r'mi\s+domain',
            ],
            'EX': [
                r'exposure\s+data',
                r'dosing\s+records',
                r'treatment\s+administration',
                r'ex\s+domain',
                r'dose\s+level.*route.*frequency',
            ],
            'PC': [
                r'pharmacokinetic\s+concentration',
                r'pk\s+sampling',
                r'plasma\s+concentration',
                r'pc\s+domain',
                r'time\s+point.*concentration.*subject',
            ],
            'PP': [
                r'pharmacokinetic\s+parameters',
                r'pk\s+parameters',
                r'cmax.*tmax.*auc',
                r'pp\s+domain',
                r'parameter.*mean.*cv',
            ]
        }
        
        # Compile patterns for performance
        compiled_patterns = {}
        for domain, pattern_list in patterns.items():
            compiled_patterns[domain] = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_list]
        
        return compiled_patterns
    
    def _get_domain_keywords(self) -> Dict[str, List[str]]:
        """Get comprehensive keyword lists for each domain"""
        return {
            'DM': ['demographics', 'subject', 'animal', 'species', 'strain', 'sex', 'age', 'birth date'],
            'BW': ['body weight', 'weight', 'bw', 'terminal weight', 'weekly weight', 'mean weight'],
            'LB': ['laboratory', 'lab', 'clinical chemistry', 'hematology', 'urinalysis', 'test result'],
            'CL': ['clinical', 'observation', 'cage side', 'behavioral', 'clinical sign'],
            'MA': ['macroscopic', 'gross pathology', 'necropsy', 'organ', 'tissue', 'abnormal'],
            'MI': ['microscopic', 'histopathology', 'tissue section', 'pathology', 'finding'],
            'EX': ['exposure', 'dose', 'administration', 'treatment', 'vehicle', 'route'],
            'DS': ['disposition', 'study completion', 'early termination', 'euthanasia'],
            'PC': ['pharmacokinetic', 'concentration', 'plasma', 'serum', 'pk', 'sampling'],
            'PP': ['parameter', 'cmax', 'tmax', 'auc', 'clearance', 'half-life'],
            'TS': ['trial summary', 'study design', 'protocol', 'objective'],
            'TE': ['trial element', 'epoch', 'arm', 'element'],
            'TA': ['trial arm', 'treatment group', 'control group'],
            'OM': ['organ measurement', 'organ weight', 'relative weight'],
            'FW': ['food consumption', 'feed intake', 'food efficiency'],
        }
    
    def detect_domains(self, pdf_document: PDFDocument) -> List[DetectedDomain]:
        """
        Detect SENDIG domains using multiple detection strategies
        
        Args:
            pdf_document: PDFDocument instance
            
        Returns:
            List of DetectedDomain instances
        """
        try:
            # Get all SENDIG domains from database
            all_domains = {domain.code: domain for domain in SENDIGDomain.objects.all()}
            
            if not all_domains:
                logger.warning("No SENDIG domains found in the database")
                return []
            
            # Extract comprehensive document data
            logger.info("Extracting comprehensive document data...")
            page_data = self.pdf_service.extract_text_with_layout(pdf_document.file.path)
            
            # Extract tables with multiple methods
            camelot_tables = self.pdf_service.extract_tables_camelot(pdf_document.file.path)
            
            # Create consolidated text representation
            consolidated_text = self._create_consolidated_text(page_data)
            
            # Perform pattern-based detection
            logger.info("Performing pattern-based detection...")
            pattern_results = self._detect_domains_by_patterns(page_data, camelot_tables)
            
            # Perform similarity-based detection
            logger.info("Performing similarity-based detection...")
            similarity_results = self._detect_domains_by_similarity(page_data, all_domains)
            
            # Perform LLM-based detection for uncertain cases
            logger.info("Performing LLM-based detection...")
            llm_results = self._detect_domains_by_llm(consolidated_text, all_domains)
            
            # Combine and rank detection results
            combined_results = self._combine_detection_results(
                pattern_results, 
                similarity_results, 
                llm_results
            )
            
            # Convert to DetectedDomain objects
            detected_domains = []
            for result in combined_results:
                if result.confidence >= 0.3:  # Minimum confidence threshold
                    domain = all_domains.get(result.domain_code)
                    if domain:
                        pages_str = self._format_page_ranges(result.pages)
                        detected_domain, created = DetectedDomain.objects.update_or_create(
                            pdf=pdf_document,
                            domain=domain,
                            defaults={
                                'pages': pages_str,
                                'selected': result.confidence >= 0.7,  # Auto-select high confidence
                                'confidence_score': result.confidence
                            }
                        )
                        detected_domains.append(detected_domain)
            
            # Update PDF document status
            pdf_document.status = 'ANALYZED'
            pdf_document.save()
            
            return detected_domains
            
        except Exception as e:
            logger.error(f"Error detecting domains in PDF {pdf_document.id}: {str(e)}")
            pdf_document.status = 'FAILED'
            pdf_document.save()
            raise
    
    def _create_consolidated_text(self, page_data: Dict) -> str:
        """Create consolidated text representation with page markers"""
        consolidated = []
        for page_num, data in page_data.items():
            text = data.get('text', '')
            if text.strip():
                consolidated.append(f"--- PAGE {page_num} ---\n{text}")
        return "\n\n".join(consolidated)
    
    def _detect_domains_by_patterns(self, 
                                   page_data: Dict, 
                                   tables: List[TableStructure]) -> List[DomainDetectionResult]:
        """Detect domains using predefined patterns"""
        results = []
        
        for page_num, data in page_data.items():
            text = data.get('text', '').lower()
            
            # Check each domain's patterns
            for domain_code, patterns in self.domain_patterns.items():
                matches = []
                for pattern in patterns:
                    matches.extend(pattern.findall(text))
                
                if matches:
                    # Find related tables
                    domain_tables = [t for t in tables if t.page_number == page_num and 
                                   self._is_table_relevant(t.data, domain_code)]
                    
                    confidence = min(len(matches) * 0.2 + 0.3, 0.9)
                    
                    results.append(DomainDetectionResult(
                        domain_code=domain_code,
                        confidence=confidence,
                        pages=[page_num],
                        evidence=matches[:3],  # Top 3 matches
                        tables=domain_tables,
                        detection_method='pattern'
                    ))
        
        return results
    
    def _detect_domains_by_similarity(self, 
                                    page_data: Dict,
                                    all_domains: Dict[str, SENDIGDomain]) -> List[DomainDetectionResult]:
        """Detect domains using TF-IDF similarity matching"""
        results = []
        
        # Create domain descriptions for comparison
        domain_descriptions = {}
        for code, domain in all_domains.items():
            # Combine name, description, and keywords
            keywords = self.domain_keywords.get(code, [])
            full_description = f"{domain.name} {domain.description} {' '.join(keywords)}"
            domain_descriptions[code] = full_description
        
        # Fit vectorizer on domain descriptions
        domain_vectors = self.vectorizer.fit_transform(domain_descriptions.values())
        
        # Process each page
        for page_num, data in page_data.items():
            text = data.get('text', '')
            if not text.strip():
                continue
            
            # Vectorize page text
            page_vector = self.vectorizer.transform([text])
            
            # Calculate similarities
            similarities = cosine_similarity(page_vector, domain_vectors)[0]
            
            # Check each domain
            for idx, (domain_code, domain) in enumerate(all_domains.items()):
                similarity = similarities[idx]
                
                if similarity > 0.2:  # Minimum similarity threshold
                    results.append(DomainDetectionResult(
                        domain_code=domain_code,
                        confidence=min(similarity * 0.8, 0.85),  # Scale down confidence
                        pages=[page_num],
                        evidence=[f"Similarity score: {similarity:.3f}"],
                        tables=[],
                        detection_method='similarity'
                    ))
        
        return results
    
    def _detect_domains_by_llm(self, 
                              consolidated_text: str,
                              all_domains: Dict[str, SENDIGDomain]) -> List[DomainDetectionResult]:
        """Detect domains using LLM for complex cases"""
        results = []
        
        # Split text into chunks for LLM processing
        chunks = self._split_text_for_llm(consolidated_text)
        
        for chunk in chunks:
            try:
                # Create domain descriptions
                domain_descriptions = {
                    code: {"name": domain.name, "description": domain.description}
                    for code, domain in all_domains.items()
                }
                
                # Call LLM for detection
                llm_results = self.llm_service.detect_sendig_domains(chunk, domain_descriptions)
                
                if llm_results and 'domains' in llm_results:
                    for domain_result in llm_results['domains']:
                        domain_code = domain_result.get('code')
                        confidence = domain_result.get('confidence', 0.0)
                        explanation = domain_result.get('explanation', '')
                        
                        if domain_code in all_domains:
                            # Extract page numbers from chunk
                            pages = self._extract_page_numbers_from_chunk(chunk)
                            
                            results.append(DomainDetectionResult(
                                domain_code=domain_code,
                                confidence=confidence,
                                pages=pages,
                                evidence=[explanation],
                                tables=[],
                                detection_method='llm'
                            ))
            
            except Exception as e:
                logger.error(f"LLM detection failed for chunk: {str(e)}")
                continue
        
        return results
    
    def _split_text_for_llm(self, text: str, max_chunk_size: int = 4000) -> List[str]:
        """Split text into manageable chunks for LLM processing"""
        chunks = []
        pages = text.split('--- PAGE')
        current_chunk = ""
        
        for page in pages:
            if not page.strip():
                continue
                
            if len(current_chunk) + len(page) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = "--- PAGE" + page
            else:
                current_chunk += ("--- PAGE" if current_chunk else "") + page
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _extract_page_numbers_from_chunk(self, chunk: str) -> List[int]:
        """Extract page numbers from a text chunk"""
        page_numbers = []
        matches = re.findall(r'--- PAGE (\d+) ---', chunk)
        for match in matches:
            try:
                page_numbers.append(int(match))
            except ValueError:
                continue
        return sorted(set(page_numbers))
    
    def _combine_detection_results(self, *result_lists) -> List[DomainDetectionResult]:
        """Combine and rank detection results from multiple methods"""
        # Group results by domain
        domain_results = defaultdict(list)
        for result_list in result_lists:
            for result in result_list:
                domain_results[result.domain_code].append(result)
        
        # Combine and rank
        final_results = []
        for domain_code, results in domain_results.items():
            # Calculate weighted confidence
            total_weight = 0
            weighted_confidence = 0
            all_pages = set()
            all_evidence = []
            all_tables = []
            
            method_weights = {
                'pattern': 0.4,
                'similarity': 0.3,
                'llm': 0.3
            }
            
            for result in results:
                weight = method_weights.get(result.detection_method, 0.25)
                weighted_confidence += result.confidence * weight
                total_weight += weight
                all_pages.update(result.pages)
                all_evidence.extend(result.evidence)
                all_tables.extend(result.tables)
            
            # Normalize confidence
            if total_weight > 0:
                final_confidence = weighted_confidence / total_weight
            else:
                final_confidence = max(r.confidence for r in results)
            
            # Create combined result
            final_result = DomainDetectionResult(
                domain_code=domain_code,
                confidence=final_confidence,
                pages=sorted(list(all_pages)),
                evidence=all_evidence[:5],  # Top 5 pieces of evidence
                tables=all_tables,
                detection_method='combined'
            )
            final_results.append(final_result)
        
        # Sort by confidence
        return sorted(final_results, key=lambda x: x.confidence, reverse=True)
    
    def _is_table_relevant(self, table: pd.DataFrame, domain_code: str) -> bool:
        """Check if a table is relevant to a specific domain"""
        if table.empty:
            return False
        
        # Get domain keywords
        keywords = self.domain_keywords.get(domain_code, [])
        
        # Check table headers
        headers = [str(col).lower() for col in table.columns]
        for keyword in keywords:
            if any(keyword.lower() in header for header in headers):
                return True
        
        # Check table content (first few rows)
        try:
            content = table.head(3).to_string().lower()
            for keyword in keywords:
                if keyword.lower() in content:
                    return True
        except:
            pass
        
        return False
    
    def _format_page_ranges(self, pages: List[int]) -> str:
        """Format a list of page numbers as comma-separated ranges"""
        if not pages:
            return ""
            
        pages = sorted(set(pages))
        ranges = []
        range_start = range_end = pages[0]
        
        for page in pages[1:]:
            if page == range_end + 1:
                range_end = page
            else:
                if range_start == range_end:
                    ranges.append(str(range_start))
                else:
                    ranges.append(f"{range_start}-{range_end}")
                range_start = range_end = page
        
        # Add the last range
        if range_start == range_end:
            ranges.append(str(range_start))
        else:
            ranges.append(f"{range_start}-{range_end}")
        
        return ", ".join(ranges)
    
    def analyze_detection_results(self, detected_domains: List[DetectedDomain]) -> Dict:
        """Analyze and provide insights on detection results"""
        analysis = {
            'total_domains': len(detected_domains),
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'coverage': {},
            'suggestions': []
        }
        
        for domain in detected_domains:
            confidence = domain.confidence_score
            
            if confidence >= 0.7:
                analysis['high_confidence'] += 1
            elif confidence >= 0.4:
                analysis['medium_confidence'] += 1
            else:
                analysis['low_confidence'] += 1
            
            # Track page coverage
            pages = domain.get_pages_list()
            for page in pages:
                if page not in analysis['coverage']:
                    analysis['coverage'][page] = []
                analysis['coverage'][page].append(domain.domain.code)
        
        # Generate suggestions
        if analysis['low_confidence'] > analysis['high_confidence']:
            analysis['suggestions'].append(
                "Many domains have low confidence. Consider reviewing the PDF quality or using OCR."
            )
        
        if len(analysis['coverage']) < 0.5 * max(analysis['coverage'].keys()) if analysis['coverage'] else 0:
            analysis['suggestions'].append(
                "Sparse page coverage detected. Some important data may be missing."
            )
        
        return analysis