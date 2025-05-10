import logging
import re
from collections import defaultdict

from .pdf_service import PDFService
from .llm_service import LLMService
from ..models import SENDIGDomain, DetectedDomain, PDFDocument

logger = logging.getLogger(__name__)

class DomainDetector:
    """
    Service to detect SENDIG domains in PDF documents and identify pages
    where domain-specific data is present
    """
    
    def __init__(self, llm_service=None):
        """
        Initialize the domain detector
        
        Args:
            llm_service: LLMService instance (if None, a new one will be created)
        """
        self.llm_service = llm_service or LLMService()
        self.pdf_service = PDFService()
        
    def detect_domains(self, pdf_document):
        """
        Detect SENDIG domains in a PDF document
        
        Args:
            pdf_document: PDFDocument instance
            
        Returns:
            List of DetectedDomain instances
        """
        try:
            # Get all SENDIG domains from the database
            all_domains = {domain.code: domain for domain in SENDIGDomain.objects.all()}
            
            if not all_domains:
                logger.warning("No SENDIG domains found in the database")
                return []
            
            # Extract text from the PDF
            page_texts = self.pdf_service.extract_text_from_pdf(pdf_document.file.path)
            
            # Create a consolidated text for initial domain detection
            consolidated_text = "\n\n".join([
                f"--- PAGE {page} ---\n{text}" 
                for page, text in page_texts.items()
            ])
            
            # Prepare domain descriptions for the LLM
            domain_descriptions = {
                code: {
                    "name": domain.name,
                    "description": domain.description
                } for code, domain in all_domains.items()
            }
            
            # Use LLM to detect domains in the consolidated text
            detection_results = self.llm_service.detect_sendig_domains(
                consolidated_text, 
                domain_descriptions
            )
            
            # No domains detected
            if not detection_results or not detection_results.get("domains"):
                logger.warning(f"No domains detected in PDF {pdf_document.id}")
                return []
            
            # Now identify pages for each detected domain
            detected_domains = []
            for domain_result in detection_results.get("domains", []):
                domain_code = domain_result.get("code")
                confidence = domain_result.get("confidence", 0.0)
                
                if domain_code not in all_domains:
                    logger.warning(f"Detected domain {domain_code} not found in database")
                    continue
                
                # Find pages where this domain is mentioned
                domain_pages = self._identify_domain_pages(
                    domain_code, 
                    all_domains[domain_code].name, 
                    page_texts
                )
                
                if domain_pages:
                    # Format pages as comma-separated ranges
                    pages_str = self._format_page_ranges(domain_pages)
                    
                    # Create or update DetectedDomain
                    detected_domain, created = DetectedDomain.objects.update_or_create(
                        pdf=pdf_document,
                        domain=all_domains[domain_code],
                        defaults={
                            'pages': pages_str,
                            'selected': True,
                            'confidence_score': confidence
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
    
    def _identify_domain_pages(self, domain_code, domain_name, page_texts):
        """
        Identify pages where a specific domain is present
        
        Args:
            domain_code: SENDIG domain code
            domain_name: SENDIG domain name
            page_texts: Dictionary mapping page numbers to page text
            
        Returns:
            List of page numbers where the domain is detected
        """
        domain_pages = []
        
        # Keywords to look for
        keywords = [
            domain_code,
            domain_name,
            # Add any other domain-specific keywords
        ]
        
        # Create regex patterns for keywords (case insensitive)
        patterns = [re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE) for kw in keywords]
        
        # Check each page for domain keywords
        for page_num, text in page_texts.items():
            for pattern in patterns:
                if pattern.search(text):
                    domain_pages.append(page_num)
                    break
        
        # If no pages found through keywords, use LLM to analyze each page
        if not domain_pages:
            for page_num, text in page_texts.items():
                prompt = f"""
                Analyze this text from page {page_num} and determine if it contains information 
                related to the SENDIG domain {domain_code} ({domain_name}).
                Respond with only "yes" or "no".
                
                Text:
                {text[:3000]}
                """
                
                response = self.llm_service.generate(prompt, max_tokens=10, temperature=0.1)
                if "yes" in response.lower():
                    domain_pages.append(page_num)
        
        return sorted(domain_pages)
    
    @staticmethod
    def _format_page_ranges(pages):
        """
        Format a list of page numbers as a string of comma-separated ranges
        e.g. [1, 2, 3, 5, 6, 9] -> "1-3, 5-6, 9"
        
        Args:
            pages: List of page numbers
            
        Returns:
            String with comma-separated page ranges
        """
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