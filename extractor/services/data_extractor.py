import logging
import json
import os
from django.core.files.base import ContentFile
from django.utils import timezone

from .pdf_service import PDFService
from .llm_service import LLMService
from .xpt_generator import XPTGenerator
from ..models import PDFDocument, DetectedDomain, ExtractedData, SENDIGDomain

logger = logging.getLogger(__name__)

class DataExtractor:
    """
    Service to extract structured data from PDFs according to SENDIG domains
    """
    
    def __init__(self, llm_service=None):
        """
        Initialize the data extractor
        
        Args:
            llm_service: LLMService instance (if None, a new one will be created)
        """
        self.llm_service = llm_service or LLMService()
        self.pdf_service = PDFService()
        self.xpt_generator = XPTGenerator()
    
    def extract_data_for_domain(self, pdf_document, detected_domain):
        """
        Extract structured data for a specific domain from a PDF
        
        Args:
            pdf_document: PDFDocument instance
            detected_domain: DetectedDomain instance
            
        Returns:
            ExtractedData instance
        """
        try:
            # Get domain details
            domain = detected_domain.domain
            domain_code = domain.code
            
            # Get required variables for this domain
            required_variables = domain.get_required_variables()
            
            # Get pages for this domain
            pages = detected_domain.get_pages_list()
            
            # Extract text from specified pages
            page_texts = self.pdf_service.extract_text_by_pages(pdf_document.file.path, pages)
            
            # Combine text from all pages
            combined_text = "\n\n".join([
                f"--- PAGE {page} ---\n{text}" 
                for page, text in page_texts.items()
            ])
            
            # Use LLM to extract structured data
            extracted_data = self.llm_service.extract_structured_data(
                combined_text,
                domain_code,
                required_variables
            )
            
            # Generate XPT file
            xpt_data = self.xpt_generator.generate_xpt(domain_code, extracted_data)
            
            # Save extracted data
            extracted_obj, created = ExtractedData.objects.update_or_create(
                pdf=pdf_document,
                domain=domain,
                defaults={
                    'data': extracted_data
                }
            )
            
            # Save XPT file
            xpt_filename = f"{pdf_document.id}_{domain_code}.xpt"
            extracted_obj.xpt_file.save(xpt_filename, ContentFile(xpt_data))
            
            return extracted_obj
            
        except Exception as e:
            logger.error(f"Error extracting data for domain {detected_domain.domain.code} in PDF {pdf_document.id}: {str(e)}")
            raise
    
    def extract_all_selected_domains(self, pdf_document):
        """
        Extract data for all selected domains in a PDF
        
        Args:
            pdf_document: PDFDocument instance
            
        Returns:
            Dictionary mapping domain codes to ExtractedData instances
        """
        try:
            # Update PDF status to processing
            pdf_document.status = 'PROCESSING'
            pdf_document.save()
            
            # Get all selected domains
            selected_domains = DetectedDomain.objects.filter(
                pdf=pdf_document,
                selected=True
            )
            
            results = {}
            for detected_domain in selected_domains:
                try:
                    extracted_data = self.extract_data_for_domain(pdf_document, detected_domain)
                    results[detected_domain.domain.code] = extracted_data
                except Exception as e:
                    logger.error(f"Error processing domain {detected_domain.domain.code}: {str(e)}")
                    # Continue with other domains even if one fails
            
            # Update PDF status to completed
            pdf_document.status = 'COMPLETED'
            pdf_document.save()
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting data for PDF {pdf_document.id}: {str(e)}")
            pdf_document.status = 'FAILED'
            pdf_document.save()
            raise