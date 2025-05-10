from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import get_object_or_404
import json
import threading
import logging

from .models import PDFDocument
from .services.domain_detector import DomainDetector

logger = logging.getLogger(__name__)

@require_POST
def api_detect_domains(request):
    """
    API endpoint to detect domains in a PDF asynchronously
    Returns immediately and continues processing in background
    """
    try:
        # Parse JSON data from request
        data = json.loads(request.body)
        document_id = data.get('document_id')
        
        if not document_id:
            return JsonResponse(
                {'error': 'No document ID provided'}, 
                status=400
            )
        
        # Get PDF document
        pdf_document = get_object_or_404(PDFDocument, id=document_id)
        
        # Check if already analyzed
        if pdf_document.status == 'ANALYZED':
            return JsonResponse({
                'status': 'success',
                'message': 'Domains already detected',
                'document_id': str(pdf_document.id)
            })
        
        # Update document status
        pdf_document.status = 'PROCESSING'
        pdf_document.save()
        
        # Start domain detection in background thread
        thread = threading.Thread(
            target=_process_domain_detection,
            args=(pdf_document,)
        )
        thread.daemon = True
        thread.start()
        
        return JsonResponse({
            'status': 'success',
            'message': 'Domain detection started',
            'document_id': str(pdf_document.id)
        })
        
    except Exception as e:
        logger.error(f"Error starting domain detection: {str(e)}")
        return JsonResponse(
            {'error': str(e)}, 
            status=500
        )

def _process_domain_detection(pdf_document):
    """
    Process domain detection in background
    
    Args:
        pdf_document: PDFDocument instance
    """
    try:
        detector = DomainDetector()
        detector.detect_domains(pdf_document)
        
        # Update document status (should be set to 'ANALYZED' by detect_domains)
        if pdf_document.status != 'ANALYZED':
            pdf_document.status = 'ANALYZED'
            pdf_document.save()
            
    except Exception as e:
        logger.error(f"Error in background domain detection: {str(e)}")
        pdf_document.status = 'FAILED'
        pdf_document.save()

@require_POST
def api_domain_detection_status(request, pdf_id):
    """
    API endpoint to check domain detection status
    """
    try:
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        
        return JsonResponse({
            'status': pdf_document.status,
            'document_id': str(pdf_document.id)
        })
        
    except Exception as e:
        logger.error(f"Error checking domain detection status: {str(e)}")
        return JsonResponse(
            {'error': str(e)}, 
            status=500
        )