import json
import os
import zipfile
from io import BytesIO

from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.urls import reverse
from django.contrib import messages
from django.views.decorators.http import require_POST
from django.views.generic import TemplateView, ListView, DetailView, View
from django.core.files.storage import default_storage
from django.conf import settings

from .models import PDFDocument, SENDIGDomain, DetectedDomain, ExtractedData
from .services.pdf_service import PDFService
from .services.domain_detector import DomainDetector
from .services.data_extractor import DataExtractor

class HomeView(TemplateView):
    """Home page view"""
    template_name = 'home.html'

class UploadPDFView(View):
    """View for uploading PDF documents"""
    
    def get(self, request):
        """Display upload form"""
        return render(request, 'extractor/upload.html')
    
    def post(self, request):
        """Handle PDF upload"""
        if 'pdf_file' not in request.FILES:
            messages.error(request, "No PDF file uploaded")
            return redirect('extractor:upload')
        
        pdf_file = request.FILES['pdf_file']
        if not pdf_file.name.endswith('.pdf'):
            messages.error(request, "Uploaded file is not a PDF")
            return redirect('extractor:upload')
        
        # Create PDFDocument
        pdf_document = PDFDocument.objects.create(
            name=pdf_file.name,
            file=pdf_file,
            status='UPLOADED'
        )
        
        # Redirect to domain detection
        return redirect('extractor:detect_domains', pdf_id=pdf_document.id)

class DetectDomainsView(View):
    """View for detecting SENDIG domains in a PDF"""
    
    def get(self, request, pdf_id):
        """Display domain detection page"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        
        # If already analyzed, show domains
        if pdf_document.status == 'ANALYZED':
            detected_domains = DetectedDomain.objects.filter(pdf=pdf_document)
            return render(request, 'extractor/domain_selection.html', {
                'pdf': pdf_document,
                'domains': detected_domains
            })
        
        # Show analysis page
        return render(request, 'extractor/detect_domains.html', {
            'pdf': pdf_document
        })
    
    def post(self, request, pdf_id):
        """Start domain detection"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        
        try:
            # Detect domains
            detector = DomainDetector()
            detected_domains = detector.detect_domains(pdf_document)
            
            # Redirect to domain selection
            return redirect('extractor:select_domains', pdf_id=pdf_document.id)
            
        except Exception as e:
            messages.error(request, f"Error detecting domains: {str(e)}")
            return redirect('extractor:upload')

class DomainSelectionView(View):
    """View for selecting domains to process"""
    
    def get(self, request, pdf_id):
        """Display domain selection page"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        detected_domains = DetectedDomain.objects.filter(pdf=pdf_document)
        
        return render(request, 'extractor/domain_selection.html', {
            'pdf': pdf_document,
            'domains': detected_domains
        })
    
    def post(self, request, pdf_id):
        """Handle domain selection"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        
        # Update selected status for domains
        selected_domains = request.POST.getlist('selected_domains')
        
        for detected_domain in DetectedDomain.objects.filter(pdf=pdf_document):
            domain_id = str(detected_domain.domain.code)
            detected_domain.selected = domain_id in selected_domains
            
            # Update pages if provided
            pages_key = f"pages_{domain_id}"
            if pages_key in request.POST:
                detected_domain.pages = request.POST[pages_key]
                
            detected_domain.save()
        
        # Redirect to processing
        return redirect('extractor:process_domains', pdf_id=pdf_document.id)

class ProcessDomainsView(View):
    """View for processing selected domains"""
    
    def get(self, request, pdf_id):
        """Display processing page"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        selected_domains = DetectedDomain.objects.filter(pdf=pdf_document, selected=True)
        
        # If already completed, redirect to results
        if pdf_document.status == 'COMPLETED':
            return redirect('extractor:results', pdf_id=pdf_document.id)
        
        return render(request, 'extractor/processing.html', {
            'pdf': pdf_document,
            'domains': selected_domains
        })
    
    def post(self, request, pdf_id):
        """Start domain processing"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        
        try:
            # Extract data for selected domains
            extractor = DataExtractor()
            extracted_data = extractor.extract_all_selected_domains(pdf_document)
            
            # Redirect to results
            return redirect('extractor:results', pdf_id=pdf_document.id)
            
        except Exception as e:
            messages.error(request, f"Error processing domains: {str(e)}")
            return redirect('extractor:select_domains', pdf_id=pdf_document.id)

class DomainProcessingStatusView(View):
    """AJAX view for checking domain processing status"""
    
    def get(self, request, pdf_id):
        """Return JSON with processing status"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        selected_domains = DetectedDomain.objects.filter(pdf=pdf_document, selected=True)
        extracted_domains = ExtractedData.objects.filter(pdf=pdf_document)
        
        # Calculate progress
        total_domains = selected_domains.count()
        processed_domains = extracted_domains.count()
        
        # Get currently processing domain (if any)
        current_domain = None
        if pdf_document.status == 'PROCESSING' and processed_domains < total_domains:
            # Find domain that is not yet extracted
            for domain in selected_domains:
                if not ExtractedData.objects.filter(pdf=pdf_document, domain=domain.domain).exists():
                    current_domain = domain.domain.code
                    break
        
        return JsonResponse({
            'status': pdf_document.status,
            'total_domains': total_domains,
            'processed_domains': processed_domains,
            'current_domain': current_domain,
            'progress_percent': int(processed_domains / max(total_domains, 1) * 100)
        })

class ResultsView(View):
    """View for displaying extraction results"""
    
    def get(self, request, pdf_id):
        """Display results page"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        extracted_data = ExtractedData.objects.filter(pdf=pdf_document)
        
        return render(request, 'extractor/results.html', {
            'pdf': pdf_document,
            'extracted_data': extracted_data
        })

class ViewXPTFileView(View):
    """View for displaying XPT file content"""
    
    def get(self, request, data_id):
        """Display XPT file content"""
        extracted_data = get_object_or_404(ExtractedData, id=data_id)
        
        # Display XPT as JSON
        return render(request, 'extractor/view_xpt.html', {
            'extracted_data': extracted_data,
            'data_json': json.dumps(extracted_data.data, indent=2)
        })

class DownloadXPTFileView(View):
    """View for downloading XPT file"""
    
    def get(self, request, data_id):
        """Download XPT file"""
        extracted_data = get_object_or_404(ExtractedData, id=data_id)
        
        if not extracted_data.xpt_file:
            messages.error(request, "XPT file not found")
            return redirect('extractor:results', pdf_id=extracted_data.pdf.id)
        
        # Prepare response
        response = HttpResponse(
            extracted_data.xpt_file.read(),
            content_type='application/octet-stream'
        )
        response['Content-Disposition'] = f'attachment; filename="{extracted_data.domain.code}.xpt"'
        
        return response

class DownloadAllFilesView(View):
    """View for downloading all XPT files as a ZIP"""
    
    def get(self, request, pdf_id):
        """Download all XPT files as a ZIP"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        extracted_data = ExtractedData.objects.filter(pdf=pdf_document)
        
        # Create ZIP file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add XPT files
            for data in extracted_data:
                if data.xpt_file:
                    zip_file.writestr(f"{data.domain.code}.xpt", data.xpt_file.read())
            
            # Add raw JSON data
            for data in extracted_data:
                json_data = json.dumps(data.data, indent=2)
                zip_file.writestr(f"{data.domain.code}_data.json", json_data)
        
        # Prepare response
        response = HttpResponse(
            zip_buffer.getvalue(),
            content_type='application/zip'
        )
        response['Content-Disposition'] = f'attachment; filename="sendig_data_{pdf_document.id}.zip"'
        
        return response