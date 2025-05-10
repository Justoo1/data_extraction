import json
import os
import zipfile
from io import BytesIO
import pandas as pd

from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.urls import reverse
from django.contrib import messages
from django.views.decorators.http import require_POST
from django.views.generic import TemplateView, ListView, DetailView, View
from django.core.files.storage import default_storage
from django.conf import settings

from .models import PDFDocument, SENDIGDomain, DetectedDomain, ExtractedData
from .services.python_service import EnhancedPDFService
from .services.python_domain_detector import EnhancedDomainDetector
from .services.python_data_extractor import EnhancedDataExtractor
from .services.xpt_generator_enhanced import EnhancedXPTGenerator

class HomeView(TemplateView):
    """Home page view"""
    template_name = 'home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Add statistics
        context['total_documents'] = PDFDocument.objects.count()
        context['completed_documents'] = PDFDocument.objects.filter(status='COMPLETED').count()
        context['total_domains'] = SENDIGDomain.objects.count()
        
        return context

class UploadPDFView(View):
    """View for uploading PDF documents with enhanced validation"""
    
    def get(self, request):
        """Display upload form"""
        context = {
            'max_file_size': settings.FILE_UPLOAD_MAX_MEMORY_SIZE,
            'supported_formats': ['.pdf']
        }
        return render(request, 'extractor/upload.html', context)
    
    def post(self, request):
        """Handle PDF upload with validation"""
        if 'pdf_file' not in request.FILES:
            messages.error(request, "No PDF file uploaded")
            return redirect('extractor:upload')
        
        pdf_file = request.FILES['pdf_file']
        
        # Validate file
        if not pdf_file.name.endswith('.pdf'):
            messages.error(request, "Uploaded file is not a PDF")
            return redirect('extractor:upload')
        
        if pdf_file.size > settings.FILE_UPLOAD_MAX_MEMORY_SIZE:
            messages.error(request, f"File size exceeds {settings.FILE_UPLOAD_MAX_MEMORY_SIZE/1024/1024:.1f}MB limit")
            return redirect('extractor:upload')
        
        try:
            # Create PDFDocument
            pdf_document = PDFDocument.objects.create(
                name=pdf_file.name,
                file=pdf_file,
                status='UPLOADED'
            )
            
            # Quick validation using enhanced PDF service
            pdf_service = EnhancedPDFService()
            try:
                # Extract first page to validate PDF structure
                page_data = pdf_service.extract_text_with_layout(pdf_document.file.path)
                if not page_data:
                    raise Exception("Unable to extract text from PDF")
                
                # Add metadata to document
                metadata = {}
                if page_data:
                    first_page = list(page_data.values())[0]
                    metadata['first_page_preview'] = first_page.get('text', '')[:500]
                    metadata['estimated_pages'] = len(page_data)
                    metadata['estimated_tables'] = sum(len(data.get('tables', [])) for data in page_data.values())
                
                pdf_document.data = {'metadata': metadata}
                pdf_document.save()
                
            except Exception as e:
                pdf_document.delete()
                messages.error(request, f"Error processing PDF: {str(e)}")
                return redirect('extractor:upload')
            
            messages.success(request, f"Successfully uploaded {pdf_file.name}")
            return redirect('extractor:detect_domains', pdf_id=pdf_document.id)
            
        except Exception as e:
            messages.error(request, f"Error uploading file: {str(e)}")
            return redirect('extractor:upload')

class DetectDomainsView(View):
    """View for detecting SENDIG domains using enhanced detector"""
    
    def get(self, request, pdf_id):
        """Display domain detection page"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        
        # If already analyzed, show domains
        if pdf_document.status == 'ANALYZED':
            detected_domains = DetectedDomain.objects.filter(pdf=pdf_document)
            
            # Add detection analysis
            detector = EnhancedDomainDetector()
            analysis = detector.analyze_detection_results(list(detected_domains))
            
            return render(request, 'extractor/domain_selection.html', {
                'pdf': pdf_document,
                'domains': detected_domains,
                'analysis': analysis
            })
        
        # Show analysis page
        context = {
            'pdf': pdf_document,
            'total_domains': SENDIGDomain.objects.count()
        }
        return render(request, 'extractor/detect_domains.html', context)
    
    def post(self, request, pdf_id):
        """Start domain detection using enhanced detector"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        
        try:
            # Use enhanced domain detector
            detector = EnhancedDomainDetector()
            detected_domains = detector.detect_domains(pdf_document)
            
            if not detected_domains:
                messages.warning(request, "No SENDIG domains were detected in this document")
            else:
                messages.success(request, f"Detected {len(detected_domains)} domains")
            
            # Redirect to domain selection
            return redirect('extractor:select_domains', pdf_id=pdf_document.id)
            
        except Exception as e:
            messages.error(request, f"Error detecting domains: {str(e)}")
            return redirect('extractor:upload')

class DomainSelectionView(View):
    """View for selecting domains with validation preview"""
    
    def get(self, request, pdf_id):
        """Display domain selection page with enhanced features"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        detected_domains = DetectedDomain.objects.filter(pdf=pdf_document)
        
        # Add domain statistics
        domain_stats = {}
        pdf_service = EnhancedPDFService()
        
        for domain in detected_domains:
            domain_code = domain.domain.code
            pages = domain.get_pages_list()
            
            try:
                # Quick preview extraction
                page_data = pdf_service.extract_text_with_layout(pdf_document.file.path)
                tables_on_pages = sum(len(page_data.get(p, {}).get('tables', [])) for p in pages)
                
                domain_stats[domain_code] = {
                    'pages': pages,
                    'estimated_tables': tables_on_pages,
                    'confidence': domain.confidence_score
                }
            except:
                domain_stats[domain_code] = {
                    'pages': pages,
                    'estimated_tables': 0,
                    'confidence': domain.confidence_score
                }
        
        context = {
            'pdf': pdf_document,
            'domains': detected_domains,
            'domain_stats': domain_stats
        }
        return render(request, 'extractor/domain_selection.html', context)
    
    def post(self, request, pdf_id):
        """Handle domain selection with validation"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        
        # Update selected status for domains
        selected_domains = request.POST.getlist('selected_domains')
        
        if not selected_domains:
            messages.warning(request, "Please select at least one domain to process")
            return redirect('extractor:select_domains', pdf_id=pdf_document.id)
        
        for detected_domain in DetectedDomain.objects.filter(pdf=pdf_document):
            domain_id = str(detected_domain.domain.code)
            detected_domain.selected = domain_id in selected_domains
            
            # Update pages if provided
            pages_key = f"pages_{domain_id}"
            if pages_key in request.POST:
                detected_domain.pages = request.POST[pages_key]
                
            detected_domain.save()
        
        messages.success(request, f"Selected {len(selected_domains)} domains for processing")
        
        # Redirect to processing
        return redirect('extractor:process_domains', pdf_id=pdf_document.id)

class ProcessDomainsView(View):
    """View for processing selected domains with enhanced extraction"""
    
    def get(self, request, pdf_id):
        """Display processing page"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        selected_domains = DetectedDomain.objects.filter(pdf=pdf_document, selected=True)
        
        # If already completed, redirect to results
        if pdf_document.status == 'COMPLETED':
            return redirect('extractor:results', pdf_id=pdf_document.id)
        
        # Calculate estimated processing time
        estimated_time = len(selected_domains) * 30  # seconds per domain
        
        context = {
            'pdf': pdf_document,
            'domains': selected_domains,
            'estimated_time': estimated_time,
            'domain_count': len(selected_domains)
        }
        return render(request, 'extractor/processing.html', context)
    
    def post(self, request, pdf_id):
        """Start domain processing with enhanced extraction"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        
        try:
            # Use enhanced data extractor
            extractor = EnhancedDataExtractor()
            extracted_data = extractor.extract_all_selected_domains(pdf_document)
            
            if not extracted_data:
                messages.warning(request, "No data was extracted from selected domains")
            else:
                messages.success(request, f"Successfully extracted data from {len(extracted_data)} domains")
            
            # Redirect to results
            return redirect('extractor:results', pdf_id=pdf_document.id)
            
        except Exception as e:
            messages.error(request, f"Error processing domains: {str(e)}")
            return redirect('extractor:select_domains', pdf_id=pdf_document.id)

class DomainProcessingStatusView(View):
    """AJAX view for checking domain processing status with detailed progress"""
    
    def get(self, request, pdf_id):
        """Return JSON with detailed processing status"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        selected_domains = DetectedDomain.objects.filter(pdf=pdf_document, selected=True)
        extracted_domains = ExtractedData.objects.filter(pdf=pdf_document)
        
        # Calculate progress
        total_domains = selected_domains.count()
        processed_domains = extracted_domains.count()
        
        # Get processing details
        processing_details = []
        for domain in selected_domains:
            extracted = ExtractedData.objects.filter(pdf=pdf_document, domain=domain.domain).first()
            if extracted:
                metadata = extracted.data.get('extraction_metadata', {})
                processing_details.append({
                    'domain': domain.domain.code,
                    'status': 'completed',
                    'confidence': metadata.get('confidence', 0),
                    'strategy': metadata.get('strategy_used', 'unknown'),
                    'tables_extracted': metadata.get('tables_extracted', 0),
                    'validation_errors': len([v for v in metadata.get('validation_results', []) if v['severity'] == 'ERROR'])
                })
            else:
                processing_details.append({
                    'domain': domain.domain.code,
                    'status': 'pending' if pdf_document.status == 'PROCESSING' else 'not_started'
                })
        
        # Get currently processing domain
        current_domain = None
        if pdf_document.status == 'PROCESSING' and processed_domains < total_domains:
            for detail in processing_details:
                if detail['status'] == 'pending':
                    current_domain = detail['domain']
                    break
        
        return JsonResponse({
            'status': pdf_document.status,
            'total_domains': total_domains,
            'processed_domains': processed_domains,
            'current_domain': current_domain,
            'progress_percent': int(processed_domains / max(total_domains, 1) * 100),
            'processing_details': processing_details,
            'estimated_time_remaining': max(0, (total_domains - processed_domains) * 30)
        })

class ResultsView(View):
    """View for displaying extraction results with validation details"""
    
    def get(self, request, pdf_id):
        """Display results page with enhanced information"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        extracted_data = ExtractedData.objects.filter(pdf=pdf_document)
        
        # Prepare detailed results
        results_with_details = []
        for data in extracted_data:
            metadata = data.data.get('extraction_metadata', {})
            validation_results = metadata.get('validation_results', [])
            
            # Count validation issues
            errors = len([v for v in validation_results if v['severity'] == 'ERROR'])
            warnings = len([v for v in validation_results if v['severity'] == 'WARNING'])
            
            results_with_details.append({
                'data': data,
                'record_count': len(data.data.get('records', [])),
                'confidence': metadata.get('confidence', 0),
                'strategy': metadata.get('strategy_used', 'unknown'),
                'tables_extracted': metadata.get('tables_extracted', 0),
                'text_blocks_extracted': metadata.get('text_blocks_extracted', 0),
                'validation_errors': errors,
                'validation_warnings': warnings,
                'validation_results': validation_results
            })
        
        context = {
            'pdf': pdf_document,
            'extracted_data': results_with_details,
            'total_records': sum(result['record_count'] for result in results_with_details),
            'total_errors': sum(result['validation_errors'] for result in results_with_details),
            'total_warnings': sum(result['validation_warnings'] for result in results_with_details)
        }
        return render(request, 'extractor/results.html', context)

class ViewXPTFileView(View):
    """View for displaying XPT file content with validation details"""
    
    def get(self, request, data_id):
        """Display XPT file content with enhanced features"""
        extracted_data = get_object_or_404(ExtractedData, id=data_id)
        
        # Prepare data with validation results
        data_with_validation = extracted_data.data.copy()
        metadata = data_with_validation.get('extraction_metadata', {})
        
        # Revalidate data for current display
        generator = EnhancedXPTGenerator()
        df = pd.DataFrame(data_with_validation.get('records', []))
        validation_results = generator.validate_domain_data(extracted_data.domain.code, df)
        
        # Group validation results by field
        validation_by_field = {}
        for result in validation_results:
            if result.field:
                if result.field not in validation_by_field:
                    validation_by_field[result.field] = []
                validation_by_field[result.field].append({
                    'severity': result.severity.value,
                    'message': result.message,
                    'value': result.value
                })
        
        # Generate Define-XML for this domain
        define_xml = generator.generate_define_xml(extracted_data.domain.code, df)
        
        context = {
            'extracted_data': extracted_data,
            'data_json': json.dumps(data_with_validation, indent=2),
            'validation_results': validation_results,
            'validation_by_field': validation_by_field,
            'extraction_metadata': metadata,
            'define_xml': define_xml,
            'record_count': len(data_with_validation.get('records', [])),
            'field_count': len(df.columns) if not df.empty else 0
        }
        return render(request, 'extractor/view_xpt.html', context)

class DownloadXPTFileView(View):
    """View for downloading XPT file with validation report"""
    
    def get(self, request, data_id):
        """Download XPT file with optional validation report"""
        extracted_data = get_object_or_404(ExtractedData, id=data_id)
        
        if not extracted_data.xpt_file:
            messages.error(request, "XPT file not found")
            return redirect('extractor:results', pdf_id=extracted_data.pdf.id)
        
        include_report = request.GET.get('include_report', 'false').lower() == 'true'
        
        if include_report:
            # Create ZIP with XPT and validation report
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add XPT file
                zip_file.writestr(f"{extracted_data.domain.code}.xpt", extracted_data.xpt_file.read())
                
                # Add validation report
                generator = EnhancedXPTGenerator()
                df = pd.DataFrame(extracted_data.data.get('records', []))
                validation_results = generator.validate_domain_data(extracted_data.domain.code, df)
                
                report_content = self._generate_validation_report(extracted_data, validation_results)
                zip_file.writestr(f"{extracted_data.domain.code}_validation_report.txt", report_content)
                
                # Add Define-XML
                define_xml = generator.generate_define_xml(extracted_data.domain.code, df)
                zip_file.writestr(f"{extracted_data.domain.code}_define.xml", define_xml)
            
            # Prepare ZIP response
            response = HttpResponse(
                zip_buffer.getvalue(),
                content_type='application/zip'
            )
            response['Content-Disposition'] = f'attachment; filename="{extracted_data.domain.code}_package.zip"'
        else:
            # Just download XPT file
            response = HttpResponse(
                extracted_data.xpt_file.read(),
                content_type='application/octet-stream'
            )
            response['Content-Disposition'] = f'attachment; filename="{extracted_data.domain.code}.xpt"'
        
        return response
    
    def _generate_validation_report(self, extracted_data, validation_results):
        """Generate a text-based validation report"""
        report_lines = [
            f"SENDIG Domain Validation Report",
            f"================================",
            f"",
            f"Domain: {extracted_data.domain.code} - {extracted_data.domain.name}",
            f"Extraction Date: {extracted_data.extraction_date}",
            f"Records: {len(extracted_data.data.get('records', []))}",
            f"",
            f"Extraction Details:",
            f"------------------"
        ]
        
        # Add extraction metadata
        metadata = extracted_data.data.get('extraction_metadata', {})
        if metadata:
            report_lines.extend([
                f"Strategy Used: {metadata.get('strategy_used', 'unknown')}",
                f"Confidence: {metadata.get('confidence', 0):.2f}",
                f"Tables Extracted: {metadata.get('tables_extracted', 0)}",
                f"Text Blocks Extracted: {metadata.get('text_blocks_extracted', 0)}",
                f""
            ])
        
        # Add validation results
        report_lines.extend([
            f"Validation Results:",
            f"------------------"
        ])
        
        errors = [v for v in validation_results if v.severity.value == 'ERROR']
        warnings = [v for v in validation_results if v.severity.value == 'WARNING']
        
        report_lines.append(f"Total Errors: {len(errors)}")
        report_lines.append(f"Total Warnings: {len(warnings)}")
        report_lines.append(f"")
        
        if errors:
            report_lines.append("ERRORS:")
            for error in errors:
                field_info = f" (Field: {error.field})" if error.field else ""
                value_info = f" (Value: {error.value})" if error.value else ""
                report_lines.append(f"  - {error.message}{field_info}{value_info}")
            report_lines.append("")
        
        if warnings:
            report_lines.append("WARNINGS:")
            for warning in warnings:
                field_info = f" (Field: {warning.field})" if warning.field else ""
                value_info = f" (Value: {warning.value})" if warning.value else ""
                report_lines.append(f"  - {warning.message}{field_info}{value_info}")
        
        return "\n".join(report_lines)

class DownloadAllFilesView(View):
    """View for downloading all XPT files with complete submission package"""
    
    def get(self, request, pdf_id):
        """Download complete submission package"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        
        try:
            # Create submission package using enhanced extractor
            extractor = EnhancedDataExtractor()
            
            # Create temporary directory for package
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create submission package
                created_files = extractor.create_submission_package(pdf_document, temp_dir)
                
                # Create ZIP archive
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for file_path in created_files:
                        # Add file to zip with relative path
                        arcname = os.path.relpath(file_path, temp_dir)
                        zip_file.write(file_path, arcname)
                    
                    # Add extraction summary
                    summary = self._generate_extraction_summary(pdf_document)
                    zip_file.writestr("extraction_summary.txt", summary)
                
                # Prepare response
                response = HttpResponse(
                    zip_buffer.getvalue(),
                    content_type='application/zip'
                )
                response['Content-Disposition'] = f'attachment; filename="sendig_submission_{pdf_document.id}.zip"'
                
                return response
                
        except Exception as e:
            messages.error(request, f"Error creating submission package: {str(e)}")
            return redirect('extractor:results', pdf_id=pdf_document.id)
    
    def _generate_extraction_summary(self, pdf_document):
        """Generate a summary of the extraction process"""
        extracted_data_list = ExtractedData.objects.filter(pdf=pdf_document)
        
        summary_lines = [
            "SENDIG Extraction Summary",
            "========================",
            f"",
            f"Document: {pdf_document.name}",
            f"Upload Date: {pdf_document.uploaded_at}",
            f"Processing Date: {pdf_document.updated_at if hasattr(pdf_document, 'updated_at') else 'N/A'}",
            f"Status: {pdf_document.status}",
            f"",
            f"Extracted Domains:",
            f"-----------------"
        ]
        
        total_records = 0
        total_errors = 0
        total_warnings = 0
        
        for data in extracted_data_list:
            metadata = data.data.get('extraction_metadata', {})
            validation_results = metadata.get('validation_results', [])
            
            records = len(data.data.get('records', []))
            errors = len([v for v in validation_results if v['severity'] == 'ERROR'])
            warnings = len([v for v in validation_results if v['severity'] == 'WARNING'])
            
            summary_lines.extend([
                f"",
                f"Domain: {data.domain.code} - {data.domain.name}",
                f"  Records: {records}",
                f"  Confidence: {metadata.get('confidence', 0):.2f}",
                f"  Strategy: {metadata.get('strategy_used', 'unknown')}",
                f"  Validation Errors: {errors}",
                f"  Validation Warnings: {warnings}"
            ])
            
            total_records += records
            total_errors += errors
            total_warnings += warnings
        
        summary_lines.extend([
            f"",
            f"Overall Statistics:",
            f"------------------",
            f"Total Domains: {len(extracted_data_list)}",
            f"Total Records: {total_records}",
            f"Total Validation Errors: {total_errors}",
            f"Total Validation Warnings: {total_warnings}",
            f"",
            f"Submission Package Contents:",
            f"---------------------------",
            f"- Individual XPT files for each domain",
            f"- Define-XML metadata file",
            f"- Original study report (PDF)",
            f"- Individual validation reports",
            f"- This summary file"
        ])
        
        return "\n".join(summary_lines)

class ValidationReportView(View):
    """View for generating comprehensive validation reports"""
    
    def get(self, request, pdf_id):
        """Generate validation report for all domains"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        extracted_data_list = ExtractedData.objects.filter(pdf=pdf_document)
        
        # Generate comprehensive validation report
        generator = EnhancedXPTGenerator()
        validation_summary = {
            'domains': [],
            'overall': {
                'total_errors': 0,
                'total_warnings': 0,
                'total_records': 0,
                'domains_with_errors': 0
            }
        }
        
        for data in extracted_data_list:
            df = pd.DataFrame(data.data.get('records', []))
            validation_results = generator.validate_domain_data(data.domain.code, df)
            
            errors = [v for v in validation_results if v.severity.value == 'ERROR']
            warnings = [v for v in validation_results if v.severity.value == 'WARNING']
            
            domain_summary = {
                'domain_code': data.domain.code,
                'domain_name': data.domain.name,
                'record_count': len(df),
                'errors': errors,
                'warnings': warnings,
                'error_count': len(errors),
                'warning_count': len(warnings),
                'metadata': data.data.get('extraction_metadata', {})
            }
            
            validation_summary['domains'].append(domain_summary)
            validation_summary['overall']['total_errors'] += len(errors)
            validation_summary['overall']['total_warnings'] += len(warnings)
            validation_summary['overall']['total_records'] += len(df)
            
            if errors:
                validation_summary['overall']['domains_with_errors'] += 1
        
        context = {
            'pdf': pdf_document,
            'validation_summary': validation_summary
        }
        
        return render(request, 'extractor/validation_report.html', context)

# New AJAX views for real-time updates
class ProcessingStatusAPIView(View):
    """API view for real-time processing status updates"""
    
    def get(self, request, pdf_id):
        """Get detailed processing status for real-time updates"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        
        # Get current processing state
        status_data = {
            'pdf_status': pdf_document.status,
            'domains': []
        }
        
        # Get all selected domains and their status
        selected_domains = DetectedDomain.objects.filter(pdf=pdf_document, selected=True)
        
        for domain in selected_domains:
            extracted_data = ExtractedData.objects.filter(pdf=pdf_document, domain=domain.domain).first()
            
            domain_status = {
                'domain_code': domain.domain.code,
                'domain_name': domain.domain.name,
                'status': 'completed' if extracted_data else 'pending',
                'pages': domain.pages,
                'confidence': domain.confidence_score
            }
            
            if extracted_data:
                metadata = extracted_data.data.get('extraction_metadata', {})
                domain_status.update({
                    'record_count': len(extracted_data.data.get('records', [])),
                    'extraction_confidence': metadata.get('confidence', 0),
                    'strategy_used': metadata.get('strategy_used', 'unknown'),
                    'validation_errors': len([v for v in metadata.get('validation_results', []) if v['severity'] == 'ERROR']),
                    'validation_warnings': len([v for v in metadata.get('validation_results', []) if v['severity'] == 'WARNING'])
                })
            
            status_data['domains'].append(domain_status)
        
        return JsonResponse(status_data)

class DomainPreviewAPIView(View):
    """API view for previewing domain data before full extraction"""
    
    def get(self, request, pdf_id, domain_code):
        """Get preview of domain data without full extraction"""
        pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
        domain = get_object_or_404(SENDIGDomain, code=domain_code)
        detected_domain = get_object_or_404(DetectedDomain, pdf=pdf_document, domain=domain)
        
        try:
            # Quick extraction for preview
            pdf_service = EnhancedPDFService()
            pages = detected_domain.get_pages_list()[:2]  # Preview first 2 pages only
            
            domain_info = {
                domain_code: {
                    'pages': pages,
                    'keywords': [domain_code, domain.name]
                }
            }
            
            # Extract sample data
            preview_results = pdf_service.extract_domain_data(pdf_document.file.path, domain_info)
            domain_data = preview_results.get(domain_code)
            
            if domain_data:
                preview_info = {
                    'domain_code': domain_code,
                    'pages_previewed': pages,
                    'tables_found': len(domain_data.tables),
                    'text_blocks_found': len(domain_data.text_blocks),
                    'sample_tables': [],
                    'sample_text': domain_data.text_blocks[:2] if domain_data.text_blocks else []
                }
                
                # Convert tables to preview format
                for table in domain_data.tables[:2]:  # Preview first 2 tables
                    preview_info['sample_tables'].append({
                        'page': table.page_number,
                        'columns': list(table.data.columns),
                        'rows': len(table.data),
                        'preview': table.data.head(3).to_dict(orient='records')
                    })
                
                return JsonResponse(preview_info)
            else:
                return JsonResponse({'error': 'No data found for preview'}, status=404)
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

# Usage example
if __name__ == "__main__":
    # These views would be connected to URL patterns
    # and used in the Django application
    pass