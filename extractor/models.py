from django.db import models
import json
import uuid
import os

def pdf_upload_path(instance, filename):
    """Generate a unique file path for uploaded PDFs"""
    ext = filename.split('.')[-1]
    filename = f"{instance.id}.{ext}"
    return os.path.join('pdfs', filename)

class PDFDocument(models.Model):
    """Model to store uploaded PDF documents and their processing status"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to=pdf_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=50,
        choices=(
            ('UPLOADED', 'Uploaded'),
            ('ANALYZED', 'Analyzed'),
            ('PROCESSING', 'Processing'),
            ('COMPLETED', 'Completed'),
            ('FAILED', 'Failed')
        ),
        default='UPLOADED'
    )
    
    def __str__(self):
        return self.name

class SENDIGDomain(models.Model):
    """Model for SENDIG standard domains"""
    code = models.CharField(max_length=10, primary_key=True)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    required_variables = models.TextField(blank=True, null=True)  # JSON list of required variables
    
    def get_required_variables(self):
        """Returns the required variables as a Python list"""
        if self.required_variables:
            return json.loads(self.required_variables)
        return []
    
    def __str__(self):
        return f"{self.code} - {self.name}"

class DetectedDomain(models.Model):
    """Model to store domains detected in a specific PDF"""
    pdf = models.ForeignKey(PDFDocument, on_delete=models.CASCADE, related_name='detected_domains')
    domain = models.ForeignKey(SENDIGDomain, on_delete=models.CASCADE)
    pages = models.CharField(max_length=255, help_text="Comma-separated list of page numbers or ranges")
    selected = models.BooleanField(default=True)
    confidence_score = models.FloatField(default=0.0)
    
    class Meta:
        unique_together = ('pdf', 'domain')
    
    def __str__(self):
        return f"{self.domain} in {self.pdf.name}"
    
    def get_pages_list(self):
        """Returns the pages as a list of integers"""
        result = []
        if not self.pages:
            return result
            
        parts = self.pages.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                result.extend(range(start, end + 1))
            else:
                result.append(int(part))
        return result

class ExtractedData(models.Model):
    """Model to store extracted domain data"""
    pdf = models.ForeignKey(PDFDocument, on_delete=models.CASCADE, related_name='extracted_data')
    domain = models.ForeignKey(SENDIGDomain, on_delete=models.CASCADE)
    data = models.JSONField(default=dict)
    xpt_file = models.FileField(upload_to='xpt_files', blank=True, null=True)
    extraction_date = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('pdf', 'domain')
    
    def __str__(self):
        return f"{self.domain} data from {self.pdf.name}"