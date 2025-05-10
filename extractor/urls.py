from django.urls import path
from . import views
from . import api_views

app_name = 'extractor'

urlpatterns = [
    # Home page
    path('', views.HomeView.as_view(), name='home'),
    
    # PDF Upload
    path('upload/', views.UploadPDFView.as_view(), name='upload'),
    
    # Domain Detection
    path('detect-domains/<uuid:pdf_id>/', views.DetectDomainsView.as_view(), name='detect_domains'),
    
    # Domain Selection
    path('select-domains/<uuid:pdf_id>/', views.DomainSelectionView.as_view(), name='select_domains'),
    
    # Domain Processing
    path('process-domains/<uuid:pdf_id>/', views.ProcessDomainsView.as_view(), name='process_domains'),
    path('processing-status/<uuid:pdf_id>/', views.DomainProcessingStatusView.as_view(), name='processing_status'),
    
    # Results
    path('results/<uuid:pdf_id>/', views.ResultsView.as_view(), name='results'),
    
    # XPT File Actions
    path('view-xpt/<int:data_id>/', views.ViewXPTFileView.as_view(), name='view_xpt'),
    path('download-xpt/<int:data_id>/', views.DownloadXPTFileView.as_view(), name='download_xpt'),
    path('download-all/<uuid:pdf_id>/', views.DownloadAllFilesView.as_view(), name='download_all'),

     # API Endpoints
    path('api/detect-domains/', api_views.api_detect_domains, name='api_detect_domains'),
    path('api/domain-detection-status/<uuid:pdf_id>/', api_views.api_domain_detection_status, name='api_domain_detection_status'),
]