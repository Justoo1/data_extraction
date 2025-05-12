import logging
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from django.core.files.base import ContentFile
from django.utils import timezone
import pandas as pd

from .python_service import EnhancedPDFService, TableStructure, DomainData
from .llm_service import LLMService
from .xpt_generator_enhanced import EnhancedXPTGenerator, ValidationResult
from ..models import PDFDocument, DetectedDomain, ExtractedData, SENDIGDomain

logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    """Results from data extraction process"""
    success: bool
    domain_code: str
    extracted_data: Optional[Dict]
    validation_results: List[ValidationResult]
    confidence: float
    tables_extracted: int
    text_blocks_extracted: int
    
class EnhancedDataExtractor:
    """
    Advanced data extractor using enhanced PDF processing,
    AI-powered extraction, and SENDIG validation
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self.pdf_service = EnhancedPDFService(use_ocr=True)
        self.xpt_generator = EnhancedXPTGenerator()
        
        # Initialize extraction strategies
        self.extraction_strategies = {
            'table': self._extract_from_tables,
            'text': self._extract_from_text,
            'hybrid': self._extract_hybrid,
            'llm': self._extract_with_llm
        }
    
    def extract_data_for_domain(self, 
                              pdf_document: PDFDocument, 
                              detected_domain: DetectedDomain) -> ExtractedData:
        """
        Extract structured data for a specific domain using multiple strategies
        
        Args:
            pdf_document: PDFDocument instance
            detected_domain: DetectedDomain instance
            
        Returns:
            ExtractedData instance
        """
        try:
            domain = detected_domain.domain
            domain_code = domain.code
            
            logger.info(f"Starting extraction for domain {domain_code}")
            
            # Get domain information
            domain_info = {
                domain_code: {
                    'pages': detected_domain.get_pages_list(),
                    'keywords': self._get_domain_keywords(domain_code),
                    'required_variables': domain.get_required_variables()
                }
            }
            
            # Extract domain data using enhanced PDF service
            domain_results = self.pdf_service.extract_domain_data(
                pdf_document.file.path, 
                domain_info
            )
            
            domain_data = domain_results.get(domain_code)
            if not domain_data:
                raise Exception(f"No data found for domain {domain_code}")
            
            # Try multiple extraction strategies
            extraction_results = []
            
            # Strategy 1: Extract from tables
            if domain_data.tables:
                table_result = self._extract_from_tables(domain_data)
                extraction_results.append(table_result)
            
            # Strategy 2: Extract from text blocks
            if domain_data.text_blocks:
                text_result = self._extract_from_text(domain_data)
                extraction_results.append(text_result)
            
            # Strategy 3: Hybrid extraction
            hybrid_result = self._extract_hybrid(domain_data)
            extraction_results.append(hybrid_result)
            
            # Strategy 4: LLM-based extraction as fallback
            if all(r.confidence < 0.7 for r in extraction_results):
                llm_result = self._extract_with_llm(domain_data)
                extraction_results.append(llm_result)
            
            # Select best result
            best_result = max(extraction_results, key=lambda x: x.confidence)
            
            if not best_result.success or not best_result.extracted_data:
                raise Exception(f"Failed to extract data for domain {domain_code}")
            
            # Validate extracted data
            df = pd.DataFrame(best_result.extracted_data.get('records', []))
            validation_results = self.xpt_generator.validate_domain_data(domain_code, df)
            
            # Generate XPT file
            xpt_data = self.xpt_generator.generate_xpt(domain_code, best_result.extracted_data)
            
            # Save extracted data
            extracted_obj, created = ExtractedData.objects.update_or_create(
                pdf=pdf_document,
                domain=domain,
                defaults={
                    'data': best_result.extracted_data
                }
            )
            
            # Save XPT file
            xpt_filename = f"{pdf_document.id}_{domain_code}.xpt"
            extracted_obj.xpt_file.save(xpt_filename, ContentFile(xpt_data))
            
            # Add extraction metadata
            extracted_obj.data['extraction_metadata'] = {
                'strategy_used': best_result.strategy,
                'confidence': best_result.confidence,
                'validation_results': [
                    {
                        'severity': v.severity.value,
                        'message': v.message,
                        'field': v.field,
                        'value': v.value
                    } for v in validation_results
                ],
                'tables_extracted': best_result.tables_extracted,
                'text_blocks_extracted': best_result.text_blocks_extracted
            }
            extracted_obj.save()
            
            return extracted_obj
            
        except Exception as e:
            logger.error(f"Error extracting data for domain {domain_code}: {str(e)}")
            raise
    
    def _get_domain_keywords(self, domain_code: str) -> List[str]:
        """Get comprehensive keywords for a domain"""
        keyword_mapping = {
            'DM': ['demographics', 'subject', 'animal', 'species', 'strain', 'sex', 'age'],
            'BW': ['body weight', 'weight', 'bw', 'terminal weight', 'weekly weight'],
            'LB': ['laboratory', 'lab', 'clinical chemistry', 'hematology', 'test', 'result'],
            'CL': ['clinical', 'observation', 'cage side', 'behavioral', 'sign'],
            'MA': ['macroscopic', 'gross pathology', 'necropsy', 'organ', 'tissue'],
            'MI': ['microscopic', 'histopathology', 'pathology', 'finding'],
            'EX': ['exposure', 'dose', 'administration', 'treatment', 'vehicle'],
            'PC': ['pharmacokinetic', 'concentration', 'plasma', 'pk', 'sampling'],
            'PP': ['parameter', 'cmax', 'tmax', 'auc', 'pharmacokinetic'],
        }
        
        return keyword_mapping.get(domain_code, [domain_code])
    
    def _extract_from_tables(self, domain_data: DomainData) -> ExtractionResult:
        """Extract data from tables using pattern matching"""
        all_records = []
        
        try:
            for table in domain_data.tables:
                # Clean and standardize table
                cleaned_table = self._clean_table(table.data)
                
                # Convert table to records
                records = self._table_to_records(cleaned_table, domain_data.domain_code)
                all_records.extend(records)
            
            confidence = 0.8 if all_records else 0.0
            
            return ExtractionResult(
                success=len(all_records) > 0,
                domain_code=domain_data.domain_code,
                extracted_data={'records': all_records},
                validation_results=[],
                confidence=confidence,
                tables_extracted=len(domain_data.tables),
                text_blocks_extracted=0
            )
            
        except Exception as e:
            logger.error(f"Table extraction failed: {str(e)}")
            return ExtractionResult(
                success=False,
                domain_code=domain_data.domain_code,
                extracted_data=None,
                validation_results=[],
                confidence=0.0,
                tables_extracted=0,
                text_blocks_extracted=0
            )
    
    def _extract_from_text(self, domain_data: DomainData) -> ExtractionResult:
        """Extract data from text blocks using pattern matching"""
        all_records = []
        
        try:
            for text_block in domain_data.text_blocks:
                # Parse text block based on domain
                records = self._parse_text_block(text_block, domain_data.domain_code)
                all_records.extend(records)
            
            confidence = min(0.6 + len(all_records) * 0.05, 0.8)
            
            return ExtractionResult(
                success=len(all_records) > 0,
                domain_code=domain_data.domain_code,
                extracted_data={'records': all_records},
                validation_results=[],
                confidence=confidence,
                tables_extracted=0,
                text_blocks_extracted=len(domain_data.text_blocks)
            )
            
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            return ExtractionResult(
                success=False,
                domain_code=domain_data.domain_code,
                extracted_data=None,
                validation_results=[],
                confidence=0.0,
                tables_extracted=0,
                text_blocks_extracted=0
            )
    
    def _extract_hybrid(self, domain_data: DomainData) -> ExtractionResult:
        """Extract data using combination of tables and text"""
        try:
            # Extract from tables first
            table_result = self._extract_from_tables(domain_data)
            
            # Extract from text as supplement
            text_result = self._extract_from_text(domain_data)
            
            # Combine results
            combined_records = []
            
            if table_result.success and table_result.extracted_data:
                combined_records.extend(table_result.extracted_data.get('records', []))
            
            if text_result.success and text_result.extracted_data:
                # Add text records that don't duplicate table data
                text_records = text_result.extracted_data.get('records', [])
                for text_record in text_records:
                    if not self._is_duplicate_record(text_record, combined_records):
                        combined_records.append(text_record)
            
            # Calculate confidence based on both sources
            confidence = min(
                (table_result.confidence * 0.6 + text_result.confidence * 0.4),
                0.85
            )
            
            return ExtractionResult(
                success=len(combined_records) > 0,
                domain_code=domain_data.domain_code,
                extracted_data={'records': combined_records},
                validation_results=[],
                confidence=confidence,
                tables_extracted=table_result.tables_extracted,
                text_blocks_extracted=text_result.text_blocks_extracted
            )
            
        except Exception as e:
            logger.error(f"Hybrid extraction failed: {str(e)}")
            return ExtractionResult(
                success=False,
                domain_code=domain_data.domain_code,
                extracted_data=None,
                validation_results=[],
                confidence=0.0,
                tables_extracted=0,
                text_blocks_extracted=0
            )
    
    def _extract_with_llm(self, domain_data: DomainData) -> ExtractionResult:
        """Extract data using LLM as fallback"""
        try:
            # Combine all available data
            combined_text = ""
            
            # Add text blocks
            if domain_data.text_blocks:
                combined_text += "\n\n=== TEXT BLOCKS ===\n"
                combined_text += "\n\n".join(domain_data.text_blocks)
            
            # Add tables as text
            if domain_data.tables:
                combined_text += "\n\n=== TABLES ===\n"
                for table in domain_data.tables:
                    combined_text += f"\nTable on page {table.page_number}:\n"
                    combined_text += table.data.to_string()
            
            # Get required variables for domain
            domain = SENDIGDomain.objects.get(code=domain_data.domain_code)
            required_variables = domain.get_required_variables()
            
            # Extract using LLM
            extracted_data = self.llm_service.extract_structured_data(
                combined_text,
                domain_data.domain_code,
                required_variables
            )
            
            confidence = 0.6  # Default LLM confidence
            
            return ExtractionResult(
                success=bool(extracted_data and extracted_data.get('records')),
                domain_code=domain_data.domain_code,
                extracted_data=extracted_data,
                validation_results=[],
                confidence=confidence,
                tables_extracted=len(domain_data.tables),
                text_blocks_extracted=len(domain_data.text_blocks)
            )
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {str(e)}")
            return ExtractionResult(
                success=False,
                domain_code=domain_data.domain_code,
                extracted_data=None,
                validation_results=[],
                confidence=0.0,
                tables_extracted=0,
                text_blocks_extracted=0
            )
    
    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize table data"""
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = [self._clean_column_name(col) for col in df.columns]
        
        # Remove header rows that aren't actually headers
        if len(df) > 1:
            # Check if first row might be a continuation of headers
            first_row_numeric = pd.to_numeric(df.iloc[0], errors='coerce').notna().sum()
            if first_row_numeric < len(df.columns) * 0.5:
                # Likely another header row, combine with column names
                new_headers = []
                for i, col in enumerate(df.columns):
                    header_part = str(df.iloc[0, i]).strip()
                    if header_part and header_part != 'nan':
                        new_headers.append(f"{col}_{header_part}")
                    else:
                        new_headers.append(col)
                df.columns = new_headers
                df = df.iloc[1:]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _clean_column_name(self, name: str) -> str:
        """Clean column name for standardization"""
        if pd.isna(name):
            return 'UNNAMED'
        
        # Convert to string and clean
        name = str(name).strip()
        
        # Remove special characters and replace with underscore
        import re
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'[-\s]+', '_', name)
        
        # Convert to uppercase
        name = name.upper()
        
        return name
    
    def _table_to_records(self, df: pd.DataFrame, domain_code: str) -> List[Dict]:
        """Convert table to domain records"""
        records = []
        
        # Get domain metadata for mapping
        domain_metadata = self.xpt_generator.sendig_metadata.get(domain_code, {})
        
        # Create column mapping
        column_mapping = self._create_column_mapping(df.columns, domain_metadata.keys())
        
        for idx, row in df.iterrows():
            record = {}
            
            # Add required standard fields
            record['STUDYID'] = self._extract_study_id(df) or ''
            record['DOMAIN'] = domain_code
            
            # Map table columns to SENDIG variables
            for table_col, sendig_var in column_mapping.items():
                if table_col in df.columns and sendig_var in domain_metadata:
                    value = row[table_col]
                    if pd.notna(value):
                        record[sendig_var] = value
            
            records.append(record)
        
        return records
    
    def _create_column_mapping(self, table_columns: List[str], sendig_variables: List[str]) -> Dict[str, str]:
        """Create mapping between table columns and SENDIG variables"""
        mapping = {}
        
        # Exact matches first
        for col in table_columns:
            if col.upper() in sendig_variables:
                mapping[col] = col.upper()
        
        # Partial matches
        remaining_cols = [col for col in table_columns if col not in mapping]
        remaining_vars = [var for var in sendig_variables if var not in mapping.values()]
        
        for col in remaining_cols:
            col_clean = col.upper().replace('_', '')
            
            for var in remaining_vars:
                var_clean = var.replace('_', '')
                
                # Check if column name is contained in variable name or vice versa
                if col_clean in var_clean or var_clean in col_clean:
                    mapping[col] = var
                    break
        
        return mapping
    
    def _extract_study_id(self, df: pd.DataFrame) -> Optional[str]:
        """Extract study ID from table if available"""
        for col in df.columns:
            if 'STUDY' in col.upper():
                values = df[col].dropna().unique()
                if len(values) > 0:
                    return str(values[0])
        return None
    
    def _parse_text_block(self, text: str, domain_code: str) -> List[Dict]:
        """Parse text block for domain data"""
        records = []
        
        # Domain-specific parsing strategies
        if domain_code == 'DM':
            records = self._parse_dm_text(text)
        elif domain_code == 'BW':
            records = self._parse_bw_text(text)
        elif domain_code == 'LB':
            records = self._parse_lb_text(text)
        else:
            # Generic parsing for other domains
            records = self._parse_generic_text(text, domain_code)
        
        return records
    
    def _parse_dm_text(self, text: str) -> List[Dict]:
        """Parse Demographics text block"""
        records = []
        
        # Look for subject information patterns
        import re
        subject_pattern = r'subject\s*(?:id|#)?:?\s*(\w+)'
        species_pattern = r'species:?\s*([a-zA-Z\s]+)'
        sex_pattern = r'sex:?\s*([MF])'
        age_pattern = r'age:?\s*(\d+(?:\.\d+)?)\s*(\w+)?'
        
        subjects = re.findall(subject_pattern, text, re.IGNORECASE)
        for subject_id in subjects:
            record = {
                'STUDYID': '',
                'DOMAIN': 'DM',
                'USUBJID': subject_id,
                'SUBJID': subject_id
            }
            
            # Extract additional demographics
            species_match = re.search(species_pattern, text, re.IGNORECASE)
            if species_match:
                record['SPECIES'] = species_match.group(1).strip()
            
            sex_match = re.search(sex_pattern, text, re.IGNORECASE)
            if sex_match:
                record['SEX'] = sex_match.group(1).upper()
            
            age_match = re.search(age_pattern, text, re.IGNORECASE)
            if age_match:
                record['AGE'] = float(age_match.group(1))
                if age_match.group(2):
                    record['AGEU'] = age_match.group(2).upper()
            
            records.append(record)
        
        return records
    
    def _parse_bw_text(self, text: str) -> List[Dict]:
        """Parse Body Weight text block"""
        records = []
        
        # Look for body weight measurements
        import re
        weight_pattern = r'(\w+)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*([kg])'
        date_pattern = r'date:?\s*(\d{4}-\d{2}-\d{2})'
        
        weight_matches = re.findall(weight_pattern, text, re.IGNORECASE)
        for match in weight_matches:
            subject_id, weight, unit = match
            
            record = {
                'STUDYID': '',
                'DOMAIN': 'BW',
                'USUBJID': subject_id,
                'BWTESTCD': 'TERMBW' if 'term' in text.lower() else 'BW',
                'BWTEST': 'Terminal Body Weight' if 'term' in text.lower() else 'Body Weight',
                'BWORRES': weight,
                'BWORRESU': unit.upper(),
                'BWSTRESN': float(weight),
                'BWSTRESU': unit.upper()
            }
            
            # Extract date if available
            date_match = re.search(date_pattern, text)
            if date_match:
                record['BWDTC'] = date_match.group(1)
            
            records.append(record)
        
        return records
    
    def _parse_lb_text(self, text: str) -> List[Dict]:
        """Parse Laboratory Results text block"""
        records = []
        
        # Look for lab test patterns
        import re
        test_pattern = r'(\w+)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*([a-zA-Z/]+)?'
        
        test_matches = re.findall(test_pattern, text)
        for match in test_matches:
            test_name, value, unit = match
            
            # Skip if this doesn't look like a lab test
            if not unit or test_name.upper() in ['SUBJECT', 'ID', 'DATE']:
                continue
            
            record = {
                'STUDYID': '',
                'DOMAIN': 'LB',
                'USUBJID': '',  # Will be filled by context
                'LBTESTCD': self._standardize_lab_code(test_name),
                'LBTEST': test_name,
                'LBORRES': value,
                'LBORRESU': unit,
                'LBSTRESN': float(value),
                'LBSTRESU': unit
            }
            
            records.append(record)
        
        return records
    
    def _parse_generic_text(self, text: str, domain_code: str) -> List[Dict]:
        """Generic text parsing for any domain"""
        records = []
        
        # Split text into potential records
        lines = text.split('\n')
        current_record = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new record
            if any(keyword in line.lower() for keyword in ['subject', 'animal', 'record']):
                if current_record:
                    records.append(current_record)
                current_record = {
                    'STUDYID': '',
                    'DOMAIN': domain_code
                }
            
            # Parse key-value pairs
            if current_record and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().upper().replace(' ', '')
                value = value.strip()
                
                # Map to SENDIG variables if possible
                sendig_var = self._map_to_sendig_variable(key, domain_code)
                if sendig_var:
                    current_record[sendig_var] = value
        
        if current_record:
            records.append(current_record)
        
        return records
    
    def _standardize_lab_code(self, test_name: str) -> str:
        """Standardize laboratory test codes"""
        # Common lab test mappings
        code_mapping = {
            'GLUCOSE': 'GLUC',
            'CHOLESTEROL': 'CHOL',
            'TRIGLYCERIDES': 'TRIG',
            'ALBUMIN': 'ALB',
            'ALT': 'ALT',
            'AST': 'AST',
            'CREATININE': 'CREAT',
            'BUN': 'BUN',
            'HEMOGLOBIN': 'HGB',
            'HEMATOCRIT': 'HCT',
            'WBC': 'WBC',
            'RBC': 'RBC',
            'PLATELET': 'PLT'
        }
        
        test_upper = test_name.upper()
        for full_name, code in code_mapping.items():
            if full_name in test_upper:
                return code
        
        # Default to first 8 characters
        return test_upper[:8]
    
    def _map_to_sendig_variable(self, key: str, domain_code: str) -> Optional[str]:
        """Map text key to SENDIG variable name"""
        domain_metadata = self.xpt_generator.sendig_metadata.get(domain_code, {})
        
        # Direct match
        if key in domain_metadata:
            return key
        
        # Partial matches
        for var in domain_metadata.keys():
            if key in var or var in key:
                return var
        
        return None
    
    def _is_duplicate_record(self, record: Dict, existing_records: List[Dict]) -> bool:
        """Check if a record is a duplicate of existing records"""
        # Compare based on key fields
        key_fields = ['USUBJID', 'SUBJID', 'VISIT', 'TESTCD', 'DTC']
        
        for existing in existing_records:
            matches = 0
            common_fields = 0
            
            for field in key_fields:
                if field.endswith('TESTCD'):
                    # Check all possible test code fields
                    test_fields = [f for f in record.keys() if f.endswith('TESTCD')]
                    for tf in test_fields:
                        if tf in record and tf in existing:
                            common_fields += 1
                            if record[tf] == existing[tf]:
                                matches += 1
                else:
                    if field in record and field in existing:
                        common_fields += 1
                        if record[field] == existing[field]:
                            matches += 1
            
            # Consider duplicate if most key fields match
            if common_fields > 0 and matches >= common_fields * 0.8:
                return True
        
        return False
    
    def extract_all_selected_domains(self, pdf_document) -> Dict[str, ExtractedData]:
        """
        Extract data for all selected domains in a PDF
        
        Args:
            pdf_document: PDFDocument instance
            
        Returns:
            Dictionary mapping domain codes to ExtractedData instances
        """
        try:
            # Get all selected domains
            selected_domains = DetectedDomain.objects.filter(
                pdf=pdf_document,
                selected=True
            )
            
            # If no selected domains, complete immediately
            if not selected_domains:
                pdf_document.status = 'COMPLETED'
                pdf_document.save()
                return {}
            
            results = {}
            
            for detected_domain in selected_domains:
                try:
                    logger.info(f"Processing domain {detected_domain.domain.code}")
                    extracted_data = self.extract_data_for_domain(pdf_document, detected_domain)
                    results[detected_domain.domain.code] = extracted_data
                    
                    # Save progress after each domain
                    pdf_document.save()
                    
                except Exception as e:
                    logger.error(f"Error processing domain {detected_domain.domain.code}: {str(e)}")
                    # Continue with other domains even if one fails
                    continue
            
            # Update PDF status to completed
            pdf_document.status = 'COMPLETED'
            pdf_document.save()
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting data for PDF {pdf_document.id}: {str(e)}")
            pdf_document.status = 'FAILED'
            pdf_document.save()
            raise
    
    def create_submission_package(self, pdf_document: PDFDocument, output_dir: str) -> List[str]:
        """
        Create a complete submission package for all extracted domains
        
        Args:
            pdf_document: PDFDocument instance
            output_dir: Directory to write submission package
            
        Returns:
            List of created file paths
        """
        try:
            # Get all extracted data
            extracted_data_list = ExtractedData.objects.filter(pdf=pdf_document)
            
            if not extracted_data_list:
                raise Exception("No extracted data found for this document")
            
            # Prepare XPT files
            xpt_files = {}
            for extracted_data in extracted_data_list:
                # Read existing XPT file
                if extracted_data.xpt_file:
                    with open(extracted_data.xpt_file.path, 'rb') as f:
                        xpt_files[extracted_data.domain.code] = f.read()
            
            # Generate Define-XML
            define_xml = self._generate_define_xml(extracted_data_list)
            
            # Create submission package
            created_files = self.xpt_generator.create_submission_package(
                xpt_files,
                output_dir,
                define_xml
            )
            
            # Add study report if needed
            if pdf_document.file:
                report_path = os.path.join(output_dir, f"study_report_{pdf_document.id}.pdf")
                with open(pdf_document.file.path, 'rb') as src, open(report_path, 'wb') as dst:
                    dst.write(src.read())
                created_files.append(report_path)
            
            return created_files
            
        except Exception as e:
            logger.error(f"Error creating submission package: {str(e)}")
            raise
    
    def _generate_define_xml(self, extracted_data_list: List[ExtractedData]) -> str:
        """Generate Define-XML for all domains"""
        from xml.etree import ElementTree as ET
        
        # Create root element
        root = ET.Element('ODM')
        root.set('xmlns', 'http://www.cdisc.org/ns/odm/v1.3')
        root.set('ODMVersion', '1.3.2')
        root.set('FileOID', f'define_{datetime.now().strftime("%Y%m%d")}')
        root.set('FileType', 'Snapshot')
        root.set('CreationDateTime', datetime.now().isoformat())
        
        # Study element
        study = ET.SubElement(root, 'Study')
        study.set('OID', 'STUDY001')
        
        # MetaDataVersion
        metadata = ET.SubElement(study, 'MetaDataVersion')
        metadata.set('OID', 'MDV.001')
        metadata.set('Name', 'Study Metadata')
        
        # Add domain metadata
        for extracted_data in extracted_data_list:
            domain_code = extracted_data.domain.code
            df = pd.DataFrame(extracted_data.data.get('records', []))
            
            # Create ItemGroupDef
            item_group = ET.SubElement(metadata, 'ItemGroupDef')
            item_group.set('OID', f'IG.{domain_code}')
            item_group.set('Name', domain_code)
            item_group.set('Repeating', 'Yes')
            item_group.set('Purpose', 'Tabulation')
            
            # Add ItemDefs
            domain_metadata = self.xpt_generator.sendig_metadata.get(domain_code, {})
            for column in df.columns:
                if column in domain_metadata:
                    meta = domain_metadata[column]
                    item_def = ET.SubElement(item_group, 'ItemDef')
                    item_def.set('OID', f'{domain_code}.{column}')
                    item_def.set('Name', column)
                    item_def.set('DataType', 'text' if meta.type == 'Char' else 'float')
                    
                    if meta.length:
                        item_def.set('Length', str(meta.length))
                    
                    # Add description
                    desc = ET.SubElement(item_def, 'Description')
                    translated = ET.SubElement(desc, 'TranslatedText')
                    translated.set('xml:lang', 'en')
                    translated.text = meta.label
        
        # Convert to string
        return ET.tostring(root, encoding='unicode', method='xml')

# Usage example
if __name__ == "__main__":
    extractor = EnhancedDataExtractor()
    
    # Example PDF document
    pdf_doc = PDFDocument.objects.get(id=some_id)
    results = extractor.extract_all_selected_domains(pdf_doc)
    
    # Create submission package
    created_files = extractor.create_submission_package(pdf_doc, 'extractor\output')
    # pass