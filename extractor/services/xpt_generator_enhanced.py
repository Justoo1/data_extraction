import logging
import io
import os
import pandas as pd
import numpy as np
import pyreadstat
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class ValidationSeverity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"

@dataclass
class ValidationResult:
    """Results from validation checks"""
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    value: Optional[str] = None

@dataclass
class SENDIGMetadata:
    """Metadata for SENDIG domain variables"""
    variable: str
    label: str
    type: str  # 'Char' or 'Num'
    length: Optional[int] = None
    format: Optional[str] = None
    required: bool = False
    controlled_terms: Optional[List[str]] = None

class EnhancedXPTGenerator:
    """
    Enhanced XPT generator with SENDIG compliance validation
    and support for complex data structures
    """
    
    def __init__(self):
        self.sendig_metadata = self._load_sendig_metadata()
        self.domain_validators = self._initialize_domain_validators()
    
    def _load_sendig_metadata(self) -> Dict[str, Dict[str, SENDIGMetadata]]:
        """Load SENDIG metadata for each domain"""
        metadata = {
            'DM': {
                'STUDYID': SENDIGMetadata('STUDYID', 'Study Identifier', 'Char', 20, required=True),
                'DOMAIN': SENDIGMetadata('DOMAIN', 'Domain Abbreviation', 'Char', 2, required=True),
                'USUBJID': SENDIGMetadata('USUBJID', 'Unique Subject Identifier', 'Char', 50, required=True),
                'SUBJID': SENDIGMetadata('SUBJID', 'Subject Identifier', 'Char', 20, required=True),
                'RFSTDTC': SENDIGMetadata('RFSTDTC', 'Subject Reference Start Date/Time', 'Char', 19),
                'RFENDTC': SENDIGMetadata('RFENDTC', 'Subject Reference End Date/Time', 'Char', 19),
                'RFXSTDTC': SENDIGMetadata('RFXSTDTC', 'Date/Time of First Study Treatment', 'Char', 19),
                'RFXENDTC': SENDIGMetadata('RFXENDTC', 'Date/Time of Last Study Treatment', 'Char', 19),
                'RFICDTC': SENDIGMetadata('RFICDTC', 'Date/Time of Informed Consent', 'Char', 19),
                'DTHFL': SENDIGMetadata('DTHFL', 'Death Flag', 'Char', 1, controlled_terms=['Y', 'N']),
                'DTHDTC': SENDIGMetadata('DTHDTC', 'Date/Time of Death', 'Char', 19),
                'SEX': SENDIGMetadata('SEX', 'Sex', 'Char', 1, required=True, controlled_terms=['M', 'F', 'U']),
                'AGE': SENDIGMetadata('AGE', 'Age', 'Num', format='8.'),
                'AGEU': SENDIGMetadata('AGEU', 'Age Units', 'Char', 10),
                'SPECIES': SENDIGMetadata('SPECIES', 'Species', 'Char', 20, required=True),
                'STRAIN': SENDIGMetadata('STRAIN', 'Strain/Substrain', 'Char', 50),
                'BIRTHDAT': SENDIGMetadata('BIRTHDAT', 'Birth Date', 'Char', 19),
                'ARMCD': SENDIGMetadata('ARMCD', 'Planned Arm Code', 'Char', 8),
                'ARM': SENDIGMetadata('ARM', 'Description of Planned Arm', 'Char', 200),
                'SETCD': SENDIGMetadata('SETCD', 'Set Code', 'Char', 8),
                'ACTARMCD': SENDIGMetadata('ACTARMCD', 'Actual Arm Code', 'Char', 8),
                'ACTARM': SENDIGMetadata('ACTARM', 'Description of Actual Arm', 'Char', 200),
            },
            'BW': {
                'STUDYID': SENDIGMetadata('STUDYID', 'Study Identifier', 'Char', 20, required=True),
                'DOMAIN': SENDIGMetadata('DOMAIN', 'Domain Abbreviation', 'Char', 2, required=True),
                'USUBJID': SENDIGMetadata('USUBJID', 'Unique Subject Identifier', 'Char', 50, required=True),
                'BWSEQ': SENDIGMetadata('BWSEQ', 'Sequence Number', 'Num', format='8.'),
                'BWTESTCD': SENDIGMetadata('BWTESTCD', 'Body Weight Test Code', 'Char', 8, required=True),
                'BWTEST': SENDIGMetadata('BWTEST', 'Body Weight Test Name', 'Char', 40, required=True),
                'BWORRES': SENDIGMetadata('BWORRES', 'Result or Finding in Original Units', 'Char', 40),
                'BWORRESU': SENDIGMetadata('BWORRESU', 'Original Units', 'Char', 20),
                'BWSTRESC': SENDIGMetadata('BWSTRESC', 'Character Result/Finding in Std Format', 'Char', 40),
                'BWSTRESN': SENDIGMetadata('BWSTRESN', 'Numeric Result/Finding in Standard Units', 'Num', format='best8.'),
                'BWSTRESU': SENDIGMetadata('BWSTRESU', 'Standard Units', 'Char', 20),
                'BWSTAT': SENDIGMetadata('BWSTAT', 'Completion Status', 'Char', 20),
                'BWREASND': SENDIGMetadata('BWREASND', 'Reason Not Done', 'Char', 200),
                'BWNAM': SENDIGMetadata('BWNAM', 'Vendor Name', 'Char', 100),
                'BWMETHOD': SENDIGMetadata('BWMETHOD', 'Method of Test', 'Char', 100),
                'VISITNUM': SENDIGMetadata('VISITNUM', 'Visit Number', 'Num', format='8.'),
                'VISIT': SENDIGMetadata('VISIT', 'Visit Name', 'Char', 100),
                'BWDTC': SENDIGMetadata('BWDTC', 'Date/Time of Collection', 'Char', 19),
                'BWDY': SENDIGMetadata('BWDY', 'Study Day of Collection', 'Num', format='8.'),
            },
            'LB': {
                'STUDYID': SENDIGMetadata('STUDYID', 'Study Identifier', 'Char', 20, required=True),
                'DOMAIN': SENDIGMetadata('DOMAIN', 'Domain Abbreviation', 'Char', 2, required=True),
                'USUBJID': SENDIGMetadata('USUBJID', 'Unique Subject Identifier', 'Char', 50, required=True),
                'LBSEQ': SENDIGMetadata('LBSEQ', 'Sequence Number', 'Num', format='8.'),
                'LBGRPID': SENDIGMetadata('LBGRPID', 'Group ID', 'Char', 50),
                'LBREFID': SENDIGMetadata('LBREFID', 'Reference ID', 'Char', 50),
                'LBSPID': SENDIGMetadata('LBSPID', 'Sponsor-Defined Identifier', 'Char', 50),
                'LBTESTCD': SENDIGMetadata('LBTESTCD', 'Lab Test Code', 'Char', 8, required=True),
                'LBTEST': SENDIGMetadata('LBTEST', 'Lab Test Name', 'Char', 40, required=True),
                'LBCAT': SENDIGMetadata('LBCAT', 'Category for Lab Test', 'Char', 20),
                'LBSCAT': SENDIGMetadata('LBSCAT', 'Subcategory for Lab Test', 'Char', 20),
                'LBORRES': SENDIGMetadata('LBORRES', 'Result or Finding in Original Units', 'Char', 40),
                'LBORRESU': SENDIGMetadata('LBORRESU', 'Original Units', 'Char', 20),
                'LBORNRLO': SENDIGMetadata('LBORNRLO', 'Reference Range Lower Limit in Orig Unit', 'Char', 40),
                'LBORNRHI': SENDIGMetadata('LBORNRHI', 'Reference Range Upper Limit in Orig Unit', 'Char', 40),
                'LBSTRESC': SENDIGMetadata('LBSTRESC', 'Character Result/Finding in Std Format', 'Char', 40),
                'LBSTRESN': SENDIGMetadata('LBSTRESN', 'Numeric Result/Finding in Standard Units', 'Num', format='best8.'),
                'LBSTRESU': SENDIGMetadata('LBSTRESU', 'Standard Units', 'Char', 20),
                'LBSTNRLO': SENDIGMetadata('LBSTNRLO', 'Reference Range Lower Limit-Std Units', 'Num', format='best8.'),
                'LBSTNRHI': SENDIGMetadata('LBSTNRHI', 'Reference Range Upper Limit-Std Units', 'Num', format='best8.'),
                'LBNRIND': SENDIGMetadata('LBNRIND', 'Reference Range Indicator', 'Char', 20),
                'LBSTAT': SENDIGMetadata('LBSTAT', 'Completion Status', 'Char', 20),
                'LBREASND': SENDIGMetadata('LBREASND', 'Reason Not Done', 'Char', 200),
                'LBNAM': SENDIGMetadata('LBNAM', 'Vendor Name', 'Char', 100),
                'LBSPEC': SENDIGMetadata('LBSPEC', 'Specimen Type', 'Char', 40),
                'LBSPCCND': SENDIGMetadata('LBSPCCND', 'Specimen Condition', 'Char', 100),
                'LBMETHOD': SENDIGMetadata('LBMETHOD', 'Method of Test', 'Char', 100),
                'VISITNUM': SENDIGMetadata('VISITNUM', 'Visit Number', 'Num', format='8.'),
                'VISIT': SENDIGMetadata('VISIT', 'Visit Name', 'Char', 100),
                'LBDTC': SENDIGMetadata('LBDTC', 'Date/Time of Specimen Collection', 'Char', 19),
                'LBDY': SENDIGMetadata('LBDY', 'Study Day of Specimen Collection', 'Num', format='8.'),
                'LBTPT': SENDIGMetadata('LBTPT', 'Planned Time Point Name', 'Char', 200),
                'LBTPTNUM': SENDIGMetadata('LBTPTNUM', 'Planned Time Point Number', 'Num', format='best8.'),
                'LBELTM': SENDIGMetadata('LBELTM', 'Planned Elapsed Time from Time Point Ref', 'Char', 40),
                'LBTPTREF': SENDIGMetadata('LBTPTREF', 'Time Point Reference', 'Char', 200),
                'LBRFTDTC': SENDIGMetadata('LBRFTDTC', 'Date/Time of Reference Time Point', 'Char', 19),
                'LBFEREF': SENDIGMetadata('LBFEREF', 'Reference Period for Feature', 'Char', 100),
                'LBFASTYN': SENDIGMetadata('LBFASTYN', 'Fasting Status', 'Char', 1, controlled_terms=['Y', 'N', 'U']),
                'LBBLFL': SENDIGMetadata('LBBLFL', 'Baseline Flag', 'Char', 1, controlled_terms=['Y']),
            }
        }
        
        # Add more domain metadata as needed
        return metadata
    
    def _initialize_domain_validators(self) -> Dict[str, callable]:
        """Initialize domain-specific validation functions"""
        return {
            'DM': self._validate_dm_domain,
            'BW': self._validate_bw_domain,
            'LB': self._validate_lb_domain,
        }
    
    def generate_xpt(self, domain_code: str, extracted_data: Dict) -> bytes:
        """
        Generate a SAS XPT file from structured data with SENDIG validation
        
        Args:
            domain_code: SENDIG domain code
            extracted_data: Dictionary containing structured data
            
        Returns:
            Binary content of the XPT file
        """
        try:
            # Convert extracted data to DataFrame
            records = extracted_data.get('records', [])
            if not records:
                raise ValueError(f"No records found for domain {domain_code}")
            
            df = pd.DataFrame(records)
            
            # Validate the data
            validation_results = self.validate_domain_data(domain_code, df)
            
            # Check for critical errors
            errors = [v for v in validation_results if v.severity == ValidationSeverity.ERROR]
            if errors:
                error_messages = [f"{v.field}: {v.message}" for v in errors]
                raise ValidationError(f"Validation errors in {domain_code}:\n" + "\n".join(error_messages))
            
            # Prepare and clean data according to SENDIG standards
            df = self._prepare_data_for_domain(domain_code, df)
            
            # Create metadata
            metadata = self._create_xpt_metadata(domain_code, df)
            
            # Create XPT file in memory
            with tempfile.NamedTemporaryFile(suffix='.xpt', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Write DataFrame to XPT file
                pyreadstat.write_xport(
                    df,
                    temp_path,
                    file_format_version=5,  # SAS Transport version 5
                    column_labels=metadata['column_labels'],
                    variable_value_labels=metadata.get('value_labels', {}),
                    variable_formats=metadata.get('formats', {}),
                    variable_types_as_int=metadata.get('types_as_int', {})
                )
                
                # Read the binary content
                with open(temp_path, 'rb') as f:
                    xpt_content = f.read()
                
                return xpt_content
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Error generating XPT file for domain {domain_code}: {str(e)}")
            raise
    
    def _prepare_data_for_domain(self, domain_code: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for a specific SENDIG domain
        
        Args:
            domain_code: SENDIG domain code
            df: pandas DataFrame with extracted data
            
        Returns:
            Prepared pandas DataFrame
        """
        # Get domain metadata
        domain_metadata = self.sendig_metadata.get(domain_code, {})
        
        # Ensure required columns exist
        for var_name, metadata in domain_metadata.items():
            if metadata.required and var_name not in df.columns:
                # Create placeholder with appropriate default
                if metadata.type == 'Char':
                    df[var_name] = ''
                else:
                    df[var_name] = np.nan
        
        # Standardize data types
        for var_name, metadata in domain_metadata.items():
            if var_name in df.columns:
                df[var_name] = self._standardize_variable(df[var_name], metadata)
        
        # Add sequence numbers if missing
        if 'SEQ' in domain_metadata and 'SEQ' not in df.columns:
            df['SEQ'] = range(1, len(df) + 1)
        
        # Sort columns by SENDIG order
        ordered_columns = list(domain_metadata.keys())
        extra_columns = [col for col in df.columns if col not in ordered_columns]
        final_columns = ordered_columns + extra_columns
        df = df[[col for col in final_columns if col in df.columns]]
        
        return df
    
    def _standardize_variable(self, series: pd.Series, metadata: SENDIGMetadata) -> pd.Series:
        """Standardize a variable according to SENDIG requirements"""
        
        if metadata.type == 'Char':
            # Convert to string and trim
            series = series.astype(str).str.strip()
            
            # Replace NaN representations
            series = series.replace(['nan', 'NaN', ''], np.nan)
            
            # Apply controlled terms if specified
            if metadata.controlled_terms:
                series = series.apply(lambda x: self._standardize_controlled_term(x, metadata.controlled_terms))
            
            # Truncate to maximum length if specified
            if metadata.length:
                series = series.str[:metadata.length]
        
        elif metadata.type == 'Num':
            # Convert to numeric
            series = pd.to_numeric(series, errors='coerce')
            
            # Apply format if specified
            if metadata.format:
                if metadata.format == 'best8.':
                    # Round to appropriate precision
                    series = series.round(8)
                elif 'date' in metadata.format.lower():
                    # Handle date formats
                    series = self._convert_to_sas_date(series)
        
        return series
    
    def _standardize_controlled_term(self, value: str, controlled_terms: List[str]) -> str:
        """Standardize a value to match controlled terms"""
        if pd.isna(value) or value == '':
            return ''
        
        # Exact match
        if value in controlled_terms:
            return value
        
        # Case-insensitive match
        for term in controlled_terms:
            if value.upper() == term.upper():
                return term
        
        # Partial match or mapping
        value_upper = value.upper()
        if 'SEX' in str(controlled_terms):
            if value_upper.startswith('M') or 'MALE' in value_upper:
                return 'M'
            elif value_upper.startswith('F') or 'FEMALE' in value_upper:
                return 'F'
            else:
                return 'U'
        
        # Default to original if no match
        return value
    
    def _convert_to_sas_date(self, series: pd.Series) -> pd.Series:
        """Convert to SAS date format (days since 1960-01-01)"""
        try:
            # Try to parse as datetime
            dates = pd.to_datetime(series, errors='coerce')
            # Convert to SAS date format
            sas_dates = (dates - pd.Timestamp('1960-01-01')).dt.days
            return sas_dates
        except:
            return series
    
    def _create_xpt_metadata(self, domain_code: str, df: pd.DataFrame) -> Dict:
        """Create metadata for XPT file"""
        domain_metadata = self.sendig_metadata.get(domain_code, {})
        
        metadata = {
            'column_labels': {},
            'value_labels': {},
            'formats': {},
            'types_as_int': {}
        }
        
        for column in df.columns:
            if column in domain_metadata:
                meta = domain_metadata[column]
                metadata['column_labels'][column] = meta.label
                
                if meta.format:
                    metadata['formats'][column] = meta.format
                
                # Set SAS variable type (0 for numeric, 1 for character)
                metadata['types_as_int'][column] = 1 if meta.type == 'Char' else 0
                
                # Add value labels for controlled terms
                if meta.controlled_terms:
                    metadata['value_labels'][column] = {
                        term: term for term in meta.controlled_terms
                    }
        
        return metadata
    
    def validate_domain_data(self, domain_code: str, df: pd.DataFrame) -> List[ValidationResult]:
        """
        Validate domain data according to SENDIG standards
        
        Args:
            domain_code: SENDIG domain code
            df: pandas DataFrame with domain data
            
        Returns:
            List of validation results
        """
        results = []
        domain_metadata = self.sendig_metadata.get(domain_code, {})
        
        # Check required variables
        for var_name, metadata in domain_metadata.items():
            if metadata.required:
                if var_name not in df.columns:
                    results.append(ValidationResult(
                        ValidationSeverity.ERROR,
                        f"Required variable {var_name} is missing",
                        field=var_name
                    ))
                elif df[var_name].isna().all():
                    results.append(ValidationResult(
                        ValidationSeverity.ERROR,
                        f"Required variable {var_name} has no values",
                        field=var_name
                    ))
        
        # Validate data types and formats
        for column in df.columns:
            if column in domain_metadata:
                metadata = domain_metadata[column]
                series = df[column]
                
                # Check data type
                if metadata.type == 'Num':
                    non_numeric = pd.to_numeric(series, errors='coerce').isna()
                    if non_numeric.any() and not series.isna().all():
                        results.append(ValidationResult(
                            ValidationSeverity.WARNING,
                            f"Variable {column} contains non-numeric values",
                            field=column,
                            value=str(series[non_numeric].iloc[0]) if non_numeric.any() else None
                        ))
                
                # Check controlled terms
                if metadata.controlled_terms:
                    invalid_values = ~series.isin(metadata.controlled_terms + ['', np.nan])
                    if invalid_values.any():
                        for idx, val in series[invalid_values].items():
                            results.append(ValidationResult(
                                ValidationSeverity.WARNING,
                                f"Variable {column} has invalid controlled term: {val}",
                                field=column,
                                value=str(val)
                            ))
                
                # Check length limits
                if metadata.length and metadata.type == 'Char':
                    long_values = series.str.len() > metadata.length
                    if long_values.any():
                        results.append(ValidationResult(
                            ValidationSeverity.WARNING,
                            f"Variable {column} has values exceeding length limit of {metadata.length}",
                            field=column
                        ))
        
        # Domain-specific validation
        validator = self.domain_validators.get(domain_code)
        if validator:
            domain_results = validator(df)
            results.extend(domain_results)
        
        return results
    
    def _validate_dm_domain(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate Demographics (DM) domain specific requirements"""
        results = []
        
        # Check unique subject IDs
        if 'USUBJID' in df.columns:
            if not df['USUBJID'].is_unique:
                results.append(ValidationResult(
                    ValidationSeverity.ERROR,
                    "USUBJID values must be unique in DM domain",
                    field='USUBJID'
                ))
        
        # Check date consistency
        date_fields = ['RFSTDTC', 'RFENDTC', 'RFXSTDTC', 'RFXENDTC', 'DTHDTC']
        for date_field in date_fields:
            if date_field in df.columns:
                # Basic date format check
                invalid_dates = ~df[date_field].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}.*', na=False)
                if invalid_dates.any():
                    results.append(ValidationResult(
                        ValidationSeverity.WARNING,
                        f"Invalid date format in {date_field}",
                        field=date_field
                    ))
        
        # Check sex values
        if 'SEX' in df.columns:
            invalid_sex = ~df['SEX'].isin(['M', 'F', 'U', ''])
            if invalid_sex.any():
                results.append(ValidationResult(
                    ValidationSeverity.ERROR,
                    "SEX must be M, F, or U",
                    field='SEX'
                ))
        
        return results
    
    def _validate_bw_domain(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate Body Weight (BW) domain specific requirements"""
        results = []
        
        # Check for positive weights
        if 'BWSTRESN' in df.columns:
            negative_weights = df['BWSTRESN'] < 0
            if negative_weights.any():
                results.append(ValidationResult(
                    ValidationSeverity.ERROR,
                    "Body weights cannot be negative",
                    field='BWSTRESN'
                ))
        
        # Check completion status consistency
        if 'BWSTAT' in df.columns and 'BWSTRESN' in df.columns:
            # If status is 'NOT DONE', result should be null
            missing_but_has_value = (df['BWSTAT'] == 'NOT DONE') & df['BWSTRESN'].notna()
            if missing_but_has_value.any():
                results.append(ValidationResult(
                    ValidationSeverity.WARNING,
                    "Body weight marked as NOT DONE but has a value",
                    field='BWSTAT'
                ))
        
        return results
    
    def _validate_lb_domain(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate Laboratory Test Results (LB) domain specific requirements"""
        results = []
        
        # Check test code and name consistency
        if 'LBTESTCD' in df.columns and 'LBTEST' in df.columns:
            # Group by test code and check for consistent test names
            grouped = df.groupby('LBTESTCD')['LBTEST'].nunique()
            inconsistent = grouped[grouped > 1]
            if not inconsistent.empty:
                for test_code in inconsistent.index:
                    results.append(ValidationResult(
                        ValidationSeverity.WARNING,
                        f"Test code {test_code} has multiple test names",
                        field='LBTESTCD'
                    ))
        
        # Check numeric results
        if 'LBSTRESN' in df.columns:
            # Results should be numeric if standardized result is present
            non_numeric = pd.to_numeric(df['LBSTRESN'], errors='coerce').isna()
            has_value = df['LBSTRESN'].notna()
            if (non_numeric & has_value).any():
                results.append(ValidationResult(
                    ValidationSeverity.ERROR,
                    "LBSTRESN must be numeric",
                    field='LBSTRESN'
                ))
        
        # Check reference range indicators
        if 'LBNRIND' in df.columns:
            valid_indicators = ['NORMAL', 'LOW', 'HIGH', 'ABNORMAL', '']
            invalid_indicators = ~df['LBNRIND'].isin(valid_indicators)
            if invalid_indicators.any():
                results.append(ValidationResult(
                    ValidationSeverity.WARNING,
                    "Invalid reference range indicator values",
                    field='LBNRIND'
                ))
        
        return results
    
    def generate_define_xml(self, domain_code: str, df: pd.DataFrame) -> str:
        """
        Generate Define-XML metadata for the XPT file
        
        Args:
            domain_code: SENDIG domain code
            df: pandas DataFrame with domain data
            
        Returns:
            Define-XML content as string
        """
        domain_metadata = self.sendig_metadata.get(domain_code, {})
        
        # Basic Define-XML structure
        xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<ItemGroupData ItemGroupOID="{domain_code}">
  <ItemGroupDef OID="{domain_code}" Name="{domain_code}" Repeating="Yes" Purpose="Tabulation">
    <ItemDef OID="{domain_code}.STUDYID" Name="STUDYID" DataType="text" Length="20"/>
'''
        
        # Add item definitions for each variable
        for column in df.columns:
            if column in domain_metadata:
                meta = domain_metadata[column]
                data_type = "text" if meta.type == "Char" else "float"
                length_attr = f' Length="{meta.length}"' if meta.length else ''
                
                xml_content += f'''    <ItemDef OID="{domain_code}.{column}" Name="{column}" DataType="{data_type}"{length_attr}>
      <Description><TranslatedText xml:lang="en">{meta.label}</TranslatedText></Description>
    </ItemDef>
'''
        
        xml_content += '''  </ItemGroupDef>
</ItemGroupData>
'''
        
        return xml_content
    
    def create_submission_package(self, 
                                 xpt_files: Dict[str, bytes], 
                                 output_dir: str,
                                 define_xml: Optional[str] = None) -> List[str]:
        """
        Create a complete submission package with XPT files and metadata
        
        Args:
            xpt_files: Dictionary mapping domain codes to XPT file content
            output_dir: Directory to write files
            define_xml: Optional Define-XML content
            
        Returns:
            List of created file paths
        """
        created_files = []
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Write XPT files
        for domain_code, content in xpt_files.items():
            xpt_path = os.path.join(output_dir, f"{domain_code.lower()}.xpt")
            with open(xpt_path, 'wb') as f:
                f.write(content)
            created_files.append(xpt_path)
        
        # Write Define-XML if provided
        if define_xml:
            define_path = os.path.join(output_dir, "define.xml")
            with open(define_path, 'w', encoding='utf-8') as f:
                f.write(define_xml)
            created_files.append(define_path)
        
        # Create submission guide
        guide_path = os.path.join(output_dir, "submission_guide.txt")
        with open(guide_path, 'w') as f:
            f.write("SENDIG Submission Package\n")
            f.write("=" * 25 + "\n\n")
            f.write("Files included:\n")
            for domain_code in xpt_files.keys():
                f.write(f"- {domain_code.lower()}.xpt: {domain_code} domain data\n")
            if define_xml:
                f.write("- define.xml: Metadata definition\n")
            f.write("\nGenerated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        created_files.append(guide_path)
        
        return created_files

# Usage example
if __name__ == "__main__":
    generator = EnhancedXPTGenerator()
    
    # Example DM data
    dm_data = {
        'records': [
            {
                'STUDYID': 'STUDY001',
                'DOMAIN': 'DM',
                'USUBJID': 'STUDY001-001',
                'SUBJID': '001',
                'SEX': 'M',
                'SPECIES': 'MICE',
                'STRAIN': 'C57BL/6J',
                'AGE': 8,
                'AGEU': 'WEEKS'
            }
        ]
    }
    
    # Generate XPT file
    xpt_content = generator.generate_xpt('DM', dm_data)
    
    # Validate data
    df = pd.DataFrame(dm_data['records'])
    validation_results = generator.validate_domain_data('DM', df)
    
    # Print validation results
    for result in validation_results:
        print(f"{result.severity}: {result.message}")