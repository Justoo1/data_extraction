import logging
import io
import os
import pandas as pd
import numpy as np
import pyreadstat
import tempfile
from datetime import datetime

logger = logging.getLogger(__name__)

class XPTGenerator:
    """
    Service to generate SAS XPT files (Transport format version 5) from structured data
    according to SENDIG standards for FDA submission
    """
    
    def __init__(self):
        """Initialize the XPT generator"""
        pass
    
    def generate_xpt(self, domain_code, extracted_data):
        """
        Generate a SAS XPT file from structured data
        
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
            
            # Validate and prepare data according to SENDIG standards
            df = self._prepare_data_for_domain(domain_code, df)
            
            # Create XPT file in memory
            with tempfile.NamedTemporaryFile(suffix='.xpt', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Create catalog entry (column metadata)
            metadata = self._get_domain_metadata(domain_code)
            
            # Write DataFrame to XPT file
            pyreadstat.write_xport(
                df,
                temp_path,
                file_format_version=5,  # SAS Transport version 5 (required by FDA)
                column_labels=metadata.get('column_labels', {}),
                variable_value_labels=metadata.get('value_labels', {}),
                variable_attributes=metadata.get('attributes', {})
            )
            
            # Read the binary content
            with open(temp_path, 'rb') as f:
                xpt_content = f.read()
                
            # Clean up temporary file
            os.unlink(temp_path)
            
            return xpt_content
            
        except Exception as e:
            logger.error(f"Error generating XPT file for domain {domain_code}: {str(e)}")
            raise
    
    def _prepare_data_for_domain(self, domain_code, df):
        """
        Prepare data for a specific SENDIG domain
        
        Args:
            domain_code: SENDIG domain code
            df: pandas DataFrame with extracted data
            
        Returns:
            Prepared pandas DataFrame
        """
        # Get domain-specific preparation method
        method_name = f"_prepare_{domain_code.lower()}_domain"
        if hasattr(self, method_name):
            prepare_method = getattr(self, method_name)
            return prepare_method(df)
        else:
            # Use default preparation if domain-specific method not found
            return self._prepare_default_domain(df)
    
    def _prepare_default_domain(self, df):
        """
        Default data preparation for any SENDIG domain
        
        Args:
            df: pandas DataFrame with extracted data
            
        Returns:
            Prepared pandas DataFrame
        """
        # Add standard columns if not present
        required_cols = [
            'STUDYID', 'DOMAIN', 'USUBJID', 'SUBJID', 'SPREFID', 'SEQ'
        ]
        
        for col in required_cols:
            if col not in df.columns:
                # Create placeholder values if missing
                if col == 'SEQ':
                    df[col] = range(1, len(df) + 1)
                else:
                    df[col] = ''
        
        # Convert date columns to SAS format (if any)
        date_cols = [col for col in df.columns if 'DT' in col or 'DATE' in col]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # Convert to SAS date format (days since 1960-01-01)
                df[col] = (df[col] - pd.Timestamp('1960-01-01')).dt.days
        
        # Ensure numeric columns are properly formatted
        numeric_cols = [col for col in df.columns if any(x in col for x in ['--STRESN', '--ORRES', '--VAL'])]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _prepare_dm_domain(self, df):
        """
        Prepare Demographics (DM) domain data
        
        Args:
            df: pandas DataFrame with DM data
            
        Returns:
            Prepared pandas DataFrame
        """
        # DM-specific preparation
        if 'SEX' in df.columns:
            # Standardize sex values
            sex_map = {
                'male': 'M',
                'female': 'F',
                'm': 'M',
                'f': 'F'
            }
            df['SEX'] = df['SEX'].str.lower().map(lambda x: sex_map.get(x, x))
        
        # Add required columns for DM domain
        if 'RFSTDTC' not in df.columns:
            df['RFSTDTC'] = ''
        
        # Convert to SAS date format if dates exist
        if 'RFSTDTC' in df.columns and df['RFSTDTC'].any():
            df['RFSTDTC'] = pd.to_datetime(df['RFSTDTC'], errors='coerce')
            df['RFSTDTC'] = (df['RFSTDTC'] - pd.Timestamp('1960-01-01')).dt.days
        
        # Apply default preparation
        return self._prepare_default_domain(df)
    
    def _prepare_lb_domain(self, df):
        """
        Prepare Laboratory Test Results (LB) domain data
        
        Args:
            df: pandas DataFrame with LB data
            
        Returns:
            Prepared pandas DataFrame
        """
        # LB-specific preparation
        required_cols = [
            'LBTEST', 'LBSPEC', 'LBORRES', 'LBORRESU', 'LBSTRESC', 'LBSTRESN', 'LBSTRESU',
            'LBSTAT', 'LBREASND', 'LBNAM', 'LBDTC', 'LBDY'
        ]
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = ''
        
        # Convert numeric values
        if 'LBSTRESN' in df.columns:
            df['LBSTRESN'] = pd.to_numeric(df['LBSTRESN'], errors='coerce')
        
        # Apply default preparation
        return self._prepare_default_domain(df)

    def _get_domain_metadata(self, domain_code):
        """
        Get metadata for a specific SENDIG domain
        
        Args:
            domain_code: SENDIG domain code
            
        Returns:
            Dictionary with column metadata
        """
        # Get domain-specific metadata method
        method_name = f"_get_{domain_code.lower()}_metadata"
        if hasattr(self, method_name):
            get_metadata = getattr(self, method_name)
            return get_metadata()
        else:
            # Use default metadata if domain-specific method not found
            return self._get_default_metadata()
    
    def _get_default_metadata(self):
        """
        Default metadata for any SENDIG domain
        
        Returns:
            Dictionary with column metadata
        """
        # Default column labels
        column_labels = {
            'STUDYID': 'Study Identifier',
            'DOMAIN': 'Domain Abbreviation',
            'USUBJID': 'Unique Subject Identifier',
            'SUBJID': 'Subject Identifier',
            'SPREFID': 'Sponsor Reference ID',
            'SEQ': 'Sequence Number'
        }
        
        # Default value labels
        value_labels = {}
        
        # Default attributes
        attributes = {
            'STUDYID': {'SASTYPE': 'char', 'LENGTH': 20},
            'DOMAIN': {'SASTYPE': 'char', 'LENGTH': 2},
            'USUBJID': {'SASTYPE': 'char', 'LENGTH': 50},
            'SUBJID': {'SASTYPE': 'char', 'LENGTH': 20},
            'SPREFID': {'SASTYPE': 'char', 'LENGTH': 20},
            'SEQ': {'SASTYPE': 'num', 'FORMAT': 'best8.'}
        }
        
        return {
            'column_labels': column_labels,
            'value_labels': value_labels,
            'attributes': attributes
        }
    
    def _get_dm_metadata(self):
        """
        Get metadata for Demographics (DM) domain
        
        Returns:
            Dictionary with DM column metadata
        """
        # Get default metadata
        metadata = self._get_default_metadata()
        
        # Add DM-specific column labels
        dm_labels = {
            'RFSTDTC': 'Subject Reference Start Date/Time',
            'RFENDTC': 'Subject Reference End Date/Time',
            'RFXSTDTC': 'Date/Time of First Study Treatment',
            'RFXENDTC': 'Date/Time of Last Study Treatment',
            'RFICDTC': 'Date/Time of Informed Consent',
            'DTHFL': 'Death Flag',
            'DTHDTC': 'Date/Time of Death',
            'SEX': 'Sex',
            'AGE': 'Age',
            'AGEU': 'Age Units',
            'SPECIES': 'Species',
            'STRAIN': 'Strain/Substrain',
            'BIRTHDAY': 'Birth Date'
        }
        metadata['column_labels'].update(dm_labels)
        
        # Add DM-specific value labels
        sex_labels = {
            'M': 'Male',
            'F': 'Female',
            'U': 'Unknown'
        }
        metadata['value_labels']['SEX'] = sex_labels
        
        # Add DM-specific attributes
        dm_attributes = {
            'RFSTDTC': {'SASTYPE': 'num', 'FORMAT': 'date9.'},
            'RFENDTC': {'SASTYPE': 'num', 'FORMAT': 'date9.'},
            'SEX': {'SASTYPE': 'char', 'LENGTH': 1},
            'AGE': {'SASTYPE': 'num', 'FORMAT': 'best8.'},
            'AGEU': {'SASTYPE': 'char', 'LENGTH': 10},
            'SPECIES': {'SASTYPE': 'char', 'LENGTH': 20},
            'STRAIN': {'SASTYPE': 'char', 'LENGTH': 50}
        }
        metadata['attributes'].update(dm_attributes)
        
        return metadata
        
    def _get_lb_metadata(self):
        """
        Get metadata for Laboratory Test Results (LB) domain
        
        Returns:
            Dictionary with LB column metadata
        """
        # Get default metadata
        metadata = self._get_default_metadata()
        
        # Add LB-specific column labels
        lb_labels = {
            'LBTEST': 'Lab Test Name',
            'LBTESTCD': 'Lab Test Code',
            'LBCAT': 'Category for Lab Test',
            'LBORRES': 'Result or Finding in Original Units',
            'LBORRESU': 'Original Units',
            'LBORNRLO': 'Reference Range Lower Limit in Orig Unit',
            'LBORNRHI': 'Reference Range Upper Limit in Orig Unit',
            'LBSTRESC': 'Character Result/Finding in Std Format',
            'LBSTRESN': 'Numeric Result/Finding in Standard Units',
            'LBSTRESU': 'Standard Units',
            'LBSTNRLO': 'Reference Range Lower Limit-Std Units',
            'LBSTNRHI': 'Reference Range Upper Limit-Std Units',
            'LBNRIND': 'Reference Range Indicator',
            'LBSPEC': 'Specimen Type',
            'LBDTC': 'Date/Time of Specimen Collection',
            'LBBLFL': 'Baseline Flag'
        }
        metadata['column_labels'].update(lb_labels)
        
        # Add LB-specific attributes
        lb_attributes = {
            'LBTEST': {'SASTYPE': 'char', 'LENGTH': 40},
            'LBTESTCD': {'SASTYPE': 'char', 'LENGTH': 8},
            'LBCAT': {'SASTYPE': 'char', 'LENGTH': 20},
            'LBORRES': {'SASTYPE': 'char', 'LENGTH': 40},
            'LBORRESU': {'SASTYPE': 'char', 'LENGTH': 20},
            'LBSTRESN': {'SASTYPE': 'num', 'FORMAT': 'best8.'},
            'LBSTRESU': {'SASTYPE': 'char', 'LENGTH': 20},
            'LBSPEC': {'SASTYPE': 'char', 'LENGTH': 40},
            'LBDTC': {'SASTYPE': 'num', 'FORMAT': 'datetime19.'}
        }
        metadata['attributes'].update(lb_attributes)
        
        return metadata