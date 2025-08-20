import re
import pandas as pd
from datetime import datetime
from typing import Union, Optional
import dateutil.parser as date_parser

class DataProcessingTools:
    
    @staticmethod
    def id_normalizer(value: Union[str, int, float]) -> Union[str, int, None]:
        """
        Standardize ID-like columns (patient_id, profile_id, biomarker_stamp, etc.)
        """
        try:
            if pd.isna(value) or value == "" or value is None:
                return None
                
            # Convert to string and strip spaces
            str_value = str(value).strip()
            
            if not str_value:
                return None
            
            # Remove all non-alphanumeric characters
            cleaned = re.sub(r'[^a-zA-Z0-9]', '', str_value)
            
            if not cleaned:
                return None
            
            # If it's all numeric, convert to integer
            if cleaned.isdigit():
                return int(cleaned)
            else:
                return cleaned
                
        except Exception:
            return value  # Return original on error
    
    @staticmethod
    def date_normalizer(value: Union[str, datetime]) -> Optional[str]:
        """
        Convert dates into standard format (YYYY-MM-DD HH:MM:SS)
        """
        try:
            if pd.isna(value) or value == "" or value is None:
                return None
            
            # If already datetime, format it
            if isinstance(value, datetime):
                return value.strftime("%Y-%m-%d %H:%M:%S")
            
            # Parse string date
            str_value = str(value).strip()
            if not str_value:
                return None
            
            # Use dateutil parser for flexible parsing
            parsed_date = date_parser.parse(str_value, fuzzy=True)
            return parsed_date.strftime("%Y-%m-%d %H:%M:%S")
            
        except Exception:
            return value  # Return original on error
    
    @staticmethod
    def flag_cleaner(value: Union[str, int, float]) -> Optional[str]:
        """
        Standardize flags into uppercase codes without special characters
        """
        try:
            if pd.isna(value) or value == "" or value is None:
                return None
            
            # Convert to string and strip
            str_value = str(value).strip().upper()
            
            if not str_value:
                return None
            
            # Remove all non-alphabet characters
            cleaned = re.sub(r'[^A-Z]', '', str_value)
            
            return cleaned if cleaned else None
            
        except Exception:
            return value  # Return original on error
    
    @staticmethod
    def text_cleaner(value: Union[str, int, float]) -> Optional[str]:
        """
        Clean free-text columns (e.g., lab_site)
        """
        try:
            if pd.isna(value) or value == "" or value is None:
                return None
            
            # Convert to string and strip
            str_value = str(value).strip()
            
            if not str_value:
                return None
            
            # Keep only alphanumeric and spaces
            cleaned = re.sub(r'[^a-zA-Z0-9 ]', '', str_value)
            
            # Replace multiple spaces with single space
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            return cleaned if cleaned else None
            
        except Exception:
            return value  # Return original on error
    
    @classmethod
    def get_tool_function(cls, tool_name: str):
        """Get the appropriate tool function by name"""
        tool_map = {
            "id_normalizer": cls.id_normalizer,
            "date_normalizer": cls.date_normalizer, 
            "flag_cleaner": cls.flag_cleaner,
            "text_cleaner": cls.text_cleaner
        }
        return tool_map.get(tool_name)
