import pytest
import pandas as pd
from src.tools.data_tools import DataProcessingTools

class TestDataProcessingTools:
    
    def test_id_normalizer(self):
        # Test cases for ID normalizer
        assert DataProcessingTools.id_normalizer(" P102-090 ") == "P102090"
        assert DataProcessingTools.id_normalizer(" 000123 ") == 123
        assert DataProcessingTools.id_normalizer(" @@ID!!45 ") == "ID45"
        assert DataProcessingTools.id_normalizer("") is None
        assert DataProcessingTools.id_normalizer(None) is None
    
    def test_date_normalizer(self):
        # Test cases for date normalizer
        result1 = DataProcessingTools.date_normalizer("12/08/25")
        assert "2025-08-12" in result1
        
        result2 = DataProcessingTools.date_normalizer("2025-Aug-12 3:45pm")
        assert "2025-08-12 15:45:00" == result2
        
        assert DataProcessingTools.date_normalizer("") is None
        assert DataProcessingTools.date_normalizer(None) is None
    
    def test_flag_cleaner(self):
        # Test cases for flag cleaner
        assert DataProcessingTools.flag_cleaner(" low ") == "LOW"
        assert DataProcessingTools.flag_cleaner(" H!gh ") == "HGH"
        assert DataProcessingTools.flag_cleaner("n/a") == "NA"
        assert DataProcessingTools.flag_cleaner("") is None
        assert DataProcessingTools.flag_cleaner(None) is None
    
    def test_text_cleaner(self):
        # Test cases for text cleaner
        assert DataProcessingTools.text_cleaner(" Lab@Site#12 ") == "LabSite12"
        assert DataProcessingTools.text_cleaner(" Main Hospital ") == "Main Hospital"
        assert DataProcessingTools.text_cleaner("***Central^Lab***") == "CentralLab"
        assert DataProcessingTools.text_cleaner("") is None
        assert DataProcessingTools.text_cleaner(None) is None
