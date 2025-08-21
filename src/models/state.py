from typing import List, Dict, Optional, Any, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import add_messages
import pandas as pd

# Pydantic Models for Schema Analysis
class ColumnAnalysis(BaseModel):
    column_name: str
    category: str  # "rule_based" or "llm_based"
    tool_name: Optional[str] = None  # Only for rule_based
    reasoning: str

class SchemaAnalysisResult(BaseModel):
    columns: List[ColumnAnalysis]

# Medical Data Review Models
class MedicalIssue(BaseModel):
    issue_type: str  # "spelling", "terminology", "missing", "duplicate", "format", "outlier", "other"
    description: str
    severity: str  # "low", "medium", "high", "critical"
    suggested_correction: Optional[str] = None

class RowReview(BaseModel):
    row_index: int
    issues: List[MedicalIssue]
    overall_status: str  # "clean", "needs_review", "needs_correction", "critical"

class BatchReview(BaseModel):
    batch_id: int
    reviews: List[RowReview]
    summary: str

class CorrectedRow(BaseModel):
    row_index: int
    corrected_data: Dict[str, Any]
    corrections_made: List[str]

class BatchCorrection(BaseModel):
    batch_id: int
    corrected_rows: List[CorrectedRow]
    correction_summary: str

# Main State Class - Using separate fields to avoid DataFrame serialization issues
class MedicalDataState(BaseModel):
    # ----------- Shared State -------------------- #
    messages: Annotated[list[HumanMessage | AIMessage | SystemMessage], add_messages] = []
    file_path: Optional[str] = None
    
    # Data Management - Store as serializable data
    headers: List[str] = []
    sample_data: List[Dict[str, Any]] = []
    data_shape: Optional[tuple] = None  # Store DataFrame shape instead of DataFrame
    
    # Schema Analysis Results
    schema_analysis: Optional[SchemaAnalysisResult] = None
    rule_based_columns: Dict[str, str] = {}  # {column_name: tool_name}
    llm_columns: List[str] = []
    
    # Processing Results - Store processed data as list of records
    processed_rule_data: Optional[List[Dict[str, Any]]] = None
    processed_llm_data: Optional[List[Dict[str, Any]]] = None
    
    # LLM Batch Processing Results
    batch_reviews: List[BatchReview] = []
    batch_corrections: List[BatchCorrection] = []
    llm_processing_stats: Dict[str, Any] = {}
    
    # Status and Output
    processing_status: str = "initialized"
    output_file_path: Optional[str] = None
    error_log: List[str] = []
    # ----------- End of Shared State ------------- #
    
    class Config:
        arbitrary_types_allowed = True
