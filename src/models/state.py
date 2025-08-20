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
    
    # Status and Output
    processing_status: str = "initialized"
    output_file_path: Optional[str] = None
    error_log: List[str] = []
    # ----------- End of Shared State ------------- #
    
    class Config:
        arbitrary_types_allowed = True
