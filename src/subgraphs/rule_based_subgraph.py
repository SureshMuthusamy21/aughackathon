import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd

try:
    from models.state import MedicalDataState
    from tools.data_tools import DataProcessingTools
except ImportError:
    from ..models.state import MedicalDataState
    from ..tools.data_tools import DataProcessingTools

class RuleBasedSubgraph:
    
    def __init__(self):
        self.tools = DataProcessingTools()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the rule-based processing subgraph"""
        graph_builder = StateGraph(MedicalDataState)
        
        # Add nodes
        graph_builder.add_node("prepare_rule_data", self.prepare_rule_data)
        graph_builder.add_node("apply_tools", self.apply_tools)
        graph_builder.add_node("finalize_rule_processing", self.finalize_rule_processing)
        
        # Add edges
        graph_builder.add_edge(START, "prepare_rule_data")
        graph_builder.add_edge("prepare_rule_data", "apply_tools")
        graph_builder.add_edge("apply_tools", "finalize_rule_processing")
        graph_builder.add_edge("finalize_rule_processing", END)
        
        # Compile with memory
        memory = MemorySaver()
        return graph_builder.compile(checkpointer=memory)
    
    def prepare_rule_data(self, state: MedicalDataState) -> dict:
        """Prepare data for rule-based processing"""
        try:
            rule_based_columns = state.rule_based_columns
            file_path = state.file_path
            
            if not rule_based_columns:
                return {
                    "processed_rule_data": [],
                    "processing_status": "no_rule_columns"
                }
            
            # Load the original data
            temp_file = f"temp_data_{hash(file_path)}.pkl"
            if os.path.exists(temp_file):
                original_df = pd.read_pickle(temp_file)
            else:
                # Fallback: reload from original file
                original_df = pd.read_excel(file_path)
            
            # Extract only rule-based columns
            available_columns = [col for col in rule_based_columns.keys() if col in original_df.columns]
            rule_data = original_df[available_columns].copy()
            
            return {
                "processed_rule_data": rule_data.to_dict('records'),
                "processing_status": "rule_data_prepared"
            }
            
        except Exception as e:
            return {
                "processing_status": "error",
                "error_log": state.error_log + [f"Rule data preparation error: {str(e)}"]
            }
    
    def apply_tools(self, state: MedicalDataState) -> dict:
        """Apply processing tools to respective columns"""
        try:
            rule_based_columns = state.rule_based_columns
            processed_data_records = state.processed_rule_data
            errors = []
            
            if not processed_data_records:
                return {
                    "processed_rule_data": [],
                    "processing_status": "no_data_to_process"
                }
            
            # Convert back to DataFrame for processing
            processed_data = pd.DataFrame(processed_data_records)
            
            for column_name, tool_name in rule_based_columns.items():
                if column_name not in processed_data.columns:
                    errors.append(f"Column {column_name} not found")
                    continue
                
                # Get the tool function
                tool_func = self.tools.get_tool_function(tool_name)
                if not tool_func:
                    errors.append(f"Tool {tool_name} not found")
                    continue
                
                # Apply tool to the column
                try:
                    processed_data[column_name] = processed_data[column_name].apply(tool_func)
                except Exception as e:
                    errors.append(f"Error applying {tool_name} to {column_name}: {str(e)}")
            
            return {
                "processed_rule_data": processed_data.to_dict('records'),
                "error_log": state.error_log + errors,
                "processing_status": "tools_applied"
            }
            
        except Exception as e:
            return {
                "processing_status": "error",
                "error_log": state.error_log + [f"Tool application error: {str(e)}"]
            }
    
    def finalize_rule_processing(self, state: MedicalDataState) -> dict:
        """Finalize rule-based processing"""
        return {
            "processing_status": "rule_processing_complete"
        }

# Create the subgraph instance
rule_based_subgraph = RuleBasedSubgraph().graph
