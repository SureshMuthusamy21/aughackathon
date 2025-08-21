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
            print(f"\n🔧 PREPARING RULE-BASED DATA")
            
            rule_based_columns = state.rule_based_columns
            file_path = state.file_path
            
            print(f"   📊 Columns to process: {len(rule_based_columns)}")
            for col, tool in rule_based_columns.items():
                print(f"      • {col} → {tool}")
            
            if not rule_based_columns:
                print(f"   ⚠️  No rule-based columns found, skipping...")
                return {
                    "processed_rule_data": [],
                    "processing_status": "no_rule_columns"
                }
            
            # Load the original data
            temp_file = f"temp_data_{hash(file_path)}.pkl"
            if os.path.exists(temp_file):
                original_df = pd.read_pickle(temp_file)
                print(f"   📁 Loaded data from temp file: {original_df.shape}")
            else:
                # Fallback: reload from original file
                original_df = pd.read_excel(file_path)
                print(f"   📁 Reloaded from original file: {original_df.shape}")
            
            # Extract only rule-based columns
            available_columns = [col for col in rule_based_columns.keys() if col in original_df.columns]
            missing_columns = [col for col in rule_based_columns.keys() if col not in original_df.columns]
            
            if missing_columns:
                print(f"   ⚠️  Missing columns: {missing_columns}")
            
            rule_data = original_df[available_columns].copy()
            print(f"   ✅ Extracted rule-based data: {rule_data.shape}")
            
            return {
                "processed_rule_data": rule_data.to_dict('records'),
                "processing_status": "rule_data_prepared"
            }
            
        except Exception as e:
            print(f"   ❌ Error preparing rule data: {str(e)}")
            return {
                "processing_status": "error",
                "error_log": state.error_log + [f"Rule data preparation error: {str(e)}"]
            }
    
    def apply_tools(self, state: MedicalDataState) -> dict:
        """Apply processing tools to respective columns"""
        try:
            print(f"\n🛠️  APPLYING RULE-BASED TOOLS")
            
            rule_based_columns = state.rule_based_columns
            processed_data_records = state.processed_rule_data
            errors = []
            
            if not processed_data_records:
                print(f"   ⚠️  No data to process")
                return {
                    "processed_rule_data": [],
                    "processing_status": "no_data_to_process"
                }
            
            # Convert back to DataFrame for processing
            processed_data = pd.DataFrame(processed_data_records)
            print(f"   📊 Processing {processed_data.shape[0]} rows, {processed_data.shape[1]} columns")
            
            for column_name, tool_name in rule_based_columns.items():
                print(f"\n   🔧 Processing column: {column_name} with tool: {tool_name}")
                
                if column_name not in processed_data.columns:
                    error_msg = f"Column {column_name} not found"
                    print(f"      ❌ {error_msg}")
                    errors.append(error_msg)
                    continue
                
                # Get the tool function
                tool_func = self.tools.get_tool_function(tool_name)
                if not tool_func:
                    error_msg = f"Tool {tool_name} not found"
                    print(f"      ❌ {error_msg}")
                    errors.append(error_msg)
                    continue
                
                # Show some before/after examples
                print(f"      📋 Sample values before processing:")
                sample_values = processed_data[column_name].head(3).tolist()
                for i, val in enumerate(sample_values, 1):
                    print(f"         {i}. {repr(val)}")
                
                # Apply tool to the column
                try:
                    processed_data[column_name] = processed_data[column_name].apply(tool_func)
                    print(f"      ✅ Tool applied successfully")
                    
                    # Show some after values
                    print(f"      📋 Sample values after processing:")
                    sample_values_after = processed_data[column_name].head(3).tolist()
                    for i, val in enumerate(sample_values_after, 1):
                        print(f"         {i}. {repr(val)}")
                        
                except Exception as e:
                    error_msg = f"Error applying {tool_name} to {column_name}: {str(e)}"
                    print(f"      ❌ {error_msg}")
                    errors.append(error_msg)
            
            if errors:
                print(f"\n   ⚠️  Encountered {len(errors)} errors during processing:")
                for error in errors:
                    print(f"      • {error}")
            else:
                print(f"\n   ✅ All tools applied successfully!")
            
            return {
                "processed_rule_data": processed_data.to_dict('records'),
                "error_log": state.error_log + errors,
                "processing_status": "tools_applied"
            }
            
        except Exception as e:
            print(f"   ❌ Error applying tools: {str(e)}")
            return {
                "processing_status": "error",
                "error_log": state.error_log + [f"Tool application error: {str(e)}"]
            }
    
    def finalize_rule_processing(self, state: MedicalDataState) -> dict:
        """Finalize rule-based processing"""
        print(f"\n   ✅ RULE-BASED PROCESSING COMPLETED")
        return {
            "processing_status": "rule_processing_complete"
        }

# Create the subgraph instance
rule_based_subgraph = RuleBasedSubgraph().graph
