import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
import pandas as pd
from datetime import datetime

try:
    from models.state import MedicalDataState
    from subgraphs.schema_subgraph import schema_analysis_subgraph
    from subgraphs.rule_based_subgraph import rule_based_subgraph
    from subgraphs.llm_subgraph import llm_subgraph
except ImportError:
    from ..models.state import MedicalDataState
    from ..subgraphs.schema_subgraph import schema_analysis_subgraph
    from ..subgraphs.rule_based_subgraph import rule_based_subgraph
    from ..subgraphs.llm_subgraph import llm_subgraph
    from ..subgraphs.rule_based_subgraph import rule_based_subgraph
    from ..subgraphs.llm_subgraph import llm_subgraph

class MedicalDataProcessor:
    
    def __init__(self):
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the main processing graph"""
        graph_builder = StateGraph(MedicalDataState)
        
        # Add nodes
        graph_builder.add_node("initialize_processing", self.initialize_processing)
        graph_builder.add_node("schema_analyzer", schema_analysis_subgraph)
        graph_builder.add_node("rule_based_processing", rule_based_subgraph)
        graph_builder.add_node("llm_processing", llm_subgraph)
        graph_builder.add_node("merge_results", self.merge_results)
        graph_builder.add_node("save_output", self.save_output)
        
        # Add edges
        graph_builder.add_edge(START, "initialize_processing")
        graph_builder.add_edge("initialize_processing", "schema_analyzer")
        
        # Add conditional edges for sequential processing
        graph_builder.add_conditional_edges(
            "schema_analyzer",
            self.route_to_processing,
            {
                "rule_based": "rule_based_processing", 
                "llm": "llm_processing", 
                "both": "rule_based_processing",
                "none": "merge_results"
            }
        )
        
        # For "both" case, go rule_based -> llm -> merge
        graph_builder.add_conditional_edges(
            "rule_based_processing",
            self.check_for_llm_processing,
            {
                "llm_needed": "llm_processing",
                "done": "merge_results"
            }
        )
        
        graph_builder.add_edge("llm_processing", "merge_results")
        graph_builder.add_edge("merge_results", "save_output")
        graph_builder.add_edge("save_output", END)
        
        # Compile with memory
        memory = MemorySaver()
        return graph_builder.compile(checkpointer=memory)
    
    def initialize_processing(self, state: MedicalDataState) -> dict:
        """Initialize the processing workflow"""
        print("\n🏗️  INITIALIZING PROCESSING WORKFLOW")
        print(f"   📁 File: {state.file_path}")
        return {
            "processing_status": "initialized"
        }
    
    def route_to_processing(self, state: MedicalDataState) -> str:
        """Route to appropriate processing based on categorization"""
        has_rule_columns = bool(state.rule_based_columns)
        has_llm_columns = bool(state.llm_columns)
        
        print(f"\n🔀 ROUTING DECISION:")
        print(f"   📊 Rule-based columns: {len(state.rule_based_columns)} found")
        print(f"   🤖 LLM columns: {len(state.llm_columns)} found")
        
        if has_rule_columns and has_llm_columns:
            print(f"   🎯 Route: BOTH (rule-based first, then LLM)")
            return "both"
        elif has_rule_columns:
            print(f"   🎯 Route: RULE-BASED ONLY")
            return "rule_based"
        elif has_llm_columns:
            print(f"   🎯 Route: LLM ONLY")
            return "llm"
        else:
            print(f"   🎯 Route: NO PROCESSING NEEDED")
            return "none"  # Skip processing if no columns to process
    
    def check_for_llm_processing(self, state: MedicalDataState) -> str:
        """Check if LLM processing is needed after rule-based processing"""
        has_llm_columns = bool(state.llm_columns)
        if has_llm_columns:
            print(f"   ➡️  Proceeding to LLM processing for {len(state.llm_columns)} columns")
            return "llm_needed"
        else:
            print(f"   ✅ Rule-based processing complete, proceeding to merge")
            return "done"
    
    def merge_results(self, state: MedicalDataState) -> dict:
        """Merge results from rule-based and LLM processing"""
        try:
            print(f"\n🔄 MERGING PROCESSING RESULTS")
            
            file_path = state.file_path
            processed_rule_data = state.processed_rule_data
            processed_llm_data = state.processed_llm_data
            
            # Load the original data
            temp_file = f"temp_data_{hash(file_path)}.pkl"
            if os.path.exists(temp_file):
                final_df = pd.read_pickle(temp_file)
                print(f"   📊 Loaded original data: {final_df.shape[0]} rows, {final_df.shape[1]} columns")
            else:
                # Fallback: reload from original file
                final_df = pd.read_excel(file_path)
                print(f"   📊 Reloaded from file: {final_df.shape[0]} rows, {final_df.shape[1]} columns")
            
            # Update with rule-based processed columns
            rule_columns_updated = 0
            if processed_rule_data and isinstance(processed_rule_data, list):
                rule_df = pd.DataFrame(processed_rule_data)
                for col in rule_df.columns:
                    if col in final_df.columns:
                        final_df[col] = rule_df[col]
                        rule_columns_updated += 1
                print(f"   🔧 Updated {rule_columns_updated} rule-based columns")
            else:
                print(f"   🔧 No rule-based columns to update")
            
            # Update with LLM processed columns
            llm_columns_updated = 0
            if processed_llm_data and isinstance(processed_llm_data, list):
                llm_df = pd.DataFrame(processed_llm_data)
                for col in llm_df.columns:
                    if col in final_df.columns:
                        final_df[col] = llm_df[col]
                        llm_columns_updated += 1
                print(f"   🤖 Updated {llm_columns_updated} LLM-processed columns")
            else:
                print(f"   🤖 No LLM columns to update")
            
            # Save the merged result temporarily
            merged_temp_file = f"temp_merged_{hash(file_path)}.pkl"
            final_df.to_pickle(merged_temp_file)
            print(f"   💾 Saved merged results to temporary file")
            
            return {
                "processing_status": "results_merged"
            }
            
        except Exception as e:
            print(f"   ❌ Error during merge: {str(e)}")
            return {
                "processing_status": "error",
                "error_log": state.error_log + [f"Merge error: {str(e)}"]
            }
    
    def save_output(self, state: MedicalDataState) -> dict:
        """Save the processed data to Excel file"""
        try:
            print(f"\n💾 SAVING OUTPUT FILE")
            
            file_path = state.file_path
            
            # Load the merged result
            merged_temp_file = f"temp_merged_{hash(file_path)}.pkl"
            if os.path.exists(merged_temp_file):
                final_df = pd.read_pickle(merged_temp_file)
                print(f"   📊 Final data shape: {final_df.shape[0]} rows, {final_df.shape[1]} columns")
            else:
                # Fallback: use original file
                final_df = pd.read_excel(file_path)
                print(f"   ⚠️  Using original file as fallback")
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{base_name}_processed_{timestamp}.xlsx"
            output_path = os.path.join(os.path.dirname(file_path), output_filename)
            
            print(f"   📄 Output file: {output_filename}")
            
            # Save to Excel
            final_df.to_excel(output_path, index=False)
            print(f"   ✅ Successfully saved to: {output_path}")
            
            # Clean up temporary files
            temp_file = f"temp_data_{hash(file_path)}.pkl"
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"   🗑️  Cleaned up temporary data file")
            if os.path.exists(merged_temp_file):
                os.remove(merged_temp_file)
                print(f"   🗑️  Cleaned up merged data file")
            
            return {
                "output_file_path": output_path,
                "processing_status": "completed"
            }
            
        except Exception as e:
            print(f"   ❌ Error saving file: {str(e)}")
            return {
                "processing_status": "error",
                "error_log": state.error_log + [f"Save error: {str(e)}"]
            }

# Create the main processor instance
medical_data_processor = MedicalDataProcessor().graph
