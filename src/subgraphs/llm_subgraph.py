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
except ImportError:
    from ..models.state import MedicalDataState

class LLMSubgraph:
    
    def __init__(self):
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LLM processing subgraph (placeholder)"""
        graph_builder = StateGraph(MedicalDataState)
        
        # Add nodes
        graph_builder.add_node("prepare_llm_data", self.prepare_llm_data)
        graph_builder.add_node("process_llm_data", self.process_llm_data)
        graph_builder.add_node("finalize_llm_processing", self.finalize_llm_processing)
        
        # Add edges
        graph_builder.add_edge(START, "prepare_llm_data")
        graph_builder.add_edge("prepare_llm_data", "process_llm_data")
        graph_builder.add_edge("process_llm_data", "finalize_llm_processing")
        graph_builder.add_edge("finalize_llm_processing", END)
        
        # Compile with memory
        memory = MemorySaver()
        return graph_builder.compile(checkpointer=memory)
    
    def prepare_llm_data(self, state: MedicalDataState) -> dict:
        """Prepare data for LLM processing"""
        try:
            print(f"\n🤖 PREPARING LLM DATA")
            
            llm_columns = state.llm_columns
            file_path = state.file_path
            
            print(f"   📊 LLM columns to process: {len(llm_columns)}")
            for col in llm_columns:
                print(f"      • {col}")
            
            if not llm_columns:
                print(f"   ⚠️  No LLM columns found, skipping...")
                return {
                    "processed_llm_data": [],
                    "processing_status": "no_llm_columns"
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
            
            # Extract only LLM columns
            available_columns = [col for col in llm_columns if col in original_df.columns]
            missing_columns = [col for col in llm_columns if col not in original_df.columns]
            
            if missing_columns:
                print(f"   ⚠️  Missing LLM columns: {missing_columns}")
            
            llm_data = original_df[available_columns].copy()
            print(f"   ✅ Extracted LLM data: {llm_data.shape}")
            
            return {
                "processed_llm_data": llm_data.to_dict('records'),
                "processing_status": "llm_data_prepared"
            }
            
        except Exception as e:
            print(f"   ❌ Error preparing LLM data: {str(e)}")
            return {
                "processing_status": "error",
                "error_log": state.error_log + [f"LLM data preparation error: {str(e)}"]
            }
    
    def process_llm_data(self, state: MedicalDataState) -> dict:
        """Process LLM data (placeholder implementation)"""
        try:
            print(f"\n🧠 PROCESSING LLM DATA")
            
            processed_data_records = state.processed_llm_data
            
            if isinstance(processed_data_records, list) and processed_data_records:
                print(f"   📊 LLM data records: {len(processed_data_records)}")
                
                # Show sample of data to be processed
                print(f"   📋 Sample LLM data (first 3 records):")
                for i, record in enumerate(processed_data_records[:3]):
                    print(f"      Record {i+1}: {record}")
                
                print(f"   🤖 Applying LLM processing...")
                # PLACEHOLDER: For now, just return the data as-is
                # In future, implement complex medical data processing here
                
                # TODO: Add actual LLM processing logic here
                # - Medical term standardization
                # - Diagnostic code mapping
                # - Free-text analysis
                
                print(f"   ✅ LLM processing completed")
                print(f"   📈 Processed {len(processed_data_records)} records")
                
                return {
                    "processed_llm_data": processed_data_records,
                    "processing_status": "llm_processed"
                }
            else:
                print(f"   ⚠️  No LLM data records to process")
                return {
                    "processed_llm_data": [],
                    "processing_status": "llm_processed"
                }
            
        except Exception as e:
            print(f"   ❌ Error in LLM processing: {str(e)}")
            return {
                "processing_status": "error",
                "error_log": state.error_log + [f"LLM processing error: {str(e)}"]
            }
    
    def finalize_llm_processing(self, state: MedicalDataState) -> dict:
        """Finalize LLM processing"""
        print(f"\n✅ FINALIZING LLM PROCESSING")
        print(f"   🎯 LLM processing pipeline completed successfully")
        
        return {
            "processing_status": "llm_processing_complete"
        }

# Create the subgraph instance  
llm_subgraph = LLMSubgraph().graph
