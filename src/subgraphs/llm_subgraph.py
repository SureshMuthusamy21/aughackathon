import os
import sys
from pathlib import Path
import asyncio
from typing import List, Dict, Any
import time

# Add the src directory to the Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd

try:
    from models.state import MedicalDataState, BatchReview, BatchCorrection
    from subgraphs.medical_agents import MedicalDataReviewer, MedicalDataCorrector
except ImportError:
    from ..models.state import MedicalDataState, BatchReview, BatchCorrection
    from .medical_agents import MedicalDataReviewer, MedicalDataCorrector

# Configuration
BATCH_SIZE = 5
MAX_WORKERS = 10
SLEEP_BETWEEN_BATCHES = 1.0  # 1 second sleep to avoid rate limiting
MAX_RETRIES = 3

class LLMSubgraph:
    
    def __init__(self):
        self.reviewer = MedicalDataReviewer()
        self.corrector = MedicalDataCorrector()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LLM processing subgraph with async batch processing"""
        graph_builder = StateGraph(MedicalDataState)
        
        # Add nodes
        graph_builder.add_node("prepare_llm_data", self.prepare_llm_data)
        graph_builder.add_node("async_batch_processing", self.async_batch_processing)
        graph_builder.add_node("finalize_llm_processing", self.finalize_llm_processing)
        
        # Add edges
        graph_builder.add_edge(START, "prepare_llm_data")
        graph_builder.add_edge("prepare_llm_data", "async_batch_processing")
        graph_builder.add_edge("async_batch_processing", "finalize_llm_processing")
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
    
    def async_batch_processing(self, state: MedicalDataState) -> dict:
        """Process LLM data using async batch processing with medical agents"""
        try:
            print(f"\n🧠 ASYNC MEDICAL DATA PROCESSING")
            
            processed_data_records = state.processed_llm_data
            llm_columns = state.llm_columns
            
            if not processed_data_records or not llm_columns:
                print(f"   ⚠️  No LLM data to process")
                return {
                    "processed_llm_data": [],
                    "llm_processing_stats": {"batches_processed": 0, "total_records": 0},
                    "processing_status": "no_llm_data"
                }
            
            total_records = len(processed_data_records)
            print(f"   📊 Total records to process: {total_records}")
            print(f"   ⚙️  Configuration: {MAX_WORKERS} workers, {BATCH_SIZE} per batch")
            
            # Split data into batches
            batches = self._create_batches(processed_data_records, BATCH_SIZE)
            total_batches = len(batches)
            print(f"   � Created {total_batches} batches")
            
            # Process batches asynchronously
            corrected_data, batch_reviews, batch_corrections, processing_stats = asyncio.run(
                self._process_batches_async(batches, llm_columns)
            )
            
            print(f"   ✅ Async processing completed!")
            print(f"   📈 Processing stats: {processing_stats}")
            
            return {
                "processed_llm_data": corrected_data,
                "batch_reviews": batch_reviews,
                "batch_corrections": batch_corrections,
                "llm_processing_stats": processing_stats,
                "processing_status": "llm_processed"
            }
            
        except Exception as e:
            print(f"   ❌ Error in async batch processing: {str(e)}")
            return {
                "processing_status": "error",
                "error_log": state.error_log + [f"Async processing error: {str(e)}"]
            }
    
    def _create_batches(self, data: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        """Split data into batches"""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches
    
    async def _process_batches_async(self, batches: List[List[Dict[str, Any]]], columns: List[str]) -> tuple:
        """Process batches asynchronously with workers"""
        
        total_batches = len(batches)
        all_corrected_data = []
        all_reviews = []
        all_corrections = []
        
        # Process batches in chunks of MAX_WORKERS
        for chunk_start in range(0, total_batches, MAX_WORKERS):
            chunk_end = min(chunk_start + MAX_WORKERS, total_batches)
            chunk_batches = batches[chunk_start:chunk_end]
            
            print(f"   🔄 Processing batch chunk {chunk_start//MAX_WORKERS + 1}/{(total_batches-1)//MAX_WORKERS + 1}")
            print(f"      Batches {chunk_start} to {chunk_end-1}")
            
            # Create tasks for this chunk
            tasks = []
            for i, batch in enumerate(chunk_batches):
                batch_id = chunk_start + i
                task = self._process_single_batch(batch, batch_id, columns)
                tasks.append(task)
            
            # Execute tasks concurrently
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in chunk_results:
                if isinstance(result, Exception):
                    print(f"      ❌ Batch processing failed: {str(result)}")
                    continue
                
                corrected_data, review, correction = result
                all_corrected_data.extend(corrected_data)
                all_reviews.append(review)
                all_corrections.append(correction)
            
            # Sleep between chunks to avoid rate limiting
            if chunk_end < total_batches:
                print(f"      😴 Sleeping {SLEEP_BETWEEN_BATCHES}s to avoid rate limiting...")
                await asyncio.sleep(SLEEP_BETWEEN_BATCHES)
        
        # Calculate processing stats
        processing_stats = {
            "total_batches": total_batches,
            "batches_processed": len(all_reviews),
            "total_records": len(all_corrected_data),
            "total_issues_found": sum(len([r for r in review.reviews if r.issues]) for review in all_reviews),
            "total_corrections_made": sum(len(correction.corrected_rows) for correction in all_corrections)
        }
        
        return all_corrected_data, all_reviews, all_corrections, processing_stats
    
    async def _process_single_batch(self, batch_data: List[Dict[str, Any]], batch_id: int, columns: List[str]) -> tuple:
        """Process a single batch with retry logic"""
        
        for attempt in range(MAX_RETRIES):
            try:
                print(f"      🎯 Processing batch {batch_id} (attempt {attempt + 1})")
                
                # Step 1: Review the batch
                review = await self.reviewer.review_batch(batch_data, batch_id, columns)
                
                # Step 2: Correct the batch based on review
                correction = await self.corrector.correct_batch(batch_data, review, columns)
                
                # Step 3: Reconstruct corrected data
                corrected_data = self._reconstruct_corrected_data(batch_data, correction)
                
                print(f"      ✅ Batch {batch_id} completed successfully")
                return corrected_data, review, correction
                
            except Exception as e:
                print(f"      ⚠️  Batch {batch_id} attempt {attempt + 1} failed: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    print(f"      ❌ Batch {batch_id} failed after {MAX_RETRIES} attempts, using original data")
                    # Return original data with empty review/correction
                    empty_review = BatchReview(batch_id=batch_id, reviews=[], summary="Failed processing")
                    empty_correction = BatchCorrection(batch_id=batch_id, corrected_rows=[], correction_summary="Failed processing")
                    return batch_data, empty_review, empty_correction
                
                # Wait before retry
                await asyncio.sleep(1)
    
    def _reconstruct_corrected_data(self, original_batch: List[Dict[str, Any]], correction: BatchCorrection) -> List[Dict[str, Any]]:
        """Reconstruct corrected data from correction results"""
        corrected_data = []
        
        # Create a mapping of corrected rows
        corrected_rows_map = {cr.row_index: cr.corrected_data for cr in correction.corrected_rows}
        
        # Reconstruct the full batch with corrections applied
        for i, original_row in enumerate(original_batch):
            if i in corrected_rows_map:
                corrected_data.append(corrected_rows_map[i])
            else:
                corrected_data.append(original_row)
        
        return corrected_data
    
    def finalize_llm_processing(self, state: MedicalDataState) -> dict:
        """Finalize LLM processing with comprehensive statistics"""
        print(f"\n✅ FINALIZING LLM PROCESSING")
        print(f"   🎯 LLM processing pipeline completed successfully")
        
        # Calculate processing statistics
        total_rows = len(state.processed_llm_data) if state.processed_llm_data is not None else 0
        total_batches = len(state.batch_reviews) if state.batch_reviews else 0
        
        # Calculate review statistics
        total_issues = 0
        critical_issues = 0
        high_issues = 0
        medium_issues = 0
        low_issues = 0
        
        for review in state.batch_reviews or []:
            for row_review in review.reviews:
                total_issues += len(row_review.issues)
                for issue in row_review.issues:
                    if issue.severity == "critical":
                        critical_issues += 1
                    elif issue.severity == "high":
                        high_issues += 1
                    elif issue.severity == "medium":
                        medium_issues += 1
                    elif issue.severity == "low":
                        low_issues += 1
        
        # Calculate correction statistics
        total_corrections = 0
        corrected_rows = 0
        
        for correction in state.batch_corrections or []:
            total_corrections += len(correction.corrected_rows)
            corrected_rows += len(correction.corrected_rows)
        
        # Log comprehensive statistics
        print(f"\n📊 LLM PROCESSING STATISTICS:")
        print(f"   📋 Total Records Processed: {total_rows:,}")
        print(f"   📦 Total Batches: {total_batches:,}")
        print(f"   🔍 Issues Identified: {total_issues:,}")
        print(f"      🔴 Critical: {critical_issues:,}")
        print(f"      🟠 High: {high_issues:,}")
        print(f"      🟡 Medium: {medium_issues:,}")
        print(f"      🟢 Low: {low_issues:,}")
        print(f"   🔧 Total Corrections Applied: {total_corrections:,}")
        print(f"   ✏️  Rows with Corrections: {corrected_rows:,}")
        
        if total_rows > 0:
            correction_rate = (corrected_rows / total_rows) * 100
            print(f"   📈 Correction Rate: {correction_rate:.2f}%")
        
        print(f"   ✅ LLM processing completed successfully!")
        
        return {
            "processing_status": "llm_processing_complete",
            "processing_stats": {
                "total_rows": total_rows,
                "total_batches": total_batches,
                "total_issues": total_issues,
                "issue_breakdown": {
                    "critical": critical_issues,
                    "high": high_issues,
                    "medium": medium_issues,
                    "low": low_issues
                },
                "total_corrections": total_corrections,
                "corrected_rows": corrected_rows,
                "correction_rate": (corrected_rows / total_rows * 100) if total_rows > 0 else 0
            }
        }

# Create the subgraph instance  
llm_subgraph = LLMSubgraph().graph
