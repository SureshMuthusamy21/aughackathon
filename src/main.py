import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first, before any other imports
load_dotenv()

# Add the src directory to the Python path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

try:
    from models.state import MedicalDataState
    from graphs.main_graph import medical_data_processor
except ImportError:
    # Fallback for relative imports when run as module
    from .models.state import MedicalDataState
    from .graphs.main_graph import medical_data_processor

def process_medical_data(file_path: str) -> dict:
    """
    Main entry point for processing medical data
    
    Args:
        file_path: Path to the Excel/CSV file to process
        
    Returns:
        dict: Processing results including output file path and status
    """
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please check your .env file or set the environment variable")
        return {
            "status": "error",
            "message": "OPENAI_API_KEY environment variable not set"
        }
    
    print(f"Using OpenAI API key: {api_key[:10]}...")  # Show first 10 chars for verification
    
    # Check if file exists
    if not os.path.exists(file_path):
        return {
            "status": "error",
            "message": f"File not found: {file_path}"
        }
    
    # Initialize state
    initial_state = MedicalDataState(file_path=file_path)
    
    # Create thread configuration
    config = {"configurable": {"thread_id": "medical_processing_thread"}}
    
    try:
        print(f"Starting processing of file: {file_path}")
        
        # Run the processing graph
        result = medical_data_processor.invoke(initial_state, config=config)
        
        return {
            "status": result.get("processing_status", "unknown"),
            "output_file": result.get("output_file_path"),
            "errors": result.get("error_log", []),
            "rule_based_columns": result.get("rule_based_columns", {}),
            "llm_columns": result.get("llm_columns", [])
        }
        
    except Exception as e:
        print(f"Processing failed with error: {str(e)}")
        return {
            "status": "error", 
            "message": f"Processing failed: {str(e)}"
        }

if __name__ == "__main__":
    # Check if .env file exists
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        print("Warning: .env file not found. Please create one with your OPENAI_API_KEY")
        print(f"Expected location: {env_file}")
    
    # Example usage
    file_path = r"data\Aug_hackathon_medical_data.xlsx"
    result = process_medical_data(file_path)
    print(f"Processing result: {result}")
