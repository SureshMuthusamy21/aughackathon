# Medical Data Processing System

A multi-agent system for processing medical data using LangGraph with schema analysis and parallel processing.

## Project Structure

```
aughackathon/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── state.py               # Pydantic models and state management
│   ├── tools/
│   │   ├── __init__.py
│   │   └── data_tools.py          # Data processing tools (ID, Date, Flag, Text normalizers)
│   ├── subgraphs/
│   │   ├── __init__.py
│   │   ├── schema_subgraph.py     # LLM-based schema analysis
│   │   ├── rule_based_subgraph.py # Rule-based data processing
│   │   └── llm_subgraph.py        # LLM-based data processing (placeholder)
│   ├── graphs/
│   │   ├── __init__.py
│   │   └── main_graph.py          # Main orchestration graph
│   ├── config.py                  # Configuration settings
│   ├── main.py                    # Main entry point
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_data_tools.py         # Unit tests for data tools
├── data/
│   └── sample_medical_data.csv    # Sample data for testing
├── referencefile/                 # Reference implementations
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Features

### 1. Schema Analysis Agent
- Uses GPT-4o to analyze Excel column headers and sample data
- Categorizes columns into "rule_based" or "llm_based" processing
- Maps rule-based columns to appropriate processing tools

### 2. Data Processing Tools
- **ID Normalizer**: Cleans ID columns (patient_id, doctor_id, etc.)
- **Date Normalizer**: Standardizes dates to YYYY-MM-DD HH:MM:SS format
- **Flag Cleaner**: Standardizes flags to uppercase without special characters
- **Text Cleaner**: Basic text cleaning for simple text columns

### 3. Parallel Processing
- Rule-based and LLM subgraphs run in parallel
- Results are merged back into the original DataFrame structure
- Error handling at individual cell level

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up OpenAI API key:
```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage
```python
from src.main import process_medical_data

result = process_medical_data("path/to/your/medical_data.xlsx")
print(result)
```

### Command Line Usage
```python
python src/main.py
# Enter file path when prompted
```

### Expected Output
```python
{
    "status": "completed",
    "output_file": "path/to/output_file_processed_20250820_143022.xlsx", 
    "errors": [],
    "rule_based_columns": {"patient_id": "id_normalizer", "result_dt": "date_normalizer"},
    "llm_columns": ["diagnosis", "medical_notes"]
}
```

## Configuration

Edit `src/config.py` to customize:
- OpenAI model settings
- File processing parameters
- Tool mappings
- Output configurations

## Testing

Run unit tests:
```bash
python -m pytest tests/
```

## Architecture

The system follows a multi-agent architecture with clear separation of concerns:

1. **Main Graph**: Orchestrates the entire workflow
2. **Schema Subgraph**: Analyzes and categorizes columns using LLM
3. **Rule-Based Subgraph**: Applies predefined tools to structured data
4. **LLM Subgraph**: Handles complex medical data (placeholder for future enhancement)

## Error Handling

- Individual cell processing errors don't stop the entire process
- Original values are preserved when processing fails
- Comprehensive error logging throughout the pipeline

## Future Enhancements

- Implement full LLM subgraph for complex medical data processing
- Add support for additional file formats
- Implement batch processing capabilities
- Add more sophisticated medical data processing tools
