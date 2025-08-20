from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd
import os

try:
    from models.state import MedicalDataState, SchemaAnalysisResult
except ImportError:
    import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd

# Add the src directory to the Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

try:
    from models.state import MedicalDataState, SchemaAnalysisResult
except ImportError:
    from ..models.state import MedicalDataState, SchemaAnalysisResult

class SchemaAnalysisSubgraph:
    
    def __init__(self):
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            temperature=0
        )
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the schema analysis subgraph"""
        graph_builder = StateGraph(MedicalDataState)
        
        # Add nodes
        graph_builder.add_node("load_file_data", self.load_file_data)
        graph_builder.add_node("analyze_schema", self.analyze_schema)
        graph_builder.add_node("categorize_columns", self.categorize_columns)
        
        # Add edges
        graph_builder.add_edge(START, "load_file_data")
        graph_builder.add_edge("load_file_data", "analyze_schema")
        graph_builder.add_edge("analyze_schema", "categorize_columns")
        graph_builder.add_edge("categorize_columns", END)
        
        # Compile with memory
        memory = MemorySaver()
        return graph_builder.compile(checkpointer=memory)
    
    def load_file_data(self, state: MedicalDataState) -> dict:
        """Load Excel file and extract headers + sample data"""
        try:
            file_path = state.file_path
            if not file_path or not os.path.exists(file_path):
                return {
                    "processing_status": "error",
                    "error_log": state.error_log + ["File not found"]
                }
            
            # Read Excel file
            df = pd.read_excel(file_path)
            
            if df.empty:
                return {
                    "processing_status": "error", 
                    "error_log": state.error_log + ["Empty file"]
                }
            
            # Extract headers and top 2 rows
            headers = df.columns.tolist()
            sample_data = df.head(2).to_dict('records')
            
            # Save the DataFrame temporarily to a pickle file for later use
            temp_file = f"temp_data_{hash(file_path)}.pkl"
            df.to_pickle(temp_file)
            
            return {
                "headers": headers,
                "sample_data": sample_data,
                "data_shape": df.shape,
                "processing_status": "file_loaded"
            }
            
        except Exception as e:
            return {
                "processing_status": "error",
                "error_log": state.error_log + [f"File loading error: {str(e)}"]
            }
    
    def analyze_schema(self, state: MedicalDataState) -> dict:
        """Analyze schema using LLM"""
        try:
            headers = state.headers
            sample_data = state.sample_data
            
            # Prepare sample data for LLM
            sample_text = ""
            for i, row in enumerate(sample_data):
                sample_text += f"Row {i+1}:\n"
                for col in headers:
                    value = row.get(col, "")
                    sample_text += f"  {col}: {value}\n"
                sample_text += "\n"
            
            # Create system message with tool definitions
            system_message = SystemMessage(content=f"""
You are a medical data schema analyzer. Analyze the column names and sample values to categorize each column.

Available Tools:
1. id_normalizer: For ID columns (patient_id, doctor_id, profile_id, biomarker_stamp, opbill_id, centre_id, row_id)
   - Use for: IDs that need cleaning of spaces/special chars
   
2. date_normalizer: For date columns (result_dt, created_at, join_dt)
   - Use for: Any date/time data that needs standardization
   
3. flag_cleaner: For flag/status columns 
   - Use for: Status flags that need uppercase standardization
   
4. text_cleaner: For simple text columns (lab_site)
   - Use for: Text that needs basic cleaning (not complex medical terms)

Rules:
- Use "rule_based" ONLY if the column clearly fits one of the 4 tools above
- Use "llm_based" for complex medical data, diagnostic codes, or anything requiring interpretation
- Be conservative - when in doubt, choose "llm_based"

Column Headers: {headers}

Sample Data:
{sample_text}

For each column, provide:
1. column_name
2. category: "rule_based" or "llm_based" 
3. tool_name: (only if rule_based)
4. reasoning: brief explanation

Respond in JSON format only.
""")
            
            human_message = HumanMessage(content="Analyze the schema and categorize columns.")
            
            # Get structured response
            structured_llm = self.llm.with_structured_output(SchemaAnalysisResult)
            response = structured_llm.invoke([system_message, human_message])
            
            return {
                "schema_analysis": response,
                "processing_status": "schema_analyzed"
            }
            
        except Exception as e:
            return {
                "processing_status": "error",
                "error_log": state.error_log + [f"Schema analysis error: {str(e)}"]
            }
    
    def categorize_columns(self, state: MedicalDataState) -> dict:
        """Categorize columns based on analysis results"""
        try:
            schema_analysis = state.schema_analysis
            
            rule_based_columns = {}
            llm_columns = []
            
            for column_analysis in schema_analysis.columns:
                if column_analysis.category == "rule_based":
                    rule_based_columns[column_analysis.column_name] = column_analysis.tool_name
                else:
                    llm_columns.append(column_analysis.column_name)
            
            return {
                "rule_based_columns": rule_based_columns,
                "llm_columns": llm_columns,
                "processing_status": "columns_categorized"
            }
            
        except Exception as e:
            return {
                "processing_status": "error",
                "error_log": state.error_log + [f"Categorization error: {str(e)}"]
            }

# Create the subgraph instance only if API key is available
try:
    schema_analysis_subgraph = SchemaAnalysisSubgraph().graph
except ValueError as e:
    print(f"Warning: {e}")
    schema_analysis_subgraph = None
