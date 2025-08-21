import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
import io
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import warnings
import logging

# Suppress PyArrow warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
logging.getLogger('streamlit.dataframe_util').setLevel(logging.ERROR)
    

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the main processing function
from src.main import process_medical_data

# Page configuration
st.set_page_config(
    page_title="🏥 Medical Data Processing System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stProgress > div > div > div > div {
        background-color: #1e3c72;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Data Processing System</h1>
        <p>Advanced AI-powered medical data cleaning and standardization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 File Upload")
        
        # Environment check
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("✅ OpenAI API Key configured")
        else:
            st.error("❌ OpenAI API Key not found")
            st.warning("Please set OPENAI_API_KEY in your .env file")
            return
        
        # File upload options
        upload_option = st.radio(
            "Choose upload method:",
            ["Upload new file", "Use existing file"]
        )
        
        uploaded_file = None
        file_path = None
        
        if upload_option == "Upload new file":
            uploaded_file = st.file_uploader(
                "Choose a medical data file",
                type=['xlsx', 'xls', 'csv'],
                help="Upload an Excel or CSV file containing medical data"
            )
            
            if uploaded_file is not None:
                # Save uploaded file to data directory
                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)
                
                file_path = data_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"✅ File saved: {uploaded_file.name}")
        
        else:
            # List existing files in data directory
            data_dir = Path("data")
            if data_dir.exists():
                existing_files = [f for f in data_dir.glob("*") if f.suffix.lower() in ['.xlsx', '.xls', '.csv']]
                
                if existing_files:
                    selected_file = st.selectbox(
                        "Select existing file:",
                        options=[f.name for f in existing_files]
                    )
                    file_path = data_dir / selected_file
                else:
                    st.warning("No data files found in the data directory")
            else:
                st.warning("Data directory not found")
        
        # Processing settings
        st.header("⚙️ Processing Settings")
        
        show_preview = st.checkbox("Show data preview", value=True)
        show_details = st.checkbox("Show processing details", value=True)
        
        # Process button
        process_button = st.button(
            "🚀 Start Processing",
            disabled=(file_path is None),
            use_container_width=True
        )
    
    # Main content area
    if file_path and file_path.exists():
        
        # File preview section
        if show_preview:
            st.header("📊 Data Preview")
            
            try:
                # Load and display file preview
                if file_path.suffix.lower() == '.csv':
                    df_preview = pd.read_csv(file_path)
                else:
                    df_preview = pd.read_excel(file_path)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df_preview))
                with col2:
                    st.metric("Total Columns", len(df_preview.columns))
                with col3:
                    st.metric("File Size", f"{file_path.stat().st_size / 1024:.1f} KB")
                
                # Show preview
                st.subheader("📋 First 10 rows:")
                st.dataframe(df_preview.head(10), use_container_width=True)
                
                # Column information
                with st.expander("📝 Column Details"):
                    col_info = pd.DataFrame({
                        'Column': df_preview.columns,
                        'Data Type': df_preview.dtypes,
                        'Non-Null Count': df_preview.count(),
                        'Null Count': df_preview.isnull().sum(),
                        'Sample Values': [str(df_preview[col].dropna().iloc[0]) if not df_preview[col].dropna().empty else 'N/A' for col in df_preview.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error loading file preview: {str(e)}")
        
        # Processing section
        if process_button:
            st.header("🔄 Processing Status")
            
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create columns for real-time updates
            col1, col2 = st.columns(2)
            
            with col1:
                processing_status = st.empty()
            with col2:
                processing_logs = st.empty()
            
            try:
                # Update progress
                progress_bar.progress(10)
                status_text.text("🔍 Initializing processing...")
                
                # Call the processing function
                result = process_medical_data(str(file_path))
                
                progress_bar.progress(100)
                status_text.text("✅ Processing completed!")
                
                # Display results
                st.header("📈 Processing Results")
                
                if result.get("status") == "error":
                    st.markdown(f"""
                    <div class="error-box">
                        <h4>❌ Processing Failed</h4>
                        <p>{result.get('message', 'Unknown error occurred')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    # Success metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Status", result.get("status", "Unknown"))
                    with col2:
                        rule_cols = len(result.get("rule_based_columns", {}))
                        st.metric("Rule-based Columns", rule_cols)
                    with col3:
                        llm_cols = len(result.get("llm_columns", []))
                        st.metric("LLM-processed Columns", llm_cols)
                    with col4:
                        error_count = len(result.get("errors", []))
                        st.metric("Errors", error_count)
                    
                    # Processing details
                    if show_details:
                        # Rule-based columns expander
                        if result.get("rule_based_columns"):
                            rule_columns = []
                            rule_data = result["rule_based_columns"]
                            
                            # Handle different possible structures
                            if isinstance(rule_data, dict):
                                # Check if it's {tool_name: [column_names]} or {column_name: tool_name}
                                first_key = next(iter(rule_data.keys()))
                                first_value = rule_data[first_key]
                                
                                if isinstance(first_value, list):
                                    # Structure: {tool_name: [column_names]}
                                    for tool, columns in rule_data.items():
                                        if isinstance(columns, list):
                                            rule_columns.extend(columns)
                                        else:
                                            rule_columns.append(str(columns))
                                else:
                                    # Structure: {column_name: tool_name}
                                    rule_columns = list(rule_data.keys())
                            elif isinstance(rule_data, list):
                                rule_columns = rule_data
                            
                            if rule_columns:
                                with st.expander(f"⚙️ Rule-based Processing ({len(rule_columns)} columns)"):
                                    cols = st.columns(3)
                                    for i, col in enumerate(rule_columns):
                                        with cols[i % 3]:
                                            st.write(f"• {col}")
                        
                        # LLM columns expander
                        if result.get("llm_columns"):
                            llm_columns = result["llm_columns"]
                            with st.expander(f"🧠 LLM Processing ({len(llm_columns)} columns)"):
                                cols = st.columns(3)
                                for i, col in enumerate(llm_columns):
                                    with cols[i % 3]:
                                        st.write(f"• {col}")
                        
                        # Errors
                        if result.get("errors"):
                            st.subheader("⚠️ Processing Errors")
                            for i, error in enumerate(result["errors"], 1):
                                st.error(f"{i}. {error}")
                    
                    # Download section
                    output_file = result.get("output_file")
                    if output_file and Path(output_file).exists():
                        st.header("⬇️ Download Processed Data")
                        
                        # Load processed data for preview
                        try:
                            processed_df = pd.read_excel(output_file)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Processed Rows", len(processed_df))
                            with col2:
                                st.metric("Output File Size", f"{Path(output_file).stat().st_size / 1024:.1f} KB")
                            
                            # Show comparison
                            with st.expander("📊 Before vs After Comparison"):
                                tab1, tab2 = st.tabs(["Original Data", "Processed Data"])
                                
                                with tab1:
                                    st.dataframe(df_preview.head(), use_container_width=True)
                                
                                with tab2:
                                    st.dataframe(processed_df.head(), use_container_width=True)
                            
                            # Download button
                            with open(output_file, "rb") as file:
                                st.download_button(
                                    label="📥 Download Processed Excel File",
                                    data=file.read(),
                                    file_name=Path(output_file).name,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                                
                        except Exception as e:
                            st.error(f"❌ Error loading processed file: {str(e)}")
                    
                    else:
                        st.warning("⚠️ No output file generated")
            
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("❌ Processing failed")
                st.markdown(f"""
                <div class="error-box">
                    <h4>❌ Processing Error</h4>
                    <p>{str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Welcome message when no file is selected
        st.header("👋 Welcome to Medical Data Processing System")
        
        st.markdown("""
        ### 🎯 What this system does:
        
        1. **🔍 Schema Analysis**: AI-powered analysis of your data structure
        2. **⚙️ Rule-based Processing**: Standardizes IDs, dates, flags, and basic text
        3. **🧠 LLM Processing**: Advanced AI correction for medical terminology and clinical data
        4. **📊 Data Validation**: Comprehensive error checking and data quality assessment
        5. **📥 Easy Download**: Get your cleaned data in Excel format
        
        ### 🚀 Getting Started:
        
        1. Upload your medical data file (Excel or CSV) using the sidebar
        2. Preview your data to understand the structure
        3. Click "Start Processing" to begin the AI-powered cleaning
        4. Download your processed and standardized data
        
        ### 📋 Supported Data Types:
        
        - **Patient Information**: IDs, demographics, contact details
        - **Medical Records**: Lab results, vital signs, diagnoses
        - **Clinical Notes**: Free-text medical observations
        - **Temporal Data**: Dates, timestamps, visit records
        - **Structured Data**: Flags, statuses, categories
        """)
        
        # System status
        st.header("🛠️ System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if api_key:
                st.success("✅ OpenAI API Connected")
            else:
                st.error("❌ API Key Missing")
        
        with col2:
            src_path = Path("src")
            if src_path.exists():
                st.success("✅ Processing Engine Ready")
            else:
                st.error("❌ Processing Engine Not Found")
        
        with col3:
            data_path = Path("data")
            if data_path.exists():
                st.success("✅ Data Directory Available")
            else:
                st.warning("⚠️ Data Directory Not Found")

if __name__ == "__main__":
    main()
