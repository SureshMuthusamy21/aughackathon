# Configuration for medical data processing

# OpenAI Configuration
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0

# File Processing Configuration
MAX_SAMPLE_ROWS = 2
SUPPORTED_FILE_FORMATS = [".xlsx", ".xls", ".csv"]

# Tool Configuration
TOOL_MAPPING = {
    "id_normalizer": ["patient_id", "doctor_id", "profile_id", "biomarker_stamp", "opbill_id", "centre_id", "row_id"],
    "date_normalizer": ["result_dt", "created_at", "join_dt"],
    "flag_cleaner": ["flag", "status"],
    "text_cleaner": ["lab_site", "location", "address"]
}

# Output Configuration
OUTPUT_FILE_SUFFIX = "_processed"
OUTPUT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
