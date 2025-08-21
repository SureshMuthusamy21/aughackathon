import os
import asyncio
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from models.state import BatchReview, BatchCorrection, MedicalIssue, RowReview, CorrectedRow
import pandas as pd

class MedicalDataReviewer:
    """Expert medical data reviewer agent that identifies issues in medical data"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1
        )
    
    async def review_batch(self, batch_data: List[Dict[str, Any]], batch_id: int, columns: List[str]) -> BatchReview:
        """Review a batch of medical data for issues"""
        
        print(f"   👨‍⚕️ Medical Data Reviewer analyzing batch {batch_id} ({len(batch_data)} records)")
        
        # Prepare data for analysis
        data_text = self._format_batch_for_review(batch_data, columns)
        
        system_prompt = """You are a Senior Medical Data Quality Expert with 20+ years of experience in healthcare informatics and medical records management.

Your expertise includes:
- Medical terminology standardization (ICD-10, SNOMED CT, CPT codes)
- Healthcare data quality standards (HL7, FHIR)
- Clinical documentation best practices
- Medical abbreviations and nomenclature
- Laboratory values and reference ranges
- Pharmaceutical naming conventions

ANALYZE the provided medical data records and identify ALL data quality issues for EACH ROW.

ISSUE TYPES to detect:
1. SPELLING: Misspelled medical terms, drug names, anatomical references
2. TERMINOLOGY: Inconsistent medical terminology, non-standard abbreviations
3. MISSING: Missing critical medical values, incomplete records
4. DUPLICATE: Duplicate patient records, redundant information
5. FORMAT: Inconsistent date formats, measurement units, case variations
6. OUTLIER: Anomalous medical values, impossible measurements, age inconsistencies
7. OTHER: Any other medical data quality issues

SEVERITY LEVELS:
- CRITICAL: Could impact patient safety or treatment decisions
- HIGH: Significant data quality issue affecting analysis
- MEDIUM: Moderate issue that should be corrected
- LOW: Minor formatting or consistency issue

For each row, provide:
1. Row index (0-based)
2. List of all issues found
3. Overall status assessment
4. Specific medical corrections where appropriate

Respond in JSON format following the BatchReview schema."""

        human_prompt = f"""Analyze the following medical data batch:

BATCH ID: {batch_id}
COLUMNS: {', '.join(columns)}
RECORDS COUNT: {len(batch_data)}

DATA:
{data_text}

Provide a comprehensive medical data quality review following standard healthcare informatics practices."""

        try:
            # Get structured response
            structured_llm = self.llm.with_structured_output(BatchReview)
            review = await structured_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            
            print(f"   ✅ Review completed for batch {batch_id}")
            print(f"      📊 Found issues in {len([r for r in review.reviews if r.issues])} records")
            
            return review
            
        except Exception as e:
            print(f"   ❌ Review failed for batch {batch_id}: {str(e)}")
            # Return empty review on failure
            return BatchReview(
                batch_id=batch_id,
                reviews=[RowReview(row_index=i, issues=[], overall_status="error") for i in range(len(batch_data))],
                summary=f"Review failed: {str(e)}"
            )
    
    def _format_batch_for_review(self, batch_data: List[Dict[str, Any]], columns: List[str]) -> str:
        """Format batch data for medical review"""
        formatted_lines = []
        
        for i, row in enumerate(batch_data):
            row_data = []
            for col in columns:
                value = row.get(col, "")
                row_data.append(f"{col}: {value}")
            formatted_lines.append(f"Row {i}: {' | '.join(row_data)}")
        
        return "\n".join(formatted_lines)


class MedicalDataCorrector:
    """Expert medical data corrector agent that applies corrections to medical data"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1
        )
    
    async def correct_batch(self, batch_data: List[Dict[str, Any]], review: BatchReview, columns: List[str]) -> BatchCorrection:
        """Correct medical data based on review findings"""
        
        print(f"   🔧 Medical Data Corrector processing batch {review.batch_id}")
        
        # Prepare correction context
        correction_context = self._format_correction_context(batch_data, review, columns)
        
        system_prompt = """You are a Senior Medical Data Correction Specialist with expertise in:
- Medical terminology standardization (WHO, ICD-10, SNOMED CT)
- Healthcare data normalization standards
- Clinical documentation guidelines
- Pharmaceutical nomenclature (FDA, WHO-DD)
- Laboratory reference values and units
- Medical abbreviation standards

Your task is to CORRECT medical data based on expert review findings while maintaining data integrity and following medical standards.

CORRECTION PRINCIPLES:
1. Maintain medical accuracy and clinical relevance
2. Follow standard medical terminology and nomenclature
3. Preserve original meaning while correcting format/spelling
4. Use internationally recognized medical standards
5. Ensure consistency across similar data elements
6. Do not make corrections that could alter clinical meaning without clear evidence

SPECIFIC DATA TYPE CORRECTIONS:

1. LAB TEST NAMES:
   - Remove unwanted punctuation (semicolons, extra spaces)
   - Standardize capitalization: "HCG - Beta Specific (Serum)"
   - Standard format: "Test Name (Specimen Type)"
   - Examples:
     * "hcg - beta specific;serum" → "HCG - Beta Specific (Serum)"
     * "cbc (complete blood count);edta" → "Complete Blood Count (EDTA)"
     * "HEPATITIS B SURFACE ANTIGEN;SERUM" → "Hepatitis B Surface Antigen (Serum)"
     * "HIV TEST ( I AND II );SERUM" → "HIV Test (I and II) (Serum)"

2. LAB RESULTS:
   - Remove "approx" and replace with exact values when possible
   - Standardize decimal separators (use . not ,)
   - Add proper units when missing
   - Remove unnecessary symbols (* unless clinically significant)
   - Clean formatting:
     * "225,69" → "225.69"
     * "181.79 approx" → "181.79"
     * "256,47 IU/L" → "256.47 IU/L"
     * " 171.21 mg/dL " → "171.21 mg/dL"

3. VITAL SIGNS (vital_remark):
   - Standardize format as JSON or structured text
   - Examples:
     * "HR normal" → "Heart Rate: Normal"
     * "BP elevated" → "Blood Pressure: Elevated"
     * "Resp rate high" → "Respiratory Rate: High"
     * "No vitals recorded" → "No Vitals Recorded"
   - For complex vitals like "BP:120/80;HR:78":
     * Convert to: "Blood Pressure: 120/80 mmHg, Heart Rate: 78 bpm"

4. CLINICAL NOTES:
   - Fix abbreviations: "pt c/o" → "Patient complains of"
   - Correct spelling and grammar
   - Standardize format:
     * "pt c/o chest pain" → "Patient complains of chest pain"
     * "mild weakness since 2 days" → "Mild weakness for 2 days"
     * "Follow up after 3 weeks" → "Follow-up scheduled in 3 weeks"
   - Remove unnecessary symbols (* unless clinically significant)

5. DIAGNOSIS FIELDS (provisional/final diagnosis):
   - Check medical spelling and terminology
   - Ensure proper medical grammar
   - Standardize medical abbreviations
   - Correct case formatting
   - Examples:
     * "dementia (Alzheimer's type)" → "Dementia (Alzheimer's Type)"
     * "b/l epididymitis" → "Bilateral Epididymitis"
     * "gastro paresis" → "Gastroparesis"

6. GENERAL FORMATTING:
   - Remove leading/trailing spaces
   - Standardize capitalization (Title Case for medical terms)
   - Remove duplicate entries
   - Ensure consistency across similar fields
   - Handle special characters appropriately

CORRECTION TYPES:
- Spelling corrections for medical terms
- Standardize medical terminology and abbreviations
- Format corrections (dates, measurements, case)
- Fill missing values only if derivable from context
- Remove clear duplicates
- Normalize outliers to standard ranges (when appropriate)
- Clean unwanted punctuation and symbols
- Standardize units and measurements

CRITICAL: You must return data in this EXACT JSON format:
{
  "batch_id": <integer>,
  "corrected_rows": [
    {
      "row_index": <integer>,
      "corrected_data": {<complete corrected row data as key-value pairs>},
      "corrections_made": ["description of correction 1", "description of correction 2"]
    }
  ],
  "correction_summary": "Brief summary of all corrections applied"
}

IMPORTANT: 
- "corrected_data" must contain the COMPLETE corrected row as a dictionary
- Include ALL columns with corrected values, not just changed ones
- Do NOT use "corrected_description" - use "corrected_data"
- Only include rows that needed corrections
- Provide detailed correction descriptions for each change made

Respond in JSON format following the BatchCorrection schema exactly."""

        human_prompt = f"""Apply medical data corrections based on the expert review:

{correction_context}

Provide corrected data following medical standards and best practices."""

        try:
            # Get structured response
            structured_llm = self.llm.with_structured_output(BatchCorrection)
            correction = await structured_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            
            # Validate the response structure
            if not isinstance(correction, BatchCorrection):
                raise ValueError(f"Invalid response type: {type(correction)}")
            
            # Validate each corrected row has the required fields
            for row in correction.corrected_rows:
                if not hasattr(row, 'corrected_data') or row.corrected_data is None:
                    raise ValueError(f"Missing 'corrected_data' field in row {row.row_index}")
                if not isinstance(row.corrected_data, dict):
                    raise ValueError(f"'corrected_data' must be a dictionary for row {row.row_index}")
            
            print(f"   ✅ Corrections applied for batch {review.batch_id}")
            print(f"      📝 Corrected {len(correction.corrected_rows)} records")
            
            return correction
            
        except Exception as e:
            print(f"   ❌ Correction failed for batch {review.batch_id}: {str(e)}")
            print(f"      🔧 Creating fallback correction with original data")
            
            # Return original data on failure
            return BatchCorrection(
                batch_id=review.batch_id,
                corrected_rows=[
                    CorrectedRow(row_index=i, corrected_data=row, corrections_made=[f"Error during correction: {str(e)}"])
                    for i, row in enumerate(batch_data)
                ],
                correction_summary=f"Correction failed, returned original data: {str(e)}"
            )
    
    def _format_correction_context(self, batch_data: List[Dict[str, Any]], review: BatchReview, columns: List[str]) -> str:
        """Format correction context for the corrector agent"""
        context_lines = [
            f"BATCH ID: {review.batch_id}",
            f"COLUMNS: {', '.join(columns)}",
            "",
            "REVIEW SUMMARY:",
            review.summary,
            "",
            "DETAILED ISSUES AND DATA:"
        ]
        
        for row_review in review.reviews:
            if row_review.issues:  # Only include rows with issues
                row_data = batch_data[row_review.row_index]
                context_lines.append(f"\nRow {row_review.row_index} - Status: {row_review.overall_status}")
                
                # Add row data
                row_info = []
                for col in columns:
                    value = row_data.get(col, "")
                    row_info.append(f"{col}: {value}")
                context_lines.append(f"Data: {' | '.join(row_info)}")
                
                # Add identified issues
                context_lines.append("Issues:")
                for issue in row_review.issues:
                    context_lines.append(f"  - {issue.issue_type} ({issue.severity}): {issue.description}")
                    if issue.suggested_correction:
                        context_lines.append(f"    Suggested: {issue.suggested_correction}")
        
        return "\n".join(context_lines)
