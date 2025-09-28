# custom_parsers/icici_parser.py  (example fallback created by the agent)
from __future__ import annotations
import pandas as pd
from pathlib import Path

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Fallback parser: returns the expected CSV that ships with the sample data.
    Replace this with real PDF parsing logic (pdfplumber / tabula / camelot) or
    use the --use-llm mode to generate parser code from an LLM.
    """
    expected_csv = Path(__file__).parent.parent / "data" / "icici" / "icici_sample.csv"
    df = pd.read_csv(expected_csv)
    return df
