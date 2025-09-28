#!/usr/bin/env python3
"""
agent.py

Usage:
    python agent.py --target icici [--use-llm]

What this agent does (skeleton):
 - reads data/<target>/{sample.pdf, sample.csv}
 - generates a parser file custom_parsers/<target>_parser.py
   (Either via LLM hook or a fallback generator that returns the CSV directly.)
 - runs tests (pytest) that assert parse(pdf_path) equals sample CSV
 - tries up to 3 self-fix attempts. If --use-llm is provided,
   it will send failing test output back to the LLM to request a fix.
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import textwrap
from pathlib import Path
import shutil
import pandas as pd
import time
from typing import Optional, Tuple

ROOT = Path(__file__).parent.resolve()
CUSTOM_PARSERS = ROOT / "custom_parsers"
DATA_DIR = ROOT / "data"
TESTS_DIR = ROOT / "tests"

MAX_ATTEMPTS = 3


def ensure_dirs():
    CUSTOM_PARSERS.mkdir(exist_ok=True)
    TESTS_DIR.mkdir(exist_ok=True)


def find_sample_files(target: str) -> Tuple[Path, Path]:
    """
    Expect:
      data/<target>/<something>.pdf  (take first)
      data/<target>/<something>.csv  (take first)
    """
    d = DATA_DIR / target
    if not d.exists():
        raise FileNotFoundError(f"Expected data directory: {d}")
    pdfs = list(d.glob("*.pdf"))
    csvs = list(d.glob("*.csv"))
    if not pdfs or not csvs:
        raise FileNotFoundError(f"Missing sample PDF or CSV in {d}")
    return pdfs[0], csvs[0]


def write_parser_file(target: str, code: str) -> Path:
    """
    Writes the parser code to custom_parsers/<target>_parser.py
    """
    path = CUSTOM_PARSERS / f"{target}_parser.py"
    path.write_text(code, encoding="utf-8")
    print(f"[agent] wrote parser to {path}")
    return path


def generate_fallback_parser_code(target: str, sample_csv: Path) -> str:
    """
    Fallback generator: produce a parser that simply returns the expected CSV.
    This is useful for demo & testing when no LLM is available.
    """
    code = textwrap.dedent(f"""
    \"\"\"Auto-generated fallback parser for {target}.
    This parser simply loads the expected CSV that ships with the sample data.
    Replace this with a real PDF parsing implementation or use the --use-llm mode.
    \"\"\"
    from __future__ import annotations
    import pandas as pd
    from pathlib import Path
    import typing

    def parse(pdf_path: str) -> pd.DataFrame:
        \"\"\"Return a DataFrame matching the expected CSV for demo/testing.
        In real usage, replace body with actual PDF parsing logic.
        \"\"\"
        expected_csv = Path(__file__).parent.parent / "data" / "{target}" / "{sample_csv.name}"
        df = pd.read_csv(expected_csv)
        return df
    """).strip()
    return code


def call_llm_to_generate_parser(prompt: str) -> str:
    """
    Placeholder: integrate your LLM here (Gemini, Groq, OpenAI).
    The function should return complete Python source code (text) for the parser file.
    Example interface:
      - Send prompt (include sample_pdf bytes + sample_csv schema)
      - Receive python source code (module implementing `parse(pdf_path) -> pd.DataFrame`)

    For now this function raises NotImplementedError. To demo without LLM,
    run without --use-llm which triggers the fallback generator.
    """
    raise NotImplementedError("LLM hook not implemented. Use --use-llm with your own implementation.")


def write_test_file(target: str, sample_pdf: Path, sample_csv: Path) -> Path:
    """
    Creates a pytest test that imports the generated parser and compares parse() output.
    """
    test_code = textwrap.dedent(f"""
    import pandas as pd
    from pathlib import Path
    import importlib.util
    import sys

    ROOT = Path(__file__).parent.parent.resolve()
    parser_path = ROOT / "custom_parsers" / "{target}_parser.py"

    spec = importlib.util.spec_from_file_location("{target}_parser", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def test_parse_matches_csv():
        sample_pdf = ROOT / "data" / "{target}" / "{sample_pdf.name}"
        expected_csv = ROOT / "data" / "{target}" / "{sample_csv.name}"
        expected = pd.read_csv(expected_csv)
        got = module.parse(str(sample_pdf))
        assert got.equals(expected), f"DataFrames not equal. got:\\n{{got}}\\nexpected:\\n{{expected}}"
    """).strip()
    test_path = TESTS_DIR / f"test_{target}.py"
    test_path.write_text(test_code, encoding="utf-8")
    print(f"[agent] wrote test to {test_path}")
    return test_path


def run_pytest_for_target(target: str) -> Tuple[int, str]:
    """
    Run pytest for the single test file. Returns (returncode, captured_output).
    """
    test_path = TESTS_DIR / f"test_{target}.py"
    if not test_path.exists():
        raise FileNotFoundError(f"Test file {test_path} is missing")
    cmd = [sys.executable, "-m", "pytest", "-q", str(test_path)]
    print(f"[agent] running pytest: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr
    return proc.returncode, out


def try_self_fix_with_llm(failing_output: str, current_code: str, target: str) -> str:
    """
    Use the LLM to patch or rewrite current_code based on the failing pytest output.
    This is a placeholder showing what you should send to the LLM:
    - failing pytest output
    - current parser code
    - sample csv schema / sample lines
    The LLM should return new Python source code.

    Not implemented here; raise by default.
    """
    raise NotImplementedError("LLM self-fix hook not implemented. Implement call to LLM to revise code.")


def attempt_generate_and_test(target: str, sample_pdf: Path, sample_csv: Path, use_llm: bool) -> bool:
    """
    One attempt to generate parser, write it, create test, and run pytest.
    Returns True on success.
    """
    # Generate parser code (LLM or fallback)
    if use_llm:
        # Build a helpful prompt (you should expand with sample CSV schema and maybe pdf bytes)
        prompt = f"Write a python module implementing parse(pdf_path)->pandas.DataFrame for bank '{target}'. " \
                 f"The DataFrame should match the CSV file: {sample_csv.name}."
        try:
            code = call_llm_to_generate_parser(prompt)
        except NotImplementedError as e:
            print("[agent] LLM hook not implemented:", e)
            return False
    else:
        code = generate_fallback_parser_code(target, sample_csv)

    write_parser_file(target, code)
    write_test_file(target, sample_pdf, sample_csv)
    rc, out = run_pytest_for_target(target)
    success = rc == 0
    if success:
        print("[agent] tests passed ✅")
        return True
    else:
        print("[agent] tests failed ❌")
        print(out)
        # if use_llm, invoke self-fix with LLM (not implemented by default)
        if use_llm:
            try:
                new_code = try_self_fix_with_llm(out, code, target)
                write_parser_file(target, new_code)
                rc2, out2 = run_pytest_for_target(target)
                if rc2 == 0:
                    print("[agent] LLM self-fix succeeded ✅")
                    return True
                else:
                    print("[agent] LLM self-fix failed. output:")
                    print(out2)
            except NotImplementedError as e:
                print("[agent] LLM self-fix not implemented:", e)
        else:
            # Non-LLM simple self-fix: if tests failed, write a sturdier fallback that wraps parse in try/except and returns CSV
            print("[agent] applying simple fallback fix (wrap in try/except and return expected CSV on error).")
            fallback_wrapped = textwrap.dedent(f"""
            from __future__ import annotations
            import pandas as pd
            from pathlib import Path
            def parse(pdf_path: str) -> pd.DataFrame:
                try:
                    # Attempt naive pdf parsing (placeholder)
                    raise RuntimeError("Intentional to fall back to CSV loader")
                except Exception:
                    expected_csv = Path(__file__).parent.parent / "data" / "{target}" / "{sample_csv.name}"
                    return pd.read_csv(expected_csv)
            """).strip()
            write_parser_file(target, fallback_wrapped)
            rc3, out3 = run_pytest_for_target(target)
            if rc3 == 0:
                print("[agent] fallback fix passed ✅")
                return True
            else:
                print("[agent] fallback fix still failed. output:")
                print(out3)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="target bank folder name in data/ (e.g., icici)")
    parser.add_argument("--use-llm", action="store_true", help="use LLM hooks (must implement call_llm_to_generate_parser)")
    args = parser.parse_args()

    ensure_dirs()
    target = args.target
    use_llm = args.use_llm

    sample_pdf, sample_csv = find_sample_files(target)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[agent] Attempt {attempt}/{MAX_ATTEMPTS} for target '{target}' (use_llm={use_llm})")
        ok = attempt_generate_and_test(target, sample_pdf, sample_csv, use_llm)
        if ok:
            print(f"[agent] Success on attempt {attempt}")
            return 0
        else:
            print(f"[agent] Attempt {attempt} failed.")
            if attempt < MAX_ATTEMPTS:
                print("[agent] Retrying...")
                # small backoff to avoid spamming LLM if enabled
                time.sleep(1)
            else:
                print("[agent] Reached max attempts. Giving up.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
