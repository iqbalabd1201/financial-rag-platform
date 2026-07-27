"""Clone FinanceBench and copy just the 22 PDFs this project's eval set uses.

Kept separate from parse_pdf.py so raw-download logic (which depends on
network/git) is isolated from pure-parsing logic (which is unit-testable
without network access).
"""
import os
import shutil
import subprocess

FINANCEBENCH_REPO = "https://github.com/patronus-ai/financebench.git"


def clone_financebench(dest: str = "/content/financebench") -> str:
    """Clone the FinanceBench repo (source of both PDFs and gold QA jsonl)."""
    if not os.path.exists(dest):
        subprocess.run(["git", "clone", "--depth", "1", FINANCEBENCH_REPO, dest], check=True)
    return dest


def copy_pdfs(financebench_dir: str, target_dir: str, doc_ids: list[str]):
    """Copy only the PDFs this project's eval set needs, not all 84+.

    Returns the list of doc_ids that were NOT found (should be empty).
    """
    os.makedirs(target_dir, exist_ok=True)
    src_dir = os.path.join(financebench_dir, "pdfs")
    missing = []
    for doc_id in doc_ids:
        src = os.path.join(src_dir, f"{doc_id}.pdf")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(target_dir, f"{doc_id}.pdf"))
        else:
            missing.append(doc_id)
    return missing
