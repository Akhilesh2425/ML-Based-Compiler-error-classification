import re
from typing import List


def extract_errors_from_build_log(log_text: str) -> List[str]:
    """
    Extract likely compiler error lines from a full build log.
    """
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    error_lines = []

    patterns = [
        r"\berror\b",
        r"fatal error",
        r"undefined reference",
        r"undeclared",
        r"expected .+ before",
    ]
    combined = re.compile("|".join(patterns), re.IGNORECASE)

    for line in lines:
        if combined.search(line):
            error_lines.append(line)

    # Preserve order, remove duplicates.
    seen = set()
    unique_errors = []
    for err in error_lines:
        if err not in seen:
            seen.add(err)
            unique_errors.append(err)
    return unique_errors

