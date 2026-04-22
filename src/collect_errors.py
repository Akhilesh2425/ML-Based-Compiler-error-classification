"""
Collects compiler error logs from C sample files.
Compiles each file with GCC and captures error output.
"""
import subprocess, os, csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
samples_dir = os.path.join(BASE_DIR, "..", "c_samples")
output_path = os.path.join(BASE_DIR, "..", "data", "raw_logs", "collected_errors.csv")

files = {
    "lexical.c":  "lexical",
    "syntax.c":   "syntax",
    "semantic.c": "semantic"
}

rows = []
for filename, label in files.items():
    filepath = os.path.join(samples_dir, filename)
    result = subprocess.run(
        ["gcc", filepath],
        stderr=subprocess.PIPE,
        text=True
    )
    for line in result.stderr.splitlines():
        if "error:" in line:
            msg = line.split("error:")[-1].strip()
            rows.append((msg, label))
            print(f"[{label}] {msg}")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["error_message", "label"])
    writer.writerows(rows)

print(f"\nCollected {len(rows)} errors → {output_path}")