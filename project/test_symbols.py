import sys; sys.path.insert(0, ".")
from backend.math_engine import solve, detect_operation

cases = [
    ("5 - 3 किती",   2),
    ("10 - 4 किती",  6),
    ("15-5 किती",    10),
    ("5 + 3 किती",   8),
    ("3 * 4 किती",   12),
    ("20 / 4 किती",  5),
    ("7 - 2",        5),
    ("100 - 1 किती", 99),
]
all_pass = True
for text, expected in cases:
    r = solve(text)
    ok = r.get("result") == expected
    if not ok: all_pass = False
    print(f"  [{'PASS' if ok else 'FAIL'}] \"{text}\" => {r.get('marathi_result', r.get('error'))}")

print()
print("All PASS!" if all_pass else "Some FAILED!")
