import json
import io
import sys
from contextlib import redirect_stdout

with open("03_agentic_app.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Environment context dictionary for executing cells sequentially
env = {}

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(source, env)
            out_text = buf.getvalue()
            cell["outputs"] = [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": out_text.splitlines(keepends=True)
                }
            ]
            cell["execution_count"] = 1
        except Exception as e:
            cell["outputs"] = [
                {
                    "output_type": "error",
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [str(e)]
                }
            ]
            cell["execution_count"] = 1

with open("03_agentic_app.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("[OK] 03_agentic_app.ipynb executed and cell outputs saved successfully!")
