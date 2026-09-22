"""
Generates the complete improved Bhavya_HospitalPatientAppointmentAnalytics.ipynb
Run: python build_notebook.py
"""
import json, textwrap

def md(id_, *lines):
    return {"cell_type": "markdown", "id": id_, "metadata": {}, "source": list(lines)}

def code(id_, *lines):
    return {"cell_type": "code", "execution_count": None, "id": id_,
            "metadata": {}, "outputs": [], "source": list(lines)}

# ─────────────────────────────────────────────────────────────────────────────
# Load the original to extract all code cells (so we never touch ML logic)
# ─────────────────────────────────────────────────────────────────────────────
with open("_original_nb.ipynb", encoding="utf-8") as f:
    orig = json.load(f)

orig_cells = {c["id"]: c for c in orig["cells"]}

def orig_code(id_):
    """Return the original code cell by id (unchanged)."""
    return orig_cells[id_]

def orig_md(id_):
    """Return the original markdown cell by id (unchanged)."""
    return orig_cells[id_]

print("Builder loaded. Run build_full_notebook() to create the output.")
