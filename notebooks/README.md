# Notebook Wrappers

The project is script-first so it works the same in every Kaggle notebook. In a notebook cell, run the commands from `docs/KAGGLE_RUNBOOK.md` with `!`.

Example:

```python
!cp -r /kaggle/input/<your-cmar-project-dataset>/CMAR /kaggle/working/cmar
%cd /kaggle/working/cmar
!pip install -r requirements.txt
!python scripts/00_environment_check.py --load-models --output /kaggle/working/cmar_environment.json
```

This avoids maintaining separate `.ipynb` files that can drift from the tested scripts.
