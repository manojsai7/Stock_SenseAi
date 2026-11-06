# Reproducing the Python tests

Use Python 3.13 on Windows from the repository root. Runtime deployment dependencies remain in `backend/requirements.txt`; the test environment pins runtime and research/test dependencies separately in `requirements-test.txt`, including matplotlib, seaborn and tabulate.

```powershell
python -m venv .venv-test
.\.venv-test\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv-test\Scripts\python.exe -m pip check
.\.venv-test\Scripts\python.exe -m pytest -q --disable-warnings
git diff --check 204dcd5
```

TensorFlow installation and its first import can take several minutes. Synthetic model tests exercise checkpoint export and inference; they do not establish market forecast performance. The recovered EXP-05 CSV uses LF line endings with its normalized file checksum recorded separately from the original Git checksum.

Install requirements-test-lock.txt for the complete resolved fresh verification environment.
