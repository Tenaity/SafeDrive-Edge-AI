python -m venv .venv
# activate:
# Windows:
#   .\.venv\Scripts\activate
# Linux/macOS:
#   source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py


# Use `uv`

Create `.venv`
```
uv venv --python 3.11
```

Activate the virtual environment
```
uv shell
```

Install dependencies
```
uv pip install -r requirements.txt
```