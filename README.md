python -m venv .venv
# activate:
# Windows:
#   .\.venv\Scripts\activate
# Linux/macOS:
#   source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
