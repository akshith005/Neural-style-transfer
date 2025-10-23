# Neural Style Transfer - small app

This workspace contains:
- `NST.py` - the existing PyTorch neural style transfer script (uses `content.png`, `style.png` in project root and writes `output.jpg`).
- `server/` - a small Flask server exposing POST /style to accept a content image and run `NST.py`.
- `frontend/` - a Vite + React app to upload a content image and display the stylized result.

Run the server:

1. Create a Python environment and install server requirements:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r server\requirements.txt
```

2. From the project root run:

```powershell
python server\app.py
```

Run the frontend:

1. From `frontend/` install and start dev server (requires Node.js):

```powershell
cd frontend; npm install; npm run dev
```

Use the frontend at http://localhost:3000. The frontend proxies `/style` to the Flask server on port 5000.

Notes:
- The style image `style.png` is expected to be in the project root (already present).
- NST is compute-heavy and may take minutes depending on hardware.
