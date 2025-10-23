from flask import Flask, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename

# Import the run_nst function we added to NST.py. Use a robust dynamic import
# so running `python server\app.py` (script mode) works regardless of package layout.
try:
    # Prefer a normal import if the package context is set up.
    from NST import run_nst
except Exception:
    import importlib.util
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("NST", os.path.join(PROJECT_ROOT, 'NST.py'))
    NST = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(NST)
    run_nst = NST.run_nst

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
CONTENT_PATH = os.path.join(PROJECT_ROOT, 'content.png')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'output.jpg')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)


@app.route('/style', methods=['POST'])
def style_image():
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename('content.png')
        save_path = os.path.join(PROJECT_ROOT, filename)
        file.save(save_path)

        # Call the NST function directly to avoid subprocess issues.
        try:
            run_nst(content_path=save_path, style_path=os.path.join(PROJECT_ROOT, 'style.png'), output_path=OUTPUT_PATH)
        except Exception as e:
            return jsonify({'error': 'NST failed', 'detail': str(e)}), 500

        if not os.path.exists(OUTPUT_PATH):
            return jsonify({'error': 'output not produced'}), 500

        return send_file(OUTPUT_PATH, mimetype='image/jpeg')

    return jsonify({'error': 'invalid file type'}), 400


@app.route('/style.png', methods=['GET'])
def style_png():
    """Serve the fixed style image from the project root so the frontend can fetch /style.png."""
    style_file = os.path.join(PROJECT_ROOT, 'style.png')
    if os.path.exists(style_file):
        return send_file(style_file, mimetype='image/png')
    return jsonify({'error': 'style image not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
