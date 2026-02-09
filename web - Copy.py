import io
import base64
import tempfile
import logging
import os

import numpy as np
import nrrd
from flask import Flask, render_template, request, send_file, session
from PIL import Image
from waitress import serve

# Import the smart inference function
from smart_infer import smart_decay_infer

ckpt_path = '3.ckpt'
host = '127.0.0.1'
port = 5000

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = 'your_secret_key'  # replace with secure key for production

ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'nrrd'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def read_uploaded_image(file_storage) -> Image.Image:
    """
    Return a PIL.Image from the uploaded FileStorage. Supports standard images and .nrrd files.
    """
    filename = getattr(file_storage, "filename", "")
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if ext == 'nrrd':
        # read bytes to a temp file because nrrd.read expects a filename
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file_storage.read())
            tmp.flush()
            tmp_name = tmp.name
        pixels, _ = nrrd.read(tmp_name)
        # Normalize/scaling to 0-255 then convert to uint8
        pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min()) * 255
        pixels = pixels.squeeze().astype(np.uint8)
        # ensure HxW xC ordering for PIL
        if pixels.ndim == 2:
            img = Image.fromarray(pixels)
        else:
            # swap axes if channels-first
            if pixels.shape[0] in (1, 3) and pixels.shape[0] != pixels.shape[-1]:
                pixels = np.moveaxis(pixels, 0, -1)
            img = Image.fromarray(pixels)
        return img
    else:
        # ensure the file pointer is at start
        try:
            file_storage.stream.seek(0)
        except Exception:
            pass
        img = Image.open(file_storage.stream if hasattr(file_storage, "stream") else file_storage)
        return img.convert('RGB')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return render_template('index.html', message='No file part')
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', message='No selected file')
            if not allowed_file(file.filename):
                return render_template('index.html', message='Unsupported file type')

            # Read image (PIL.Image)
            img = read_uploaded_image(file)

            # Original image base64 (for inline display if template expects it)
            output = io.BytesIO()
            img.save(output, format='JPEG')
            image_data_bytes = output.getvalue()
            image_data_orig = base64.b64encode(image_data_bytes).decode('utf-8')

            # Convert to numpy for model inference
            image_data_np = np.array(img)

            # Use smart inference for decay detection
            image_data_pred_np, decay_percent, advice_text = smart_decay_infer(image_data_np, ckpt_path)

            # Convert prediction image to base64 for template
            img_pred = Image.fromarray(image_data_pred_np.astype(np.uint8))
            output_pred = io.BytesIO()
            img_pred.save(output_pred, format='JPEG')
            image_data_pred = base64.b64encode(output_pred.getvalue()).decode('utf-8')

            # store results in session for download route
            session['decay_percent'] = float(decay_percent)
            session['advice_text'] = str(advice_text)

            return render_template(
                'index.html',
                message='File uploaded successfully',
                image_data_orig=image_data_orig,
                image_data=image_data_pred,
                decay_percent=f"{decay_percent:.2f}",
                advice_text=advice_text
            )

        # GET
        return render_template('index.html')

    except Exception as e:
        log.exception("Error in upload_file")
        return render_template('index.html', message=f"Error: {str(e)}")


@app.route('/download_report')
def download_report():
    decay_percent = session.get('decay_percent', 0.0)
    advice_text = session.get('advice_text', 'No advice available')
    doctor_path = os.path.join(app.static_folder, 'doctor.webp')  # put doctor.jpg into static/
    buf = io.BytesIO()

    try:
        # Preferred: use reportlab if available for proper PDF layout
        import importlib
        try:
            pagesizes = importlib.import_module('reportlab.lib.pagesizes')
            pdfgen_canvas = importlib.import_module('reportlab.pdfgen.canvas')
            lib_utils = importlib.import_module('reportlab.lib.utils')
            platypus = importlib.import_module('reportlab.platypus')
            styles_mod = importlib.import_module('reportlab.lib.styles')
        except Exception:
            # If reportlab isn't available, raise to trigger the Pillow fallback below
            raise ImportError("reportlab not available")

        A4 = pagesizes.A4
        canvas = pdfgen_canvas
        ImageReader = lib_utils.ImageReader
        Paragraph = platypus.Paragraph
        getSampleStyleSheet = styles_mod.getSampleStyleSheet

        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4

        # draw small doctor photo top-left
        if os.path.isfile(doctor_path):
            img_reader = ImageReader(doctor_path)
            img_w, img_h = 80, 80
            c.drawImage(img_reader, 40, height - 40 - img_h, width=img_w, height=img_h, preserveAspectRatio=True)

        # header + values
        text_x = 140
        text_y = height - 60
        c.setFont("Helvetica-Bold", 16)
        c.drawString(text_x, text_y, "DentalAI Report")
        c.setFont("Helvetica", 12)
        c.drawString(text_x, text_y - 26, f"Decay Area: {float(decay_percent):.2f}%")

        # wrapped advice
        styles = getSampleStyleSheet()
        p = Paragraph(f"<b>Advice:</b> {advice_text}", styles["Normal"])
        avail_w = width - text_x - 40
        p.wrapOn(c, avail_w, height)
        p.drawOn(c, text_x, text_y - 100)

        c.showPage()
        c.save()
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name="dental_report.pdf", mimetype="application/pdf")

    except Exception:
        # Fallback: create a single-page PDF using Pillow (no extra deps required)
        from PIL import Image, ImageDraw, ImageFont
        import textwrap

        W, H = 595, 842  # approx A4 at 72dpi
        page = Image.new('RGB', (W, H), 'white')
        draw = ImageDraw.Draw(page)

        # paste doctor photo if exists
        if os.path.isfile(doctor_path):
            doc_img = Image.open(doctor_path).convert('RGB')
            doc_img.thumbnail((80, 80))
            page.paste(doc_img, (40, 40))

        x, y = 140, 60
        try:
            font_b = ImageFont.truetype("arial.ttf", 16)
            font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font_b = None
            font = None

        draw.text((x, y), "DentalAI Report", fill='black', font=font_b)
        y += 30
        draw.text((x, y), f"Decay Area: {float(decay_percent):.2f}%", fill='black', font=font)
        y += 26

        for line in textwrap.wrap(advice_text, width=60):
            draw.text((x, y), line, fill='black', font=font)
            y += 18

        out = io.BytesIO()
        page.save(out, format='PDF')
        out.seek(0)
        return send_file(out, as_attachment=True, download_name="dental_report.pdf", mimetype="application/pdf")


if __name__ == '__main__':
    log.info("Starting server on %s:%s", host, port)
    print(f"🦷 DentalAI Server running on http://{host}:{port}")
    print("📁 Upload dental X-rays to detect decay areas!")
    serve(app, host=host, port=port)