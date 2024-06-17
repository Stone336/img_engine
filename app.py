from flask import Flask, request, render_template, send_file, jsonify
from model import load_model, generate_image
from io import BytesIO
import logging

app = Flask(__name__)

# Cấu hình log
logging.basicConfig(level=logging.DEBUG)

# Tải mô hình Stable Diffusion từ Hugging Face
model_id = "runwayml/stable-diffusion-v1-5"
pipeline = load_model(model_id)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    try:
        prompt = request.form['prompt']
        app.logger.debug(f"Received prompt: {prompt}")

        # Sinh ảnh từ văn bản
        image = generate_image(pipeline, prompt)

        # Lưu ảnh vào bộ nhớ đệm
        img_io = BytesIO()
        image.save(img_io, 'JPEG')
        img_io.seek(0)

        # Trả về ảnh dưới dạng response
        app.logger.debug("Image generated successfully")
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        app.logger.error(f"Error generating image: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
