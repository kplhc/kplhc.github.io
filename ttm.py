from flask import Flask, request, send_file, jsonify
from io import BytesIO
from PIL import Image
from transformers import pipeline

app = Flask(__name__)

# 使用预训练的图像生成模型
generator = pipeline('stable-diffusion', model='CompVis/stable-diffusion-v1-4')

def generate_image(prompt):
    images = generator(prompt, num_inference_steps=50, guidance_scale=7.5)
    return images[0]['image']

@app.route('/prompt/<description>', methods=['GET'])
def generate_image_endpoint(description):
    try:
        image = generate_image(description)
        
        # 将图像保存到内存中
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
