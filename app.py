import base64
from logging import debug

import cv2
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import numpy as np
import torch
from iopaint.model_manager import ModelManager
from iopaint.runtime import check_device
from iopaint.schema import InpaintRequest, SDSampler, LDMSampler, HDStrategy
from torch.types import Device

app = Flask(__name__)

# Khởi tạo IOPaint
device = "cpu"
sd_steps = check_device("cpu")
name = 'migan'

model = ModelManager(
    name=name,
    device=torch.device(device),
    disable_nsfw=True,
    sd_cpu_textencoder=True,
)

config = InpaintRequest(
    image="",
    mask="",
    sd_sampler=SDSampler.uni_pc,
    ldm_steps=1,
    ldm_sampler=LDMSampler.plms,
    hd_strategy=HDStrategy.ORIGINAL,
    hd_strategy_crop_margin=32,
    hd_strategy_crop_trigger_size=200,
    hd_strategy_resize_limit=200,
)


@app.route('/process_image', methods=['POST'])
def process_image():
    # Kiểm tra xem có ảnh và box được cung cấp không
    if 'image' not in request.json or 'mask_x' not in request.json or 'mask_y' not in request.json or 'mask_width' not in request.json or 'mask_height' not in request.json:
        return jsonify({'error': 'Image and mask parameters are required'}), 400

    # Lấy chuỗi Base64 của ảnh
    image_base64 = request.json['image']

    # Giải mã chuỗi Base64 thành ảnh
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Lấy các giá trị của bounding box
    mask_x = request.json['mask_x']
    mask_y = request.json['mask_y']
    mask_width = request.json['mask_width']
    mask_height = request.json['mask_height']

    # Tạo mask dựa trên box
    mask = Image.new('L', image.size, 0)  # Tạo mask đen
    mask.paste(255, (mask_x, mask_y, mask_x + mask_width, mask_y + mask_height))  # Box trắng cho mask

    # Chuyển đổi ảnh và mask thành mảng numpy
    image_np = np.array(image)
    mask_np = np.array(mask)

    # Xử lý ảnh với IOPaint
    output_image = model(image_np, mask_np, config)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Chuyển đổi kết quả về định dạng PIL
    output_image_pil = Image.fromarray(output_image)

    # Lưu ảnh đầu ra vào bộ nhớ
    img_byte_arr = io.BytesIO()
    try:
        output_image_pil.save(img_byte_arr, format='JPEG')
    except Exception as e:
        debug(e)
        return send_file(io.BytesIO(image_data), mimetype='image/webp')

    img_byte_arr.seek(0)

    # Trả về ảnh đã xử lý dưới dạng nhị phân
    return send_file(img_byte_arr, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(port=5000)
