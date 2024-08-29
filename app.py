from flask import Flask, request, send_from_directory, jsonify, url_for
from flask_cors import CORS
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import uuid

from model import U2NET
from torch.autograd import Variable
from skimage import io, transform
from PIL import Image

app = Flask(__name__)
CORS(app)

# الدليل الحالي
currentDir = os.path.dirname(os.path.abspath(__file__))

# تحميل النموذج
model_name = 'u2net'
model_dir = os.path.join(currentDir, 'saved_models', model_name, model_name + '.pth')
net = U2NET(3, 1)
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    file = request.files.get('file')  # استخدم get لتجنب KeyError
    if file:
        filename = str(uuid.uuid4()) + '.jpg'
        input_path = os.path.join(currentDir, 'static/inputs/', filename)
        output_path = os.path.join(currentDir, 'static/results/', filename.replace('.jpg', '.png'))
        file.save(input_path)

        status = removeBg(input_path, output_path)
        if status == "---Success---":
            download_url = url_for('download_file', filename=filename.replace('.jpg', '.png'), _external=True)
            return jsonify({'message': 'Image processed successfully', 'download_url': download_url})
        else:
            return jsonify({'error': status}), 500
    return jsonify({'error': 'No file provided'}), 400

def save_output(image_name, output_name, pred, d_dir, type):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np * 255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]))
    pb_np = np.array(imo)
    if type == 'image':
        # Make and apply mask
        mask = pb_np[:, :, 0]
        mask = np.expand_dims(mask, axis=2)
        imo = np.concatenate((image, mask), axis=2)
        imo = Image.fromarray(imo, 'RGBA')
    imo.save(os.path.join(d_dir, output_name))

def removeBg(imagePath, outputPath):
    try:
        img = io.imread(imagePath)
        img = transform.resize(img, (320, 320), mode='constant')
        tmpImg = np.zeros((img.shape[0], img.shape[1], 3))
        tmpImg[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        tmpImg[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpImg = np.expand_dims(tmpImg, 0)
        image = torch.from_numpy(tmpImg).type(torch.FloatTensor)
        image = Variable(image)
        if torch.cuda.is_available():
            image = image.cuda()

        d1, d2, d3, d4, d5, d6, d7 = net(image)
        pred = d1[:, 0, :, :].squeeze()
        save_output(imagePath, outputPath, pred, currentDir + '/static/results/', 'image')
        return "---Success---"
    except Exception as e:
        return str(e)

@app.route('/static/results/<filename>')
def download_file(filename):
    return send_from_directory(os.path.join(currentDir, 'static/results/'), filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
