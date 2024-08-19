import os
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Load the trained model
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(num_classes=2)
model.load_state_dict(torch.load("nightowls_model.pth", map_location=device))
model.to(device)
model.eval()


def transform_image(image_bytes):
    my_transforms = T.Compose([T.ToTensor()])
    image = Image.open(image_bytes).convert("RGB")
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    tensor = tensor.to(device)
    with torch.no_grad():
        outputs = model(tensor)
    return outputs


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get prediction
            with open(file_path, 'rb') as f:
                outputs = get_prediction(f)
            num_people = sum([1 for score in outputs[0]['scores'].cpu().numpy() if score > 0.5])

            # Visualize results
            image = Image.open(file_path).convert("RGB")
            img = np.array(image)

            plt.figure(figsize=(12, 8))
            plt.imshow(img)

            boxes = outputs[0]['boxes'].cpu().numpy()
            scores = outputs[0]['scores'].cpu().numpy()

            for box, score in zip(boxes, scores):
                if score > 0.5:
                    x0, y0, x1, y1 = box
                    plt.gca().add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color='red', linewidth=3))
                    plt.text(x0, y0, f'Score: {score:.2f}', color='red', fontsize=12,
                             bbox=dict(facecolor='yellow', alpha=0.5))

            plt.axis('off')
            plt_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{filename}')
            plt.savefig(plt_path)
            plt.close()

            return render_template('result.html', num_people=num_people, result_image=plt_path)

    return render_template('upload.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)
