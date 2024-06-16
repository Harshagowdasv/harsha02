from flask import Flask, request, send_file, render_template, jsonify
import cv2
import numpy as np
import moviepy.editor as mpy
import os
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

logging.basicConfig(level=logging.DEBUG)

def create_fade_in_video(image_path, output_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not loaded correctly")

    height, width, _ = image.shape
    logging.debug(f'Image loaded with shape: {height}x{width}')

    # Initialize a list to store the frames
    frames = []
    num_frames = 120  # 5 seconds * 24 fps
    for i in range(num_frames):
        alpha = i / float(num_frames - 1)
        # Create a fade-in effect by adjusting alpha
        frame = cv2.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), 0, 0)
        frames.append(frame)

    # Verify frame generation by checking a few frames
    logging.debug(f'First frame mean value: {np.mean(frames[0])}')
    logging.debug(f'Last frame mean value: {np.mean(frames[-1])}')

    # Convert the frames to a video using MoviePy
    clip = mpy.ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=24)
    clip.write_videofile(output_path, codec='libx264', fps=24)
    logging.debug(f'Video written to: {output_path}')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logging.debug(f'File saved to: {file_path}')

        animation = request.form['animation']
        output_path = os.path.join(PROCESSED_FOLDER, 'output.mp4')

        if animation == 'fade_in':
            create_fade_in_video(file_path, output_path)

        return send_file(output_path, mimetype='video/mp4', as_attachment=True, attachment_filename='output.mp4')

if __name__ == '__main__':
    app.run(debug=True)
