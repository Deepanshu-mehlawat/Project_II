from flask import Flask, render_template, request, jsonify, Response
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.applications import ResNet50
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os
import json


app = Flask(__name__)

app.config['STATIC_FOLDER'] = 'static'
app.config['STATIC_URL'] = '/static'


embedding_size = 128 #dimensionality of the word embedding
max_len = 40
vocab_size=8254

def capture_frames_laptop():
    camera_laptop = cv2.VideoCapture(0)  # Use 0 for the first camera
    while True:
        success, frame = camera_laptop.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed_laptop')
def video_feed_laptop():
    return Response(capture_frames_laptop(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Load the image captioning model
model = load_model('model.h5')
resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')

def preprocessing(img_path):
    im = image.load_img(img_path, target_size=(224, 224, 3))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    return im

def get_encoding(model, img):
    image = preprocessing(img)
    pred = model.predict(image).reshape(2048)
    return pred

def predict_captions(image, model, max_len, word_2_indices, indices_2_word):
    start_word = ["<start>"]
    while True:
        par_caps = [word_2_indices[i] for i in start_word]
        par_caps = pad_sequences([par_caps], maxlen=max_len, padding='post')
        preds = model.predict([np.array([image]), np.array(par_caps)])
        word_pred = indices_2_word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])

with open('word_2_indices.json', 'r') as f:
    word_2_indices = json.load(f)

word_2_indices = {key: int(value) for key, value in word_2_indices.items()}

with open('indices_2_word.json', 'r') as f:
    indices_2_word = json.load(f)

indices_2_word = {int(key): value for key, value in indices_2_word.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['STATIC_FOLDER'], filename)
        file.save(file_path)
        # Get the image encoding
        image_encoding = get_encoding(resnet, file_path)
        # Generate caption for the image
        predicted_caption = predict_captions(image_encoding, model, max_len, word_2_indices, indices_2_word)
        print(predicted_caption)
        return jsonify({'caption': predicted_caption})

if __name__ == '__main__':
    app.run(debug=True)
