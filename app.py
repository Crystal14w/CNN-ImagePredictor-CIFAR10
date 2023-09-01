from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('cifar10_model.h5')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the image from the POST request
        img = request.files['image'].read()
        img = Image.open(io.BytesIO(img)).resize((32, 32))  # Uses the io module
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]

        return render_template('index.html', prediction=predicted_class)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
