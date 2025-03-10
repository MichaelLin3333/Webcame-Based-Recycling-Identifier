# Webcame-Based-Recycling-Identifier

Below is a detailed, step-by-step developer guide for building an **AI-Powered Recycling Identifier**. This guide is designed for students with basic Python and HTML skills and assumes a total development period of one month (four 1‑hour sessions). Each session builds on the previous one. References are provided at each step to help you dive deeper.

---

## **Session 1: Environment Setup & Basic Flask Application**

### **1.1. Set Up Your Development Environment**

- **Install Python:**  
  Ensure you have Python 3.8+ installed. You can download it from [python.org](https://www.python.org/downloads/).

- **Create a Virtual Environment:**  
  Open your terminal/command prompt and run:
  ```bash
  python -m venv env
  ```
  Then, activate your virtual environment:
  - **Windows:**  
    ```bash
    env\Scripts\activate
    ```
  - **Mac/Linux:**  
    ```bash
    source env/bin/activate
    ```

- **Install Flask and Other Dependencies:**  
  Install Flask and any additional libraries using pip:
  ```bash
  pip install flask pillow requests
  ```
  *Note:* The `Pillow` library is used for basic image handling.

- **Reference:**  
  - [Flask Quickstart](https://flask.palletsprojects.com/en/2.2.x/quickstart/)  
  - [Python Virtual Environments Tutorial](https://realpython.com/python-virtual-environments-a-primer/)

### **1.2. Create a Basic Flask Application**

- **File Structure:**  
  Create a new folder for your project and inside it, create:
  - `app.py` – Your main Python file.
  - `templates/` – A folder to store your HTML file (e.g., `index.html`).

- **app.py – Basic Setup:**  
  Create a simple Flask app that renders an HTML page:
  ```python
  from flask import Flask, render_template, request, jsonify
  from PIL import Image
  import io

  app = Flask(__name__)

  @app.route('/')
  def index():
      return render_template('index.html')

  # Placeholder for image processing endpoint
  @app.route('/upload', methods=['POST'])
  def upload():
      if 'file' not in request.files:
          return jsonify({'error': 'No file uploaded'}), 400
      file = request.files['file']
      # Convert file to image (future processing can be added here)
      image = Image.open(file.stream)
      # For now, just return a dummy response
      return jsonify({'result': 'Recyclable'}), 200

  if __name__ == '__main__':
      app.run(debug=True)
  ```
- **Reference:**  
  - [Flask Tutorial Video for Beginners](https://www.youtube.com/watch?v=Z1RJmh_OqeA)

---

## **Session 2: Creating the Frontend with Webcam Integration**

### **2.1. Design a Basic HTML Page**

- **templates/index.html:**  
  Create an HTML file that will be served by Flask. This page includes a video element for the webcam and a button to capture an image.
  ```html
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Recycling Identifier</title>
      <style>
          video, canvas {
              width: 100%;
              max-width: 400px;
              margin: 10px auto;
              display: block;
          }
      </style>
  </head>
  <body>
      <h1>AI-Powered Recycling Identifier</h1>
      <video id="video" autoplay></video>
      <button id="captureBtn">Capture Image</button>
      <canvas id="canvas" style="display: none;"></canvas>
      <p id="result"></p>

      <script>
          // Access webcam
          const video = document.getElementById('video');
          const canvas = document.getElementById('canvas');
          const captureBtn = document.getElementById('captureBtn');
          const resultP = document.getElementById('result');

          // Request webcam access
          navigator.mediaDevices.getUserMedia({ video: true })
              .then(stream => {
                  video.srcObject = stream;
              })
              .catch(err => {
                  console.error("Error accessing webcam: ", err);
              });

          // Capture image and send to server
          captureBtn.addEventListener('click', () => {
              const context = canvas.getContext('2d');
              canvas.width = video.videoWidth;
              canvas.height = video.videoHeight;
              context.drawImage(video, 0, 0, canvas.width, canvas.height);

              // Convert canvas image to blob and send to server
              canvas.toBlob(blob => {
                  const formData = new FormData();
                  formData.append('file', blob, 'capture.png');

                  fetch('/upload', {
                      method: 'POST',
                      body: formData
                  })
                  .then(response => response.json())
                  .then(data => {
                      resultP.textContent = "Prediction: " + data.result;
                  })
                  .catch(error => {
                      console.error("Error:", error);
                  });
              }, 'image/png');
          });
      </script>
  </body>
  </html>
  ```
- **Reference:**  
  - [MDN Web Docs: Using the MediaDevices API](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)  
  - [JavaScript Tutorial on Webcam Integration](https://www.youtube.com/watch?v=IqPNSqbmhFk)

### **2.2. Testing the Frontend**

- Run your Flask app:
  ```bash
  python app.py
  ```
- Open your browser at `http://127.0.0.1:5000/` and ensure the webcam feed is visible. Click the "Capture Image" button to verify that an image is captured and sent (even though the backend currently returns a dummy prediction).

---

## **Session 3: Integrating a Pre-trained AI Model for Image Classification**

### **3.1. Choosing an AI Model**

- **Model Selection:**  
  For simplicity, use a pre-trained image classification model available via Hugging Face or a similar repository. For example, you can use a model that distinguishes recyclable objects from non-recyclable ones.  
  *Tip:* If you do not have a custom dataset, you might use a generic image classification model and later customize the thresholds or labels.

- **Using a Pre-trained Model:**  
  You can use the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library for some models, or if your task is computer vision, consider the [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index) or use a simpler TensorFlow/Keras model from [TensorFlow Hub](https://tfhub.dev/).

- **Reference:**  
  - [Hugging Face Image Classification Tutorial](https://huggingface.co/blog/image-classification)  
  - [TensorFlow Hub Image Classifier Example](https://www.tensorflow.org/tutorials/images/classification)

### **3.2. Installing Additional Dependencies**

- **Example (Using TensorFlow/Keras):**  
  If you choose a TensorFlow-based model, install TensorFlow:
  ```bash
  pip install tensorflow
  ```
  *Alternatively*, if you select a PyTorch model, install PyTorch accordingly.

### **3.3. Loading and Using the Model in Flask**

- **Update app.py to Process Images:**  
  For demonstration, here’s a simplified example using TensorFlow:
  ```python
  from flask import Flask, render_template, request, jsonify
  from PIL import Image
  import numpy as np
  import tensorflow as tf
  import io

  app = Flask(__name__)

  # Load a pre-trained model from TensorFlow Hub or a saved model
  model = tf.keras.applications.MobileNetV2(weights='imagenet')  # as an example

  def preprocess_image(image: Image.Image):
      # Resize image to 224x224 and convert to array
      image = image.resize((224, 224))
      img_array = np.array(image)
      # Expand dimensions and preprocess for MobileNetV2
      img_array = np.expand_dims(img_array, axis=0)
      img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
      return img_array

  def classify_image(image: Image.Image):
      processed_image = preprocess_image(image)
      predictions = model.predict(processed_image)
      # Decode predictions (this example returns the top prediction from ImageNet)
      decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0]
      # For recycling, you would customize the labels and logic accordingly.
      label = decoded[1]
      confidence = float(decoded[2])
      return label, confidence

  @app.route('/')
  def index():
      return render_template('index.html')

  @app.route('/upload', methods=['POST'])
  def upload():
      if 'file' not in request.files:
          return jsonify({'error': 'No file uploaded'}), 400
      file = request.files['file']
      image = Image.open(file.stream).convert('RGB')
      label, confidence = classify_image(image)
      # Here, you could map certain labels to "Recyclable" or "Not Recyclable"
      result = f"{label} ({confidence*100:.1f}%)"
      return jsonify({'result': result}), 200

  if __name__ == '__main__':
      app.run(debug=True)
  ```
  *Note:* This example uses MobileNetV2, which is trained on ImageNet. For a recycling identifier, you may eventually need to fine-tune a model or create a mapping that interprets certain labels (e.g., "plastic bottle," "can") as recyclable.

- **Reference:**  
  - [TensorFlow Keras Applications Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2)  
  - [Image Preprocessing with Pillow and TensorFlow](https://www.youtube.com/watch?v=0Lt9w-BxKFQ)

---

## **Session 4: Integration, Testing, and Final Touches**

### **4.1. Integrating Frontend and Backend**

- **Ensure Your HTML Page Sends Data:**  
  Verify that your JavaScript in `index.html` correctly sends the captured image to the `/upload` endpoint and that the backend processes the image and returns a prediction.
- **Display Results:**  
  Once the Flask endpoint returns the prediction, update the HTML to display the prediction text dynamically.

### **4.2. Testing and Debugging**

- **Local Testing:**  
  - Run your Flask server (`python app.py`).
  - Test capturing images using your webcam and check the JSON response.
- **Debugging Tips:**  
  - Use your browser’s developer console (F12) to check for JavaScript errors.
  - Use `print()` statements or Python’s logging module to debug backend issues.
- **Reference:**  
  - [Chrome DevTools Overview](https://developers.google.com/web/tools/chrome-devtools)  
  - [Debugging Flask Applications](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)

### **4.3. Customization and Final Presentation**

- **Customize the AI Model:**  
  - Once the basic integration is working, you may refine the classification. For instance, create a mapping function that interprets ImageNet labels as “Recyclable” or “Non-Recyclable.”
  - Consider adding a section on your webpage explaining the model, the data it was trained on, and how the prediction relates to recycling.
- **Webpage Design Enhancements:**  
  - Improve the styling using CSS frameworks like Bootstrap.
  - Add informational sections about Earth Day and sustainability practices.
- **Final Testing:**  
  - Test on different devices if possible.
  - Gather feedback from peers before the exhibition.

---

By following these detailed steps, your club members can create a fully functional AI-powered recycling identifier while learning key concepts in web development, image processing, and AI model integration. This project is not only a valuable learning experience but also serves a meaningful purpose for Earth Day. Happy coding!
