# Webcame-Based-Recycling-Identifier

**some GitHub projects for reference: **

- [waste-classification-model](https://github.com/manuelamc14/waste-classification-model)
- [waste-classification-using-YOLOv8](https://github.com/teamsmcorg/Waste-Classification-using-YOLOv8) (You might need to download Jupyter Notebook for this)

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

**Session 3 (Revised): Integrating a Pre-trained YOLOv8 Model with PyTorch**  
*(Replaces the original Session 3 to focus on object detection and recycling-specific classification)*  

---

### **3.1. Choosing an AI Model**  
For recycling tasks, **YOLOv8** (You Only Look Once) is ideal due to its real-time object detection capabilities and support for custom training [[5]][[8]]. Unlike MobileNetV2, YOLOv8 can detect and classify multiple objects in an image (e.g., bottles, cans) and map them to recycling categories. If no pre-trained model exists for your target classes (e.g., glass, plastic), follow the optional training steps below.  

---

### **3.2. Installing Dependencies**  
Install PyTorch and YOLOv8:  
```bash  
pip install torch torchvision  # PyTorch for GPU/CPU support  
pip install ultralytics        # Official YOLOv8 library  
```  
Verify your installation:  
```python  
import torch  
print(torch.__version__)  # Should output a version ≥ 2.0  
```  

---

### **3.3. Loading the Pre-trained YOLOv8 Model**  
1. **Use a Pre-trained Model**:  
   Download a YOLOv8 model trained on recycling datasets (if available). For example:  
   ```python  
   from ultralytics import YOLO  
   model = YOLO("yolov8n.pt")  # Nano-sized model for speed  
   ```  
   If no recycling-specific model exists, use the default YOLOv8 and map its output classes (e.g., "bottle" → "plastic").  

2. **Custom Class Mapping**:  
   Create a dictionary to map YOLOv8’s detected classes to recycling categories:  
   ```python  
   RECYCLING_CATEGORIES = {  
       "bottle": "Plastic",  
       "can": "Metal",  
       "cardboard": "Paper",  
       "trash": "Non-Recyclable"  
   }  
   ```  

---

### **3.4. Modifying `app.py` for YOLOv8 Inference**  
Update the `/upload` route to process images with YOLOv8:  
```python  
from ultralytics import YOLO  
from PIL import Image  
import io  
import numpy as np  

app = Flask(__name__)  
model = YOLO("yolov8n.pt")  # Load pre-trained model  

@app.route('/upload', methods=['POST'])  
def upload():  
    if 'file' not in request.files:  
        return jsonify({'error': 'No file uploaded'}), 400  
    file = request.files['file']  
    image = Image.open(file.stream).convert('RGB')  
    image_np = np.array(image)  

    # Run YOLOv8 inference  
    results = model.predict(image_np, conf=0.5)  # Confidence threshold = 50%  
    detections = results[0].boxes  # Get detection results  

    # Format results for frontend  
    output = []  
    for box in detections:  
        class_id = int(box.cls[0])  
        class_name = model.names[class_id]  
        confidence = float(box.conf[0])  
        output.append({  
            "class": RECYCLING_CATEGORIES.get(class_name, "Unknown"),  
            "confidence": f"{confidence * 100:.1f}%"  
        })  

    return jsonify({'result': output}), 200  
```  

---

### **3.5. (Optional) Training a Custom YOLOv8 Model**  
If no pre-trained model fits your needs, train one using a recycling dataset like **TrashNet** or **TACO Dataset**.  

#### **Step 1: Prepare Your Dataset**  
1. **Download Data**:  
   - [TrashNet Dataset](https://github.com/garythung/trashnet) (2,500+ images of waste).  
   - [TACO Dataset](https://tacodataset.org/) (15,000+ annotated images).  
2. **Organize Data**:  
   ```  
   dataset/  
   ├── images/  
   │   ├── train/  
   │   └── val/  
   └── labels/  
       ├── train/  
       └── val/  
   ```  

#### **Step 2: Configure Training**  
Create a YAML file (`recycling.yaml`) to define classes and data paths:  
```yaml  
train: dataset/images/train  
val: dataset/images/val  
names: ["glass", "paper", "metal", "plastic", "cardboard", "trash"]  
```  

#### **Step 3: Train the Model**  
```python  
from ultralytics import YOLO  

model = YOLO("yolov8n.pt")  # Start from pre-trained weights  
model.train(  
    data="recycling.yaml",  
    epochs=50,  
    batch=16,  
    imgsz=640,  
    device="cpu"  # Use "0" for GPU  
)  
```  

#### **Step 4: Evaluate and Export**  
```python  
metrics = model.val()  # Validate performance  
model.export(format="onnx")  # Export for deployment  
```  

---

### **3.6. Update Session 4 (Integration & Testing)**  
1. **Frontend Adjustments**:  
   Modify `index.html` to display multiple detections:  
   ```javascript  
   .then(data => {  
       const results = data.result;  
       let html = "";  
       results.forEach(item => {  
           html += `<div>${item.class}: ${item.confidence}</div>`;  
       });  
       resultP.innerHTML = html;  
   })  
   ```  
2. **Test with Real Data**:  
   Use sample images of recyclables to verify detection accuracy.  

---

### **References**  
- [YOLOv8 Documentation](https://docs.ultralytics.com/)  
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)  
- [TrashNet Dataset](https://github.com/garythung/trashnet)  
- [TACO Dataset](https://tacodataset.org/)  

This revised session provides a robust pipeline for integrating YOLOv8, with flexibility for customization. Beginners can start with pre-trained models and later explore training on specialized datasets.
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

This project is not only a valuable learning experience but also serves a meaningful purpose for Earth Day. Happy coding!
