# Face Recognition Flask API 🎭✨

A Flask-based application that performs face recognition by processing an uploaded image. It leverages **MTCNN** for face detection and **InceptionResnetV1** for generating face embeddings. The embeddings are compared against a pre-saved dataset (using pickle) to identify known individuals. Additionally, recognized names are mapped to student IDs using an Excel file, and the results are returned in JSON format.

---

## Table of Contents 📚🔎

- [Overview 🔍📸](#overview-📸)
- [Features 🚀💡](#features-💡)
- [Requirements 🛠️📦](#requirements-📦)
- [Installation & Setup 🔧📥](#installation--setup-📥)
- [Usage 🚀🖥️](#usage-🖥️)
- [Example Request & Response 🎉🔄](#example-request--response-🔄)
- [License 📄⚖️](#license-⚖️)

---

## Overview 🔍📸

The application follows these steps:

1. **Image Upload & Conversion:**  
   - Upload an image file.
   - Automatically convert the image to PNG format while handling EXIF orientation.

2. **Face Detection:**  
   - Utilize **MTCNN** to accurately detect faces in the image.

3. **Face Embedding:**  
   - Pre-process the detected faces.
   - Generate embeddings using **InceptionResnetV1**.

4. **Face Comparison:**  
   - Compare the generated embeddings with a pre-saved dataset using a cosine distance metric and a specified threshold.

5. **ID Mapping:**  
   - Map the recognized face names to student IDs using an Excel file.

6. **Response:**  
   - Return a JSON response containing recognized names, corresponding IDs, and a count of any unknown faces.

---

## Features 🚀💡

- **Robust Image Handling:**  
  Automatically converts uploaded images to PNG, ensuring correct orientation via EXIF data.

- **Accurate Face Detection:**  
  Uses **MTCNN** with probability thresholds for high accuracy in detecting faces.

- **Effective Face Recognition:**  
  Employs **InceptionResnetV1** pretrained on VGGFace2 for generating high-quality face embeddings.

- **Seamless Data Integration:**  
  Maps recognized names to student IDs using an Excel file for effortless data management.

- **RESTful API:**  
  Provides a simple endpoint for face recognition, making it easy to integrate with other systems.

---

## Requirements 🛠️📦

- **Python 3.x**
- [Flask](https://palletsprojects.com/p/flask/) 🐍
- [Pillow](https://python-pillow.org/) 🖼️
- [pandas](https://pandas.pydata.org/) 📊
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) 🤖
- [scipy](https://www.scipy.org/) 🔬
- [torch](https://pytorch.org/) 🔥
- [numpy](https://numpy.org/) ➗

*Additional modules such as pickle are part of Python’s standard library.*

---

## Installation & Setup 🔧📥

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/KeroKamal/FaceRec.git
   cd FaceRec
   ```

2. **Set Up a Virtual Environment and Install Dependencies:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start the Flask Application:**

   ```bash
   python src/app.py
   ```

---

## Usage 🚀🖥️

### Endpoint

- **POST** `/recognize_faces`

### Description

- Upload an image file using a form-data request with the key `file`.

### Example using cURL

```bash
curl -X POST http://localhost:5000/recognize_faces -F "file=@path/to/your/image.jpg"
```

---

## Example Request & Response 🎉🔄

### Request

- **Endpoint:** POST `/recognize_faces`
- **Payload:** An image file uploaded as form-data with key `file`.

### Response

```json
[
  {
    "Student Name": "John Doe",
    "ID": "123456"
  },
  {
    "unknown_faces": 1
  }
]
```

---

Happy coding! 😎🎉