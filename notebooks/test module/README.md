# Face Recognition Using Embeddings Comparison

This project implements a face recognition system that detects faces in an image, extracts their embeddings using a pretrained model, and then compares the embeddings against a stored dataset to identify known individuals. It also maps recognized names to corresponding IDs using an Excel file.

## How It Works

1. **Initialization:**
   - **Face Detection:**  
     The script uses MTCNN to detect faces from input images. MTCNN is configured with custom thresholds and a minimum face size.
   - **Embedding Extraction:**  
     The InceptionResnetV1 model (pretrained on VGGFace2) is used to compute face embeddings from detected faces. The model is set to evaluation mode.

2. **Loading the Dataset:**
   - **Embeddings Dataset:**  
     A dataset containing precomputed face embeddings is loaded from a pickle (`.pkl`) file using the `load_embeddings` function. This dataset is organized as a dictionary with person names as keys and a list of embeddings as values.
   - **ID Mapping:**  
     An Excel file is used to map recognized person names to corresponding IDs. The mapping is loaded into a dictionary via the `load_id_mapping` function.

3. **Face Comparison:**
   - **Normalization:**  
     Both the input face embedding and stored embeddings are normalized.
   - **Cosine Distance Calculation:**  
     The cosine distance between the input embedding and each stored embedding is computed using the `compare_faces` function. The smallest distance is tracked.
   - **Threshold-Based Identification:**  
     If the minimum distance is below a specified threshold, the corresponding name is returned as the identified person; otherwise, the face is considered unknown.

4. **Downloading the Input Image:**
   - An image is downloaded from a Google Drive URL by constructing a direct download link. The image is then decoded using OpenCV and converted to a PIL image for further processing.

5. **Face Recognition Pipeline:**
   - The main function `recognize_faces` orchestrates the process:
     - Loads the embeddings dataset and ID mapping.
     - Downloads the input image from the provided URL.
     - Detects faces in the image using MTCNN and filters out detections with low confidence.
     - For each detected face, extracts its embedding and compares it with the dataset.
     - Prints the recognized person's name along with their ID if a match is found. If no match is found, counts the face as unknown and prints the number of unknown faces.

## Requirements

- Python 3.x
- PyTorch
- facenet-pytorch
- Pandas
- SciPy
- Pillow
- OpenCV
- Requests

## Setup and Usage

1. **Install Dependencies:**
   ```bash
   pip install torch facenet-pytorch pandas scipy pillow opencv-python requests

2. **Prepare Required Files:** 

- Embeddings File: Ensure you have a face_model.pkl file that contains a dictionary of face embeddings.

- Excel File: Have an Excel file (e.g., test.xlsx) with columns Name and ID to map recognized faces to their IDs.

3. **Configure the Script:** 

  Update the model_file, excel_file, and image_url variables in the script with the correct paths and URL.

4. **Output:**

- The script prints the recognized student's name and ID for each detected face.
- It also notifies if any faces are unknown or if no faces are detected in the input image.

## Conclusion

This script provides an end-to-end solution for recognizing faces by comparing extracted embeddings with a pre-stored dataset and mapping them to corresponding IDs.
