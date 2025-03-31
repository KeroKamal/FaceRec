# Face Embedding Extraction and Saving

This project extracts face embeddings from a dataset of face images using the Facenet PyTorch models and saves the resulting embeddings to a pickle file.

## Overview

The code performs the following steps:
1. **Initialize Models:**  
   - Uses MTCNN for face detection.
   - Uses InceptionResnetV1 for face recognition and embedding extraction.  
   The InceptionResnetV1 model is pre-trained on the VGGFace2 dataset and set to evaluation mode.

2. **Extract Embeddings from Dataset:**  
   - The function `extract_embeddings_from_dataset` scans through the dataset directory.
   - For each person (sub-directory), it loads each image.
   - The image is processed by MTCNN to detect faces.
   - For each detected face, the face is passed through the InceptionResnetV1 model to extract its embedding.
   - Embeddings for each person are stored in a dictionary where the keys are the person names and the values are lists of embeddings.

3. **Save Embeddings:**  
   - The embeddings dictionary is saved as a pickle (.pkl) file using the `save_embeddings_to_file` function.

4. **Execution:**  
   - The dataset directory path and the output file path are defined.
   - The embeddings are extracted and then saved to the specified output path.
   - A message is printed indicating the successful save operation.

## Requirements

- Python
- PyTorch
- Facenet-PyTorch
- NumPy
- Pillow
- Pickle (standard library)

## Setup and Usage

1. **Install Dependencies:**  

   Install the required libraries using pip:
   ```bash
   pip install torch facenet-pytorch numpy pillow

2. **Prepare the Dataset:** 

  Organize your dataset as a directory where each sub-directory is named after a person and contains face images of that person.

3. **Configure Paths:** 

- Update the dataset_path variable with the path to your dataset directory.
- Update the output_path variable with the path where you want to save the embeddings.

4. **Output:**

 The embeddings will be saved in a pickle file at the specified output_path and a confirmation message will be printed.

## Conclusion

This script provides a simple yet effective way to create a face recognition model by extracting and saving face embeddings for further use in recognition tasks.




