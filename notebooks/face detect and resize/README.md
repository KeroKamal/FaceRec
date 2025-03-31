# Face Detection and Resizing Script

This script processes all images in a given input folder by detecting faces using the MTCNN face detector, cropping the detected face regions, resizing them to a specified size, and saving the resized face images in an output folder.

## How It Works

1. **Environment Setup:**  
   The script imports the required libraries:
   - **PIL (Python Imaging Library):** For image processing.
   - **os:** For file and directory operations.
   - **torch:** To check and use CUDA if available.
   - **facenet_pytorch.MTCNN:** For face detection.

2. **Device Configuration:**  
   The script sets the device to GPU (CUDA) if available, otherwise it defaults to CPU.

3. **MTCNN Initialization:**  
   The MTCNN model is initialized with `keep_all=False` so that only one face is detected per image if multiple faces are present.

4. **Function to Process Images:**  
   The function `detect_and_resize_faces_in_folder`:
   - Checks if the output folder exists and creates it if necessary.
   - Iterates through each image in the input folder.
   - Opens each image and converts it to RGB.
   - Uses MTCNN to detect faces in the image.
   - If faces are detected:
     - Crops the image to the face region.
     - Resizes the cropped face to the specified dimensions (default is 160x160).
     - Saves the resized face image in the output folder with a filename indicating the source image and the face index.
   - If no faces are detected in an image, it prints a message indicating so.
   - Finally, it prints a summary message after processing all images.

5. **Execution:**  
   The script defines the input folder (where the original images are located) and the output folder (where the processed images will be saved), then calls the function to process the images.

## Requirements

- Python 3.x
- Pillow
- PyTorch
- facenet-pytorch

## Setup and Usage

1. **Install Dependencies:**  
   Install the required libraries using pip:
   ```bash
   pip install pillow torch facenet-pytorch

2. **Prepare the Input Folder:** 

  Place your images in the input folder. Update the input_folder variable in the script with the path to your images.

3. **Set the Output Folder:** 

  Specify the path for the output folder in the script using the output_folder variable. Processed images will be saved here.

4. **Output:**

  The script will process each image, detect and resize the face, and save the processed images in the output folder. A message will be printed for each image where no faces are detected, and a final summary message will be displayed after processing all images.

## Conclusion

This script is useful for preprocessing face images before further tasks like face recognition or analysis.