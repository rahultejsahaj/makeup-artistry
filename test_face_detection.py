import face_recognition
from PIL import Image
import numpy as np

def test_face_detection(image_path):
    """Test face detection on an image"""
    print(f"Testing face detection on: {image_path}")
    
    # Load image
    image = face_recognition.load_image_file(image_path)
    print(f"Image shape: {image.shape}")
    
    # Find face locations
    face_locations = face_recognition.face_locations(image)
    print(f"Found {len(face_locations)} face location(s)")
    
    # Find face landmarks
    face_landmarks_list = face_recognition.face_landmarks(image)
    print(f"Found {len(face_landmarks_list)} face landmark set(s)")
    
    for i, face_landmarks in enumerate(face_landmarks_list):
        print(f"\nFace {i+1} landmarks:")
        for landmark_name, landmark_points in face_landmarks.items():
            print(f"  {landmark_name}: {len(landmark_points)} points")
    
    return len(face_landmarks_list) > 0

if __name__ == "__main__":
    # Test with a sample image if available
    import os
    if os.path.exists("test_image.jpg"):
        test_face_detection("test_image.jpg")
    else:
        print("No test image found. Please add a test_image.jpg file to test face detection.")
