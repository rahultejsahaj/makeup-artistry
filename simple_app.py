import os
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
import face_recognition

def apply_makeup(image, makeup_style):
    """Apply makeup to the image based on the selected style"""
    # Convert image to RGB if needed
    if image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]  # Remove alpha channel
    elif image.shape[2] == 3:  # RGB
        pass
    else:
        return image, "Invalid image format."
    
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)
    
    if not face_landmarks_list:
        return image, "No faces detected in the image. Try a clearer face photo with good lighting."
    
    print(f"Found {len(face_landmarks_list)} face(s) with landmarks")
    
    # Create PIL image from the full original image
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image, 'RGBA')
    
    for face_landmarks in face_landmarks_list:
        # Check if we have the required landmarks
        required_landmarks = ['left_eyebrow', 'right_eyebrow', 'top_lip', 'bottom_lip', 'left_eye', 'right_eye']
        if not all(landmark in face_landmarks for landmark in required_landmarks):
            continue
        
        if makeup_style == 1:  # Deep Gray Eyebrows, Red Lip, Gray Eyes, Black Eyeliner
            # DEEP GRAY EB - More visible
            d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150))
            d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150))
            d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 200), width=2)
            d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 200), width=2)
            
            # RED LP - More vibrant
            d.polygon(face_landmarks['top_lip'], fill=(200, 0, 0, 180))
            d.polygon(face_landmarks['bottom_lip'], fill=(200, 0, 0, 180))
            d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 255), width=2)
            d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 255), width=2)
            
            # GRAY ES - More visible
            d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 80))
            d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 80))
            
            # BLACK EL - Thicker
            d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 200), width=6)
            d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 200), width=6)
            
        elif makeup_style == 2:  # Brown Eyebrows, Hot Pink Lip, Gray Eyes, Brown Eyeliner
            # BROWN EB
            d.polygon(face_landmarks['left_eyebrow'], fill=(110, 38, 14, 70))
            d.polygon(face_landmarks['right_eyebrow'], fill=(110, 38, 14, 70))
            d.line(face_landmarks['left_eyebrow'], fill=(110, 38, 14, 70), width=1)
            d.line(face_landmarks['right_eyebrow'], fill=(110, 38, 14, 70), width=1)
            
            # HOT PINK LP
            d.polygon(face_landmarks['top_lip'], fill=(199, 21, 133, 128))
            d.polygon(face_landmarks['bottom_lip'], fill=(199, 21, 133, 128))
            d.line(face_landmarks['top_lip'], fill=(199, 21, 133, 128), width=2)
            d.line(face_landmarks['bottom_lip'], fill=(199, 21, 133, 128), width=2)
            
            # GRAY ES
            d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
            d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))
            
            # BROWN EL
            d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(139, 69, 19, 100), width=4)
            d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(139, 69, 19, 100), width=4)
            
        elif makeup_style == 3:  # Deep Gray Eyebrows, Dark Orange Brown Lip, Gray Eyes, Black Eyeliner
            # DEEP GRAY EB
            d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90))
            d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90))
            d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90), width=1)
            d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90), width=1)
            
            # DARK ORANGE BROWN LP
            d.polygon(face_landmarks['top_lip'], fill=(210, 105, 30, 128))
            d.polygon(face_landmarks['bottom_lip'], fill=(210, 105, 30, 128))
            d.line(face_landmarks['top_lip'], fill=(210, 105, 30, 128), width=2)
            d.line(face_landmarks['bottom_lip'], fill=(210, 105, 30, 128), width=2)
            
            # GRAY ES
            d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
            d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))
            
            # BLACK EL
            d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 80), width=4)
            d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 80), width=4)
            
        elif makeup_style == 4:  # Deep Gray Eyebrows, Light Pink Lip, Gray Eyes, Brown Eyeliner
            # DEEP GRAY EB
            d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90))
            d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90))
            d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90), width=1)
            d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90), width=1)
            
            # LIGHT PINK LP
            d.polygon(face_landmarks['top_lip'], fill=(255, 105, 180, 60))
            d.polygon(face_landmarks['bottom_lip'], fill=(255, 105, 180, 60))
            d.line(face_landmarks['top_lip'], fill=(255, 105, 180, 60), width=2)
            d.line(face_landmarks['bottom_lip'], fill=(255, 105, 180, 60), width=2)
            
            # GRAY ES
            d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
            d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))
            
            # BROWN EL
            d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(139, 69, 19, 100), width=4)
            d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(139, 69, 19, 100), width=4)
            
        elif makeup_style == 5:  # Deep Gray Eyebrows, Crimson Lip, Gray Eyes, Black Eyeliner
            # DEEP GRAY EB
            d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90))
            d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90))
            d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90), width=1)
            d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90), width=1)
            
            # CRIMSON LP
            d.polygon(face_landmarks['top_lip'], fill=(220, 20, 1, 60))
            d.polygon(face_landmarks['bottom_lip'], fill=(220, 20, 1, 60))
            d.line(face_landmarks['top_lip'], fill=(220, 20, 1, 60), width=2)
            d.line(face_landmarks['bottom_lip'], fill=(220, 20, 1, 60), width=2)
            
            # GRAY ES
            d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
            d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))
            
            # BLACK EL
            d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 80), width=4)
            d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 80), width=4)
    
    # Convert back to numpy array preserving full image
    result_image = np.array(pil_image)
    return result_image, f"Successfully applied makeup to {len(face_landmarks_list)} face(s)."

def process_image(input_image, makeup_style):
    """Main processing function"""
    if input_image is None:
        return None, "Please upload an image."
    
    try:
        # Convert PIL image to numpy array if needed
        if hasattr(input_image, 'convert'):
            image = np.array(input_image.convert('RGB'))
        else:
            image = input_image
        
        print(f"Processing image with shape: {image.shape}")
        print(f"Makeup style: {makeup_style}")
        
        # Apply makeup
        makeup_image, message = apply_makeup(image, makeup_style)
        
        print(f"Processing result: {message}")
        return makeup_image, message
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, f"Error processing image: {str(e)}"

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="AI Face Makeup Application", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎭 AI Face Makeup Application")
        gr.Markdown("Upload an image and apply different makeup styles using AI-powered face recognition.")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=400
                )
                
                makeup_style = gr.Dropdown(
                    choices=[
                        ("Style 1: Deep Gray Eyebrows, Red Lip, Gray Eyes, Black Eyeliner", 1),
                        ("Style 2: Brown Eyebrows, Hot Pink Lip, Gray Eyes, Brown Eyeliner", 2),
                        ("Style 3: Deep Gray Eyebrows, Dark Orange Brown Lip, Gray Eyes, Black Eyeliner", 3),
                        ("Style 4: Deep Gray Eyebrows, Light Pink Lip, Gray Eyes, Brown Eyeliner", 4),
                        ("Style 5: Deep Gray Eyebrows, Crimson Lip, Gray Eyes, Black Eyeliner", 5),
                    ],
                    label="Makeup Style",
                    value=1
                )
                
                process_btn = gr.Button("🎨 Apply Makeup", variant="primary")
                
                status_text = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column():
                gr.Markdown("### Result")
                
                makeup_output = gr.Image(
                    label="With Makeup",
                    height=400
                )
        
        # Event handlers
        process_btn.click(
            fn=process_image,
            inputs=[input_image, makeup_style],
            outputs=[makeup_output, status_text]
        )
        
        gr.Markdown("""
        ### How to use:
        1. Upload an image containing faces
        2. Select a makeup style from the dropdown
        3. Click "Apply Makeup" to process
        4. View the result with makeup applied
        
        ### Note:
        - The application uses AI to detect faces and apply makeup
        - Processing may take a few seconds depending on your hardware
        - Make sure the image contains clear, visible faces
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        show_error=True
    )
