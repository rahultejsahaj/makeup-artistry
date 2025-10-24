import os
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
import face_recognition
import base64

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
        return None, None, "Please upload an image."
    
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
        return image, makeup_image, message
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None, f"Error processing image: {str(e)}"

def create_comparison_slider(original_image, makeup_image):
    """Create an HTML-based image comparison slider"""
    if original_image is None or makeup_image is None:
        return None
    
    # Convert images to base64
    def image_to_base64(img):
        if hasattr(img, 'convert'):
            img = img.convert('RGB')
        else:
            img = Image.fromarray(img).convert('RGB')
        
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    original_b64 = image_to_base64(original_image)
    makeup_b64 = image_to_base64(makeup_image)
    
    # Create HTML for the comparison slider
    html_content = f"""
    <div style="position: relative; width: 100%; height: 400px; overflow: hidden; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <img src="{original_b64}" style="width: 100%; height: 100%; object-fit: cover; position: absolute; top: 0; left: 0;" alt="Original">
        <div style="position: absolute; top: 0; left: 0; width: 50%; height: 100%; overflow: hidden; clip-path: inset(0 50% 0 0);">
            <img src="{makeup_b64}" style="width: 200%; height: 100%; object-fit: cover; position: absolute; top: 0; left: -100%;" alt="With Makeup">
        </div>
        <div style="position: absolute; top: 0; left: 50%; width: 2px; height: 100%; background: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.5); transform: translateX(-50%); cursor: ew-resize;" id="slider"></div>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(255,255,255,0.8); border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-size: 20px; cursor: ew-resize;" id="handle">↔</div>
    </div>
    <script>
        const slider = document.getElementById('slider');
        const handle = document.getElementById('handle');
        const container = slider.parentElement;
        const overlay = container.children[1];
        
        let isDragging = false;
        
        function updateSlider(x) {{
            const rect = container.getBoundingClientRect();
            const percentage = Math.max(0, Math.min(100, ((x - rect.left) / rect.width) * 100));
            slider.style.left = percentage + '%';
            handle.style.left = percentage + '%';
            overlay.style.clipPath = `inset(0 ${{100 - percentage}}% 0 0)`;
        }}
        
        slider.addEventListener('mousedown', (e) => {{
            isDragging = true;
            e.preventDefault();
        }});
        
        handle.addEventListener('mousedown', (e) => {{
            isDragging = true;
            e.preventDefault();
        }});
        
        document.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                updateSlider(e.clientX);
            }}
        }});
        
        document.addEventListener('mouseup', () => {{
            isDragging = false;
        }});
        
        container.addEventListener('click', (e) => {{
            updateSlider(e.clientX);
        }});
    </script>
    """
    
    return html_content

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="RegalRadiance - AI Face Makeup Studio", theme=gr.themes.Soft()) as demo:
        # Header with logo
        with gr.Row():
            gr.HTML("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="color: #D4AF37; font-family: serif; margin: 0;">
                    <span style="font-size: 2.5em;">👑</span><br>
                    <span style="font-size: 2em;">RegalRadiance</span><br>
                    <span style="font-size: 1.2em; color: #B8860B;">RADIANCE</span>
                </h1>
                <p style="color: #666; font-size: 1.1em; margin-top: 10px;">
                    AI-Powered Face Makeup Studio
                </p>
            </div>
            """)
        
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
                gr.Markdown("### Before & After Comparison")
                
                # Image comparison slider
                comparison_slider = gr.HTML(
                    label="Before/After Comparison Slider",
                    value="<div style='text-align: center; padding: 20px; color: #666;'>Upload an image and apply makeup to see the comparison slider</div>"
                )
                
                # Individual result images
                with gr.Row():
                    with gr.Column():
                        original_output = gr.Image(
                            label="Original",
                            height=200,
                            visible=False
                        )
                    with gr.Column():
                        makeup_output = gr.Image(
                            label="With Makeup",
                            height=200,
                            visible=False
                        )
        
        # Event handlers
        def process_with_comparison(input_image, makeup_style):
            original, makeup, message = process_image(input_image, makeup_style)
            if original is not None and makeup is not None:
                comparison_html = create_comparison_slider(original, makeup)
                return comparison_html, original, makeup, message
            else:
                return None, None, None, message
        
        process_btn.click(
            fn=process_with_comparison,
            inputs=[input_image, makeup_style],
            outputs=[comparison_slider, original_output, makeup_output, status_text]
        )
        
        gr.Markdown("""
        ### How to use:
        1. Upload an image containing faces
        2. Select a makeup style from the dropdown
        3. Click "Apply Makeup" to process
        4. Use the comparison slider to see before/after effects
        5. Drag the slider handle to reveal the transformation
        
        ### Features:
        - **Interactive Comparison Slider**: Drag to see before/after effects
        - **AI Face Detection**: Automatically detects facial features
        - **5 Makeup Styles**: Choose from different looks
        - **Real-time Processing**: See results instantly
        - **Full Image Preservation**: No cropping or quality loss
        
        ### Note:
        - The application uses AI to detect faces and apply makeup
        - Processing may take a few seconds depending on your hardware
        - Make sure the image contains clear, visible faces
        - Use the slider to compare original vs. makeup-applied images
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
