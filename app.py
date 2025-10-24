import os
import sys
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
import face_recognition
from torchvision.transforms.functional import normalize

# Add CodeFormer to path
sys.path.append('/Users/ramkotagiri/FaceEnhancementAndMakeup/CodeFormer')
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.archs.codeformer_arch import CodeFormer

# Model configuration
pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

# Global variables for model
device = None
net = None
face_helper = None

def initialize_model():
    """Initialize the CodeFormer model and face helper"""
    global device, net, face_helper
    
    device = get_device()
    
    # Set up CodeFormer restorer
    net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                     connect_list=['32', '64', '128', '256']).to(device)
    
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                   model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()
    
    # Set up FaceRestoreHelper
    face_helper = FaceRestoreHelper(
        upscale=2,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device)

def apply_makeup(image, makeup_style):
    """Apply makeup to the image based on the selected style"""
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)
    
    pil_image = Image.fromarray(image)
    for face_landmarks in face_landmarks_list:
        d = ImageDraw.Draw(pil_image, 'RGBA')
        
        if makeup_style == 1:  # Deep Gray Eyebrows, Red Lip, Gray Eyes, Black Eyeliner
            # DEEP GRAY EB
            d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90))
            d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90))
            d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90), width=1)
            d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90), width=1)
            
            # RED LP
            d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
            d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
            d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=2)
            d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=2)
            
            # GRAY ES
            d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
            d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))
            
            # BLACK EL
            d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 80), width=4)
            d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 80), width=4)
            
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
    
    return np.array(pil_image)

def enhance_face_with_codeformer(image, w=0.7, enhancement_level=2):
    """Enhance face using CodeFormer"""
    global net, face_helper
    
    if net is None or face_helper is None:
        return image, "Model not initialized. Please restart the application."
    
    try:
        opencv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Process with face helper
        face_helper.read_image(opencv_image)
        num_det_faces = face_helper.get_face_landmarks_5(resize=640, eye_dist_threshold=5)
        
        if num_det_faces == 0:
            return image, "No faces detected in the image."
        
        # Align and warp each face
        face_helper.align_warp_face()
        
        # Face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # Prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
            
            try:
                with torch.no_grad():
                    output = net(cropped_face_t, w=w, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'Failed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
            
            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)
        
        # Paste back the restored faces
        face_helper.get_inverse_affine(None)
        restored_img = face_helper.paste_faces_to_input_image()
        
        if enhancement_level == 2:  # High enhancement
            brightness = 4
            contrast = 1.2
            restored_img = cv2.addWeighted(restored_img, contrast, np.zeros(restored_img.shape, restored_img.dtype), 0, brightness)
        
        # Convert back to RGB for display
        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        
        return restored_img, f"Successfully enhanced {num_det_faces} face(s)."
        
    except Exception as e:
        return image, f"Error during enhancement: {str(e)}"

def process_image(input_image, makeup_style, enhancement_level, enhancement_weight):
    """Main processing function"""
    if input_image is None:
        return None, None, "Please upload an image."
    
    try:
        # Convert PIL image to numpy array if needed
        if hasattr(input_image, 'convert'):
            image = np.array(input_image.convert('RGB'))
        else:
            image = input_image
        
        # Apply makeup
        makeup_image = apply_makeup(image, makeup_style)
        
        # Apply enhancement
        enhanced_image, message = enhance_face_with_codeformer(makeup_image, w=enhancement_weight, enhancement_level=enhancement_level)
        
        return makeup_image, enhanced_image, message
        
    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"

# Create Gradio interface
def create_interface():
    # Initialize model
    print("Initializing model...")
    initialize_model()
    print("Model initialized successfully!")
    
    with gr.Blocks(title="AI Face Enhancement & Makeup Application", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎭 AI Face Enhancement & Makeup Application")
        gr.Markdown("Upload an image and apply different makeup styles with AI-powered face enhancement using CodeFormer.")
        
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
                
                enhancement_level = gr.Dropdown(
                    choices=[("Low Enhancement", 1), ("High Enhancement", 2)],
                    label="Enhancement Level",
                    value=2
                )
                
                enhancement_weight = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Enhancement Weight"
                )
                
                process_btn = gr.Button("🎨 Apply Makeup & Enhance", variant="primary")
                
                status_text = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column():
                gr.Markdown("### Results")
                
                makeup_output = gr.Image(
                    label="With Makeup",
                    height=400
                )
                
                enhanced_output = gr.Image(
                    label="Enhanced Result",
                    height=400
                )
        
        # Event handlers
        process_btn.click(
            fn=process_image,
            inputs=[input_image, makeup_style, enhancement_level, enhancement_weight],
            outputs=[makeup_output, enhanced_output, status_text]
        )
        
        gr.Markdown("""
        ### How to use:
        1. Upload an image containing faces
        2. Select a makeup style from the dropdown
        3. Choose the enhancement level and weight
        4. Click "Apply Makeup & Enhance" to process
        5. View the results with makeup and enhancement applied
        
        ### Note:
        - The application uses AI to detect faces and apply makeup
        - CodeFormer is used for face enhancement and restoration
        - Processing may take a few seconds depending on your hardware
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
