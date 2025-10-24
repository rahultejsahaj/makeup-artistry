import os
import json
import random

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from config import create_cfg, merge_possible_with_base
from modeling import build_model
from modeling.text_translation import TextTranslationDiffusion
from modeling.translation import TranslationDiffusion

SOURCE_TO_NUM = {"makeup": 0, "non-makeup": 1}

# Beauty features and options
BEAUTY_FEATURES = [
    "Lipstick", "Mascara", "Eyeshadow", "Eyeliner", "Eyebrows", 
    "Foundation", "Makeup", "Concealer", "Contour", "Blush", "Highlighter", "Lipliner"
]

FUNCTION_OPTIONS = [
    "Short Top", "Short Bottom", "Short Top Bottom", "Medium Thin", "Medium Thick",
    "Medium Top", "Medium Bottom", "Medium Top Bottom", "Long Top", "Long Bottom", "Long Top Bottom"
]

FINISH_OPTIONS = ["Satin", "Matte", "Natural", "Dewy", "Glossy"]

COLOR_PALETTE = [
    "#FFFFFF", "#000000", "#8B0000", "#000080", "#8B4513", 
    "#4169E1", "#FFB6C1", "#FFD700", "#32CD32", "#FF6347"
]

# Sample product recommendations
PRODUCT_DATABASE = {
    "Lipstick": [
        {"name": "Magnetic Fuchsia", "shade": "Magnetic fuchsia", "color": "#FF1493"},
        {"name": "Classic Red", "shade": "Classic red", "color": "#DC143C"},
        {"name": "Nude Pink", "shade": "Nude pink", "color": "#FFB6C1"}
    ],
    "Mascara": [
        {"name": "Royal Blue", "shade": "Royal blue", "color": "#4169E1"},
        {"name": "Jet Black", "shade": "Jet black", "color": "#000000"},
        {"name": "Brown", "shade": "Brown", "color": "#8B4513"}
    ],
    "Foundation": [
        {"name": "Linen Beige 11C Matte", "shade": "Linen Beige 11C Matte", "color": "#F5DEB3"},
        {"name": "Linen Beige 11C Natural", "shade": "Linen Beige 11C Natural", "color": "#F5DEB3"},
        {"name": "Ivory 10N", "shade": "Ivory 10N", "color": "#FFF8DC"}
    ]
}


def copy_parameters(
    from_parameters: torch.nn.Parameter, to_parameters: torch.nn.Parameter
):
    to_parameters = list(to_parameters)
    assert len(from_parameters) == len(to_parameters)
    for s_param, param in zip(from_parameters, to_parameters):
        param.data.copy_(s_param.to(param.device).data)


def create_diffusion_model(cfg_path: str, device: str):
    cfg = create_cfg()
    merge_possible_with_base(cfg, cfg_path)
    cfg.MODEL.PRETRAINED = "Justin900/MAD"
    model = build_model(cfg).to(device)
    model.eval()
    diffuser = TranslationDiffusion(cfg, device)
    return diffuser, model


def create_text_diffusion_model(device: str):
    diffuser = TextTranslationDiffusion(
        img_size=512,
        scheduler="ddpm",
        device=device,
        model_path="Justin900/MAD",
        sample_steps=200,
    )

    return diffuser


def domain_translation(
    config_file: str,
    device: str,
    start_from_step: int,
    source_label: str,
    target_label: str,
    image_input: Image.Image,
    mask_input: Image.Image,
):
    diffuser, diffusion_model = create_diffusion_model(config_file, device)

    transform_result = diffuser.domain_translation(
        source_model=diffusion_model,
        target_model=diffusion_model,
        source_image=image_input,
        source_class_label=SOURCE_TO_NUM[source_label],
        target_class_label=SOURCE_TO_NUM[target_label],
        parsing_mask=mask_input,
        use_inversion=False,
        start_from_step=start_from_step,
    )

    del diffuser, diffusion_model
    torch.cuda.empty_cache()
    return Image.fromarray(
        (transform_result[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    )


def reference_translation(
    config_file: str,
    device: str,
    source_label: str,
    target_label: str,
    source_image_input: Image.Image,
    target_image_input: Image.Image,
    source_mask_input: Image.Image,
    target_mask_input: Image.Image,
):
    diffuser, diffusion_model = create_diffusion_model(config_file, device)

    transform_result = diffuser.image_translation(
        source_model=diffusion_model,
        target_model=diffusion_model,
        source_image=source_image_input,
        target_image=target_image_input,
        source_class_label=SOURCE_TO_NUM[source_label],
        target_class_label=SOURCE_TO_NUM[target_label],
        source_parsing_mask=source_mask_input,
        target_parsing_mask=target_mask_input,
        use_morphing=True,
        use_encode_eps=True,
        use_cam=True,
        inpainting=True,
    )
    del diffuser, diffusion_model
    torch.cuda.empty_cache()
    return Image.fromarray(
        (transform_result[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    )


def text_manipulate(
    device: str,
    source_image_with_brush_mask: Image.Image,
    source_mask: Image.Image,
    prompt: str,
    guidance_scale: float,
):
    text_translation = create_text_diffusion_model(device)
    source_image = source_image_with_brush_mask["background"].convert("RGB")
    source_brush_mask = source_image_with_brush_mask["layers"][0]

    # Trick for transform all the other component to non-change and leave only the brush
    source_brush_mask = np.array(source_brush_mask)[..., 3]
    if np.sum(source_brush_mask) != 0:
        source_brush_mask[source_brush_mask != 0] = (
            4  # Pretend it to be 1 and will not be filtered
        )
        contours, _ = cv2.findContours(
            ((source_brush_mask > 0) * 255).astype("uint8"),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        mask = np.zeros((*source_brush_mask.shape, 3), dtype="uint8")
        mask[source_brush_mask != 0] = np.array([255, 205, 235])
        for c in contours:
            cv2.drawContours(mask, [c], -1, (0, 255, 0), 2)

        result = cv2.addWeighted(
            mask, 0.5, cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR), 0.5, 0
        )
        cv2.imwrite("temp.png", result)
        source_brush_mask[source_brush_mask == 0] = 255

        if source_mask is not None:
            source_mask = np.array(
                source_mask.resize(source_brush_mask.shape[:2][::-1])
            )
            source_brush_mask[(source_mask == 1) | (source_mask == 6)] = 255

    transform_result = text_translation.modify_with_text(
        image=source_image,
        prompt=[prompt],
        mask=source_brush_mask,
        guidance_scale=guidance_scale,
        start_from_step=180,
    )
    del text_translation
    torch.cuda.empty_cache()
    return source_mask, transform_result[0]


def analyze_skin_tone(image):
    """Analyze skin tone from uploaded image"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Simple skin tone analysis (in a real app, this would be more sophisticated)
    # For demo purposes, return a random skin tone match
    skin_tones = ["Light", "Medium", "Tan", "Deep"]
    return random.choice(skin_tones)


def find_matching_shades(skin_tone, selected_features, finish_preference):
    """Find matching shades based on skin tone and preferences"""
    recommendations = []
    
    for feature in selected_features:
        if feature in PRODUCT_DATABASE:
            # Filter by finish preference
            products = PRODUCT_DATABASE[feature]
            if finish_preference != "All shades":
                products = [p for p in products if finish_preference.lower() in p["name"].lower()]
            
            # Add 1-2 recommendations per feature
            recommendations.extend(products[:2])
    
    return recommendations


def create_before_after_comparison(original_image, modified_image):
    """Create a before/after comparison image"""
    if original_image is None or modified_image is None:
        return None
    
    # Resize images to same size
    width, height = original_image.size
    modified_resized = modified_image.resize((width, height))
    
    # Create side-by-side comparison
    comparison_width = width * 2
    comparison_height = height
    
    comparison = Image.new('RGB', (comparison_width, comparison_height))
    comparison.paste(original_image, (0, 0))
    comparison.paste(modified_resized, (width, 0))
    
    # Add dividing line
    draw = ImageDraw.Draw(comparison)
    draw.line([(width, 0), (width, height)], fill='white', width=2)
    
    return comparison


def generate_product_recommendations(selected_features, selected_colors, finish_preference):
    """Generate product recommendations based on selections"""
    recommendations = []
    
    for feature in selected_features:
        if feature in PRODUCT_DATABASE:
            products = PRODUCT_DATABASE[feature]
            if finish_preference != "All shades":
                products = [p for p in products if finish_preference.lower() in p["name"].lower()]
            
            # Match with selected colors if possible
            for product in products:
                if any(color in product["color"] for color in selected_colors):
                    recommendations.append({
                        "feature": feature,
                        "product": product["name"],
                        "shade": product["shade"],
                        "color": product["color"]
                    })
    
    return recommendations


def shade_finder_analysis(uploaded_image):
    """Analyze uploaded image for shade matching"""
    if uploaded_image is None:
        return None, "Please upload an image first"
    
    # Analyze skin tone
    skin_tone = analyze_skin_tone(uploaded_image)
    
    # Get matching shades
    matching_shades = find_matching_shades(skin_tone, ["Foundation"], "All shades")
    
    # Create HTML display for matching shades
    html_content = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
    for shade in matching_shades:
        html_content += f"""
        <div class='product-card' style='text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 8px;'>
            <div style='width: 40px; height: 40px; background-color: {shade['color']}; border-radius: 50%; margin: 0 auto 5px;'></div>
            <div style='font-size: 12px; font-weight: bold;'>{shade['name']}</div>
            <div style='font-size: 10px; color: #666;'>{shade['shade']}</div>
        </div>
        """
    html_content += "</div>"
    
    return skin_tone, html_content


def update_product_recommendations(selected_features, color_choice, finish_choice):
    """Update product recommendations based on selections"""
    if not selected_features:
        return "<p>Please select some beauty features first.</p>"
    
    recommendations = generate_product_recommendations(selected_features, [color_choice], finish_choice)
    
    if not recommendations:
        return "<p>No recommendations found for your selections.</p>"
    
    html_content = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
    for rec in recommendations:
        html_content += f"""
        <div class='product-card' style='text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 8px;'>
            <div style='width: 50px; height: 50px; background-color: {rec['color']}; border-radius: 8px; margin: 0 auto 5px;'></div>
            <div style='font-size: 14px; font-weight: bold;'>{rec['feature']}</div>
            <div style='font-size: 12px;'>{rec['product']}</div>
            <div style='font-size: 10px; color: #666;'>{rec['shade']}</div>
        </div>
        """
    html_content += "</div>"
    
    return html_content


def create_color_swatches():
    """Create HTML for color swatches"""
    html_content = "<div style='display: flex; flex-wrap: wrap; gap: 5px;'>"
    for i, color in enumerate(COLOR_PALETTE):
        html_content += f"""
        <div class='color-swatch' style='width: 30px; height: 30px; background-color: {color}; border-radius: 50%; border: 2px solid #ddd; cursor: pointer;' 
             onclick='selectColor(this, {i})'></div>
        """
    html_content += "</div>"
    return html_content


# Custom CSS for enhanced styling
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.color-swatch {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    border: 2px solid #ddd;
    cursor: pointer;
    display: inline-block;
    margin: 5px;
}
.color-swatch.selected {
    border-color: #ff6b6b;
    border-width: 3px;
}
.product-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 10px;
    margin: 5px;
    background: #f9f9f9;
}
"""

with gr.Blocks(title="Advanced Makeup Studio", css=custom_css) as demo:
    gr.Markdown("# 🎨 Advanced Makeup Studio")
    gr.Markdown("Transform your look with AI-powered makeup tools and personalized recommendations")
    
    with gr.Row():
        with gr.Column():
            config_file = gr.Dropdown(
                ["configs/model_256_256.yaml"],
                value="configs/model_256_256.yaml",
                label="Model Configuration",
            )
        with gr.Column():
            device = gr.Dropdown(["cpu", "cuda"], value="cuda", label="Device")

    with gr.Tab("🎯 Shade Finder"):
        gr.Markdown("### Find Your Perfect Shade Match")
        with gr.Row():
            with gr.Column(scale=2):
                shade_finder_image = gr.Image(type="pil", label="Upload Your Photo")
                shade_analysis_btn = gr.Button("🔍 Analyze Skin Tone", variant="primary")
                skin_tone_result = gr.Textbox(label="Detected Skin Tone", interactive=False)
                
                with gr.Row():
                    before_after_image = gr.Image(label="Before/After Comparison", interactive=False)
                    before_after_slider = gr.Slider(0, 100, value=50, label="Comparison Slider")
            
            with gr.Column(scale=1):
                gr.Markdown("### Your Matching Shades")
                matching_shades_display = gr.HTML()
                
                gr.Markdown("### Finish Options")
                finish_preference = gr.Radio(
                    FINISH_OPTIONS + ["All shades"], 
                    value="All shades", 
                    label="Preferred Finish"
                )
                
                gr.Markdown("### All Available Shades")
                all_shades_display = gr.HTML()

    with gr.Tab("💄 Beauty Features"):
        gr.Markdown("### Select Your Beauty Features")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Beauty Features")
                beauty_features = gr.CheckboxGroup(
                    BEAUTY_FEATURES,
                    value=["Lipstick", "Mascara"],
                    label="Select Features"
                )
                
                gr.Markdown("#### Function Options")
                function_options = gr.Radio(
                    FUNCTION_OPTIONS,
                    value="Short Top",
                    label="Function"
                )
                
                gr.Markdown("#### Finish Options")
                finish_options = gr.Radio(
                    FINISH_OPTIONS,
                    value="Satin",
                    label="Finish"
                )
            
            with gr.Column():
                gr.Markdown("#### Color Palette")
                color_palette = gr.Radio(
                    [f"Color {i+1}" for i in range(len(COLOR_PALETTE))],
                    value="Color 6",
                    label="Select Color"
                )
                
                gr.Markdown("#### Products for You")
                product_recommendations = gr.HTML()

    with gr.Tab("🔄 Domain Translation"):
        gr.Markdown("### Convert Between Makeup Styles")
        with gr.Row():
            domain_image_input = gr.Image(type="pil", label="Input Image")
            domain_mask_input = gr.Image(type="pil", label="Input Mask", image_mode="L")
            domain_image_output = gr.Image(type="pil", label="Output Image", interactive=False)
        
        with gr.Row():
            domain_source_label = gr.Dropdown(
                ["makeup", "non-makeup"], label="Source Label", value="non-makeup"
            )
            domain_target_label = gr.Dropdown(
                ["makeup", "non-makeup"], label="Target Label", value="makeup"
            )
            start_from_step = gr.Slider(minimum=0, maximum=1000, value=0, label="Start from step")
        
        domain_submit = gr.Button("🔄 Transform", variant="primary")
        
        with gr.Row():
            gr.Examples(
                examples=["data/mtdataset/images/non-makeup/xfsy_0307.png"],
                inputs=domain_image_input,
                label="Image Examples",
            )
            gr.Examples(
                examples=["data/mtdataset/parsing/non-makeup/xfsy_0307.png"],
                inputs=domain_mask_input,
                label="Mask Examples",
            )

    with gr.Tab("🎭 Reference Translation"):
        gr.Markdown("### Transfer Makeup from Reference Image")
        with gr.Row():
            reference_image_input_source = gr.Image(type="pil", label="Source Image")
            reference_mask_input_source = gr.Image(type="pil", label="Source Mask", image_mode="L")
        
        with gr.Row():
            reference_image_input_target = gr.Image(type="pil", label="Reference Image")
            reference_mask_input_target = gr.Image(type="pil", label="Reference Mask", image_mode="L")

        reference_image_output = gr.Image(type="pil", label="Output Image", interactive=False)
        
        with gr.Row():
            reference_source_label = gr.Dropdown(
                ["makeup", "non-makeup"], label="Source Label", value="non-makeup"
            )
            reference_target_label = gr.Dropdown(
                ["makeup", "non-makeup"], label="Target Label", value="makeup"
            )
        
        reference_submit = gr.Button("🎭 Transfer Makeup", variant="primary")
        
        with gr.Row():
            gr.Examples(
                examples=["data/mtdataset/images/non-makeup/vSYYZ223.png"],
                inputs=reference_image_input_source,
                label="Source Examples",
            )
            gr.Examples(
                examples=["data/mtdataset/parsing/non-makeup/vSYYZ223.png"],
                inputs=reference_mask_input_source,
                label="Source Mask Examples",
            )
        with gr.Row():
            gr.Examples(
                examples=["data/mtdataset/images/makeup/vFG66.png"],
                inputs=reference_image_input_target,
                label="Reference Examples",
            )
            gr.Examples(
                examples=["data/mtdataset/parsing/makeup/vFG66.png"],
                inputs=reference_mask_input_target,
                label="Reference Mask Examples",
            )

    with gr.Tab("✍️ Text Manipulation"):
        gr.Markdown("### Modify Makeup with Text Prompts")
        with gr.Row():
            text_image_input = gr.ImageMask(type="pil", label="Input Image")
            text_mask_input = gr.Image(type="pil", label="Input Mask", image_mode="L")

        with gr.Row():
            text_brush_mask = gr.Image(type="pil", label="Brush Mask", image_mode="L", interactive=False)
            text_image_output = gr.Image(type="pil", label="Output Image", interactive=False)

        with gr.Row():
            text_input = gr.Textbox(lines=1, label="Text Prompt", placeholder="e.g., 'makeup with smoky eyeshadow'")
            text_guidance_scale = gr.Slider(minimum=0, maximum=30, value=15, label="Guidance Scale")
        
        text_submit = gr.Button("✍️ Apply Text Changes", variant="primary")
        
        with gr.Row():
            gr.Examples(
                examples=[
                    "data/mtdataset/images/non-makeup/xfsy_0327.png",
                    "data/mtdataset/images/makeup/vFG805.png",
                ],
                inputs=text_image_input,
                label="Image Examples",
            )
            gr.Examples(
                examples=[
                    "data/mtdataset/parsing/non-makeup/xfsy_0327.png",
                    "data/mtdataset/parsing/makeup/vFG805.png",
                ],
                inputs=text_mask_input,
                label="Mask Examples",
            )
            gr.Examples(
                examples=["makeup with smoky eyeshadow", "bold red lipstick", "natural foundation"],
                inputs=text_input,
                label="Text Examples",
            )

    # Event handlers
    shade_analysis_btn.click(
        shade_finder_analysis,
        [shade_finder_image],
        [skin_tone_result, matching_shades_display]
    )
    
    beauty_features.change(
        update_product_recommendations,
        [beauty_features, color_palette, finish_options],
        [product_recommendations]
    )
    
    color_palette.change(
        update_product_recommendations,
        [beauty_features, color_palette, finish_options],
        [product_recommendations]
    )
    
    finish_options.change(
        update_product_recommendations,
        [beauty_features, color_palette, finish_options],
        [product_recommendations]
    )

    domain_submit.click(
        domain_translation,
        [config_file, device, start_from_step, domain_source_label, domain_target_label, domain_image_input, domain_mask_input],
        [domain_image_output],
    )

    reference_submit.click(
        reference_translation,
        [config_file, device, reference_source_label, reference_target_label, reference_image_input_source, reference_image_input_target, reference_mask_input_source, reference_mask_input_target],
        [reference_image_output],
    )

    text_submit.click(
        text_manipulate,
        [device, text_image_input, text_mask_input, text_input, text_guidance_scale],
        [text_brush_mask, text_image_output],
    )

# Launch the demo
demo.queue(max_size=1).launch(
    share=False,
    debug=True,
)
