import os
import json
import random
import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

def analyze_skin_tone(image):
    """Analyze skin tone from uploaded image"""
    if image is None:
        return "Please upload an image first"
    
    # Simple skin tone analysis (in a real app, this would be more sophisticated)
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

def generate_product_recommendations(selected_features, color_choice, finish_choice):
    """Generate product recommendations based on selections"""
    if not selected_features:
        return "<p>Please select some beauty features first.</p>"
    
    recommendations = []
    
    for feature in selected_features:
        if feature in PRODUCT_DATABASE:
            products = PRODUCT_DATABASE[feature]
            if finish_choice != "All shades":
                products = [p for p in products if finish_choice.lower() in p["name"].lower()]
            
            # Match with selected colors if possible
            for product in products:
                if any(color in product["color"] for color in [color_choice]):
                    recommendations.append({
                        "feature": feature,
                        "product": product["name"],
                        "shade": product["shade"],
                        "color": product["color"]
                    })
    
    if not recommendations:
        return "<p>No recommendations found for your selections.</p>"
    
    html_content = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
    for rec in recommendations:
        html_content += f"""
        <div style='text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background: #f9f9f9;'>
            <div style='width: 50px; height: 50px; background-color: {rec['color']}; border-radius: 8px; margin: 0 auto 5px;'></div>
            <div style='font-size: 14px; font-weight: bold;'>{rec['feature']}</div>
            <div style='font-size: 12px;'>{rec['product']}</div>
            <div style='font-size: 10px; color: #666;'>{rec['shade']}</div>
        </div>
        """
    html_content += "</div>"
    
    return html_content

def shade_finder_analysis(uploaded_image):
    """Analyze uploaded image for shade matching"""
    if uploaded_image is None:
        return "Please upload an image first", "<p>No image uploaded</p>"
    
    # Analyze skin tone
    skin_tone = analyze_skin_tone(uploaded_image)
    
    # Get matching shades
    matching_shades = find_matching_shades(skin_tone, ["Foundation"], "All shades")
    
    # Create HTML display for matching shades
    html_content = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
    for shade in matching_shades:
        html_content += f"""
        <div style='text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background: #f9f9f9;'>
            <div style='width: 40px; height: 40px; background-color: {shade['color']}; border-radius: 50%; margin: 0 auto 5px;'></div>
            <div style='font-size: 12px; font-weight: bold;'>{shade['name']}</div>
            <div style='font-size: 10px; color: #666;'>{shade['shade']}</div>
        </div>
        """
    html_content += "</div>"
    
    return skin_tone, html_content

def simple_makeup_transform(image, style):
    """Simple makeup transformation for demo"""
    if image is None:
        return None
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Simple color adjustments based on style
    if style == "makeup":
        # Add warmth and saturation
        img_array = cv2.convertScaleAbs(img_array, alpha=1.1, beta=10)
        # Add slight red tint for makeup effect
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.05, 0, 255)
    elif style == "non-makeup":
        # Reduce saturation and add natural tone
        img_array = cv2.convertScaleAbs(img_array, alpha=0.95, beta=-5)
    
    return Image.fromarray(img_array)

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

    with gr.Tab("🔄 Makeup Transform"):
        gr.Markdown("### Transform Your Look")
        with gr.Row():
            transform_image_input = gr.Image(type="pil", label="Upload Your Photo")
            transform_image_output = gr.Image(type="pil", label="Transformed Image", interactive=False)
        
        with gr.Row():
            transform_style = gr.Radio(
                ["makeup", "non-makeup"], 
                value="makeup", 
                label="Transform Style"
            )
            transform_btn = gr.Button("🔄 Transform", variant="primary")
        
        transform_btn.click(
            simple_makeup_transform,
            [transform_image_input, transform_style],
            [transform_image_output]
        )

    # Event handlers
    shade_analysis_btn.click(
        shade_finder_analysis,
        [shade_finder_image],
        [skin_tone_result, matching_shades_display]
    )
    
    beauty_features.change(
        generate_product_recommendations,
        [beauty_features, color_palette, finish_options],
        [product_recommendations]
    )
    
    color_palette.change(
        generate_product_recommendations,
        [beauty_features, color_palette, finish_options],
        [product_recommendations]
    )
    
    finish_options.change(
        generate_product_recommendations,
        [beauty_features, color_palette, finish_options],
        [product_recommendations]
    )

# Launch the demo
if __name__ == "__main__":
    demo.queue(max_size=1).launch(
        share=False,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860
    )
