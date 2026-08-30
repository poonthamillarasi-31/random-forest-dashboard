"""
Vision and Color Palette Extraction Engine
Extracts dominant color palettes, HEX codes, percentage shares,
color harmony, and visual attributes from fashion garment images using K-Means clustering.
"""

import io
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Tuple

# Reference Color Database for Naming
NAMED_PALETTE = {
    "#8A9A86": "Sage Green",
    "#B5A7D6": "Digital Lavender",
    "#BA131A": "Fiery Cherry Red",
    "#4B3728": "Mocha Espresso Brown",
    "#0047AB": "Cobalt Blue",
    "#C0C0C0": "Metallic Silver",
    "#FFF1A8": "Butter Yellow",
    "#FFBE98": "Peach Fuzz",
    "#1C1C1E": "Obsidian Black",
    "#F8F9FA": "Optic White",
    "#D2B48C": "Classic Camel Tan",
    "#800020": "Burgundy Wine",
    "#2E8B57": "Forest Emerald",
    "#4682B4": "Steel Denim Blue",
    "#FF69B4": "Hot Barbiecore Pink"
}


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Converts (R, G, B) tuple to #RRGGBB hex string."""
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2])).upper()


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Converts #RRGGBB hex string to (R, G, B) tuple."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def find_nearest_color_name(rgb: Tuple[int, int, int]) -> str:
    """Finds the closest fashion color name in Euclidean RGB space."""
    min_dist = float("inf")
    closest_name = "Custom Tone"

    for hex_code, name in NAMED_PALETTE.items():
        ref_rgb = hex_to_rgb(hex_code)
        dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, ref_rgb)))
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name


def extract_dominant_colors_from_image(
    image: Image.Image,
    num_colors: int = 5,
    sample_size: int = 200
) -> List[Dict[str, Any]]:
    """
    Applies K-Means clustering to extract dominant color clusters, HEX codes,
    percentages, and closest fashion names from a PIL image.
    """
    # Resize image for fast clustering
    img = image.convert("RGB")
    img.thumbnail((sample_size, sample_size))
    
    img_arr = np.array(img)
    pixels = img_arr.reshape((-1, 3))

    # Run KMeans
    kmeans = KMeans(n_clusters=num_colors, n_init=5, random_state=42)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Calculate percentage distribution
    counts = np.bincount(labels)
    total_pixels = len(labels)
    percentages = (counts / total_pixels) * 100

    results = []
    # Sort by percentage descending
    sorted_indices = np.argsort(percentages)[::-1]

    for idx in sorted_indices:
        rgb = tuple(colors[idx].astype(int))
        hex_code = rgb_to_hex(rgb)
        nearest_name = find_nearest_color_name(rgb)
        pct = round(float(percentages[idx]), 1)

        results.append({
            "hex": hex_code,
            "rgb": rgb,
            "name": nearest_name,
            "percentage": pct
        })

    return results


def analyze_color_harmony(extracted_colors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates color relationships to determine aesthetic harmony
    (e.g., Monochromatic, Complementary, Earthy Warm, High-Contrast Modern).
    """
    if not extracted_colors:
        return {"harmony_type": "Neutral", "description": "Insufficient color data."}

    # Extract RGB values
    rgbs = [c["rgb"] for c in extracted_colors[:3]]
    
    # Calculate variance across R, G, B
    rgbs_arr = np.array(rgbs)
    mean_color = np.mean(rgbs_arr, axis=0)
    variance = np.mean(np.var(rgbs_arr, axis=0))
    
    # Average saturation proxy
    max_min_diff = np.mean(np.max(rgbs_arr, axis=1) - np.min(rgbs_arr, axis=1))

    if max_min_diff < 30:
        harmony_type = "Quiet Luxury Monochrome / Minimalist"
        desc = "Muted, low-saturation neutral tones favored in timeless capsule tailoring."
    elif variance < 600:
        harmony_type = "Analogous Harmony"
        desc = "Adjacent color tones on the spectrum offering cohesive and balanced styling."
    elif variance > 3000:
        harmony_type = "Dynamic High-Contrast / Streetwear Bold"
        desc = "Vibrant contrasting color accents ideal for statement outerwear and graphic pieces."
    else:
        harmony_type = "Organic Complementary"
        desc = "Harmonious balance of warm and cool undertones."

    return {
        "harmony_type": harmony_type,
        "description": desc,
        "primary_hue": extracted_colors[0]["name"]
    }


def create_sample_fashion_image(aesthetic_theme: str = "Quiet Luxury") -> Image.Image:
    """
    Generates a synthetic fashion moodboard / garment swatch image for demo testing.
    """
    width, height = 400, 400
    img = Image.new("RGB", (width, height), color="#F0EBE1")
    draw = ImageDraw.Draw(img)

    if aesthetic_theme == "Quiet Luxury":
        # Sage green, camel tan, off-white
        draw.rectangle([20, 20, 380, 180], fill="#8A9A86")
        draw.rectangle([20, 200, 190, 380], fill="#D2B48C")
        draw.rectangle([210, 200, 380, 380], fill="#4B3728")
    elif aesthetic_theme == "Cherry Red Bold":
        draw.rectangle([20, 20, 380, 220], fill="#BA131A")
        draw.rectangle([20, 240, 190, 380], fill="#1C1C1E")
        draw.rectangle([210, 240, 380, 380], fill="#C0C0C0")
    else:
        # Butter yellow & cobalt
        draw.rectangle([20, 20, 280, 380], fill="#FFF1A8")
        draw.rectangle([300, 20, 380, 190], fill="#0047AB")
        draw.rectangle([300, 210, 380, 380], fill="#B5A7D6")

    return img
