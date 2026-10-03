import cv2
import numpy as np
import os

def create_adjustments(
    image_path: str,
    output_dir: str,
    brightness_delta: int = 40,
    contrast_factor: float = 1.5,
    threshold: int = 127,
) -> dict:
    """Create labeled brightness, contrast, and threshold results."""

    if not (0 <= threshold <= 255):
        raise ValueError("threshold must be in range 0-255")

    os.makedirs(output_dir, exist_ok=True)
    
    # Image read
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Brightness - add delta and clip
    bright = np.clip(gray.astype(np.int16) + brightness_delta, 0, 255).astype(np.uint8)

    # 2. Contrast - multiply by factor and clip
    contrast = np.clip(gray.astype(np.float32) * contrast_factor, 0, 255).astype(np.uint8)

    # 3. Threshold - > threshold white else black
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Helper for labeled image
    def add_label(image, text):
        # Convert to BGR if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        # Black strip for label
        labeled = np.zeros((h + 40, w, 3), dtype=np.uint8)
        labeled[40:, :] = image
        cv2.putText(labeled, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return labeled

    img1 = add_label(bright, f"Brightness +{brightness_delta}")
    img2 = add_label(contrast, f"Contrast x{contrast_factor}")
    img3 = add_label(thresh, f"Threshold {threshold}")

    # Montage - 1 row, 3 columns
    montage = np.hstack([img1, img2, img3])
    
    output_path = os.path.join(output_dir, "adjustments.png")
    cv2.imwrite(output_path, montage)
    
    print(f"Saved: {output_path}")
    print(f"Brightness: {brightness_delta}, Contrast: {contrast_factor}, Threshold: {threshold}")

    return {
        "brightness_delta": brightness_delta,
        "contrast_factor": contrast_factor,
        "threshold": threshold,
        "output_path": output_path
    }

if __name__ == "__main__":
    create_adjustments("Images/orignal.jpeg", "outputs")