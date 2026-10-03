import cv2
import os
import numpy as np

def create_pixel_views(image_path: str, output_dir: str) -> dict:
    """Create the labeled channel, grayscale, and downsampled views."""
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    height, width = img.shape[:2]
    original_size = [width, height]

    # 1. Grayscale view
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 2. Downsampled view (pixels dikhane ke liye chota karke wapas bada)
    down_w, down_h = width // 20, height // 20
    if down_w < 1: down_w = 16
    if down_h < 1: down_h = 16
    downsampled_size = [down_w, down_h]

    small = cv2.resize(img, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

    # 3. Montage banao with labels
    def add_label(image, text):
        out = image.copy()
        cv2.rectangle(out, (0,0), (out.shape[1], 30), (0,0,0), -1)
        cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return out

    top_row = np.hstack([add_label(img, "Original"), add_label(gray_bgr, "Grayscale")])
    bottom_row = np.hstack([add_label(pixelated, f"Downsampled {down_w}x{down_h}"), add_label(img, "Original Copy")])

    # Agar size mismatch ho to resize
    if top_row.shape[1]!= bottom_row.shape[1]:
        bottom_row = cv2.resize(bottom_row, (top_row.shape[1], bottom_row.shape[0]))

    montage = np.vstack([top_row, bottom_row])

    output_path = os.path.join(output_dir, "pixel_views.png")
    cv2.imwrite(output_path, montage)

    print(f"Saved to: {output_path}")
    print(f"Original: {original_size}, Downsampled: {downsampled_size}")

    return {
        "original_size": original_size,
        "downsampled_size": downsampled_size,
        "output_path": output_path
    }

if __name__ == "__main__":
    create_pixel_views("Images/orignal.jpeg", "outputs")