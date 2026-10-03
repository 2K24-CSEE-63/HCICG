import cv2
import numpy as np
import os

def create_blur_and_edges(
    image_path: str,
    output_dir: str,
    kernel_size: int = 5,
) -> dict:
    """Create labeled grayscale, mean-blur, and Sobel-edge results."""
    
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    
    os.makedirs(output_dir, exist_ok=True)
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Mean blur
    blurred = cv2.blur(gray, (kernel_size, kernel_size))
    
    # 2. Sobel edges - x and y then magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)

    def add_label(image, text):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        labeled = np.zeros((h + 40, w, 3), dtype=np.uint8)
        labeled[40:, :] = image
        cv2.putText(labeled, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return labeled

    img1 = add_label(gray, "Grayscale")
    img2 = add_label(blurred, f"Mean Blur k={kernel_size}")
    img3 = add_label(sobel_mag, f"Sobel Edges k={kernel_size}")

    montage = np.hstack([img1, img2, img3])
    
    output_path = os.path.join(output_dir, "blur_and_edges.png")
    cv2.imwrite(output_path, montage)
    
    print(f"Saved: {output_path}")
    print(f"Kernel: {kernel_size}")

    return {
        "kernel_size": kernel_size,
        "output_path": output_path
    }

if __name__ == "__main__":
    create_blur_and_edges("Images/orignal.jpeg", "outputs")