import os
import cv2
import numpy as np

def load_and_describe(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))
    return {"width": w, "height": h, "channels": c, "shape": [h, w, c], "mean_intensity": mean_intensity}

def create_pixel_views(image_path: str, output_dir: str, downsample_factor: int = 20) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small_h, small_w = max(1, h // downsample_factor), max(1, w // downsample_factor)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    def to_bgr(im): return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if len(im.shape)==2 else im
    def add_label(im, text):
        im = to_bgr(im); H,W = im.shape[:2]
        out = np.zeros((H+40, W, 3), dtype=np.uint8); out[40:,:]=im
        cv2.putText(out, text, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return out
    montage = np.vstack([np.hstack([add_label(img, "Original"), add_label(gray, "Grayscale")]), np.hstack([add_label(pixelated, f"Pixelated x{downsample_factor}"), add_label(img, "Original")])])
    out_path = os.path.join(output_dir, "pixel_views.png")
    cv2.imwrite(out_path, montage)
    return {"downsample_factor": downsample_factor, "original_shape": [h,w], "downsampled_shape": [small_h, small_w], "output_path": out_path}

def create_adjustments(image_path: str, output_dir: str, brightness_delta: int = 40, contrast_factor: float = 1.5, threshold: int = 127) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright = np.clip(gray.astype(np.int16)+brightness_delta,0,255).astype(np.uint8)
    contrast = np.clip(gray.astype(np.float32)*contrast_factor,0,255).astype(np.uint8)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    def add_label(im, text):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if len(im.shape)==2 else im
        h,w=im.shape[:2]; out=np.zeros((h+40,w,3),dtype=np.uint8); out[40:,:]=im
        cv2.putText(out,text,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2); return out
    montage = np.hstack([add_label(bright,f"Brightness +{brightness_delta}"), add_label(contrast,f"Contrast x{contrast_factor}"), add_label(thresh,f"Threshold {threshold}")])
    out_path = os.path.join(output_dir, "adjustments.png")
    cv2.imwrite(out_path, montage)
    return {"brightness_delta":brightness_delta,"contrast_factor":contrast_factor,"threshold":threshold,"output_path":out_path}

def create_blur_and_edges(image_path: str, output_dir: str, kernel_size: int = 5) -> dict:
    if kernel_size<=0 or kernel_size%2==0: raise ValueError("kernel_size must be positive odd")
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (kernel_size, kernel_size))
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    mag = np.clip(cv2.magnitude(sx,sy),0,255).astype(np.uint8)
    def add_label(im, text):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if len(im.shape)==2 else im
        h,w=im.shape[:2]; out=np.zeros((h+40,w,3),dtype=np.uint8); out[40:,:]=im
        cv2.putText(out,text,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2); return out
    montage = np.hstack([add_label(gray,"Grayscale"), add_label(blurred,f"Mean Blur k={kernel_size}"), add_label(mag,f"Sobel Edges k={kernel_size}")])
    out_path = os.path.join(output_dir, "blur_and_edges.png")
    cv2.imwrite(out_path, montage)
    return {"kernel_size":kernel_size,"output_path":out_path}

# ===== TASK 5 =====
def run_lab(image_path: str, output_dir: str) -> dict:
    """Run Tasks 1-4 and return their results together."""
    os.makedirs(output_dir, exist_ok=True)
    t1 = load_and_describe(image_path)
    t2 = create_pixel_views(image_path, output_dir)
    t3 = create_adjustments(image_path, output_dir)
    t4 = create_blur_and_edges(image_path, output_dir)
    return {"task1": t1, "task2": t2, "task3": t3, "task4": t4}

def main() -> None:
    """Run the lab using the required repository paths."""
    image_path = "images/original.jpg"
    output_dir = "outputs/"
    os.makedirs(output_dir, exist_ok=True)
    # fallback for your current name
    if not os.path.exists(image_path):
        if os.path.exists("images/original.jpeg"):
            image_path = "images/original.jpeg"
    print(f"Running lab: {image_path} -> {output_dir}")
    results = run_lab(image_path, output_dir)
    print("Done:", list(results.keys()))

if __name__ == "__main__":
    main()