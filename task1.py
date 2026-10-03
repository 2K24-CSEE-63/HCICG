import cv2

def inspect_image(image_path: str) -> dict:
    """Load the image and return its measured image-data properties."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # cv2 shape is (height, width, channels)
    height, width, channels = img.shape
    
    shape = [height, width, channels]  # autograder ko list chahiye, tuple nahi
    pixel_count = width * height
    estimated_bytes = pixel_count * channels  # 1 byte per channel
    color_order = "BGR"  # cv2 hamesha BGR me load karta hai

    # README ke liye print karna hai
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"Channels: {channels}")
    print(f"Shape: {shape}")
    print(f"Pixel Count: {pixel_count}")
    print(f"Estimated Bytes: {estimated_bytes}")
    print(f"Color Order: {color_order}")

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "shape": shape,
        "pixel_count": pixel_count,
        "estimated_bytes": estimated_bytes,
        "color_order": color_order
    }

if __name__ == "__main__":
    inspect_image("images/original.jpeg")