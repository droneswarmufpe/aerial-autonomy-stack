import cv2
import os

def flip_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if img is None:
        print("Error: Unable to load image.")
        return

    # Flip the image horizontally
    img_flipped = cv2.flip(img, 1)

    # Extract the filename and extension
    filename, extension = os.path.splitext(image_path)

    # Define the output file path
    output_path = f"{filename}_flipped{extension}"

    # Save the flipped image
    cv2.imwrite(output_path, img_flipped)

    print(f"Flipped image saved as: {output_path}")

# Example usage:
image_path = "april_6x6_80x80cm_A0_page-0001.png"  # Replace with your image file path
flip_image(image_path)