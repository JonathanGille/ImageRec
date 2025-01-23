from PIL import Image

# Open the image file
img = Image.open('test_im.png')

# Resize the image to 224x224 pixels
img_resized = img.resize((224, 224))

# Save the resized image
img_resized.save('resized_image.png')