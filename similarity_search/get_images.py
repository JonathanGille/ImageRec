import os
from PIL import Image

class c_image():
    def __init__(self, path, name):
        self.img = Image.open(path)
        self.name = name
        self.path = path

def get_images(folder):
    dirlist = os.listdir(folder)
    # image_paths = [os.path.join(folder,img) for img in dirlist]
    # image_names = [img[:-4] for img in dirlist]

    images = [c_image(path=os.path.join(folder,img), name=img[:-4]) for img in dirlist]

    # images = [c_image(path) for path in image_paths]
    # return (images, image_names)

    return images