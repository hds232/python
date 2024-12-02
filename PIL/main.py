import os
from PIL import Image

def main():
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    image_files = [f for f in os.listdir() if os.path.isfile(os.path.join('.', f)) 
                and f.lower().endswith(image_extensions)]
    open_images = [Image.open(f) for f in image_files]
    total_width, max_height = sum([i.size[1] for i in open_images]), max([i.size[0] for i in open_images])
    new_im = Image.new('RGB', (max_height, total_width), (255, 255, 255))
    for im in open_images:
        new_im.paste(im, (0, sum([i.size[1] for i in open_images[:open_images.index(im)]])))
    new_im.save('new_image.jpg')
if __name__ == '__main__':
    main()