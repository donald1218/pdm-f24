import os
from PIL import Image


def make_gif(output):
    image_folder = 'data_collection/first_floor/rgb'
    gif_filename = output + '.gif'

    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            images.append(img_path)

    images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    if images:
        with Image.open(images[0]) as first_image:
            frames = []
            for img_path in images[1:]:
                with Image.open(img_path) as img:
                    frames.append(img.copy())  


            first_image.save(gif_filename, save_all=True, append_images=frames, duration=100, loop=0)

        print(f'GIF saved as {gif_filename}')
    else:
        print('No images found in the specified folder.')


if __name__ == "__main__":
    make_gif("cooktop")