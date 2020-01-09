from PIL import Image
from PIL import ImageFilter

if __name__ == '__main__':
    image = Image.open('ruina2.png')
    edges = image.filter(ImageFilter.EDGE_ENHANCE)
    edges.save('test.png')
