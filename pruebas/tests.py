from PIL import Image
from PIL import ImageFilter


class ClassOne:
    def __init__(self):
        self.nombre = 'class one'

    @staticmethod
    def sum(a, b):
        return a + b


class ClassTwo(ClassOne):

    @staticmethod
    def subs(a, b):
        return a - b


if __name__ == '__main__':
    # image = Image.open('ruina2.png')
    # edges = image.filter(ImageFilter.EDGE_ENHANCE)
    # edges.save('test.png')
    thing = ClassTwo()
    # print(thing.apellido)
    print(thing.nombre)
