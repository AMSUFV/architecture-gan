from models import Pix2Pix, ResNet


def get_model(name, training):

    if training == 'color_assisted':
        heads = 2
    else:
        heads = 1

    if name.lower() == 'pix2pix':
        return Pix2Pix(heads=heads)

    elif name.lower() == 'resnet':
        return ResNet()
