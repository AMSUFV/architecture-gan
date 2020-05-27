from models import Pix2Pix, ResNet


def get_model(name, training, *args):

    if training == 'color_assisted':
        heads = 2
    else:
        heads = 1

    if name.lower() == 'pix2pix':
        return Pix2Pix(*args)

    elif name.lower() == 'resnet':
        return ResNet()
