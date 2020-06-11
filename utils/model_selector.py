from models import Pix2Pix, ResNet, Assisted, TextAssisted


def get_model(name, training, *args):
    if name.lower() == 'pix2pix':
        if training.lower() == 'color_assisted':
            return Assisted(*args)
        elif training.lower() == 'text_assisted':
            return TextAssisted(*args)
        else:
            return Pix2Pix(*args)

    elif name.lower() == 'resnet':
        return ResNet()
