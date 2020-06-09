from models import Pix2Pix, ResNet, Assisted


def get_model(name, training, *args):
    if name.lower() == 'pix2pix':
        if training.lower() == 'color_assisted':
            return Assisted(*args)
        elif training.lower() == 'text_assisted':
            pass
        else:
            return Pix2Pix(*args)

    elif name.lower() == 'resnet':
        return ResNet()
