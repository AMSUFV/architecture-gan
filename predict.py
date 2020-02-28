import argparse

parser = argparse.ArgumentParser(description='Launch model prediction')

parser.add_argument('-model', '-m', type=str, required=True,
                    help='Model to use for the prediction(s).')
parser.add_argument('-source', '-s', type=str, required=True,
                    help='What to predict. Path to a single image or to an image folder')
