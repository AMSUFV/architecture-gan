"""Settings for the temple reconstruction training.
DATASET: Type of dataset. Available:
    * color_assisted:       (temple ruins, segmented temple, temple)
    * color_reconstruction: (temple ruins segmented, temple segmented)
    * reconstruction:       (temple ruins, temple)
    * segmentation:         (temple segmented, temple)
    * de-segmentation:      (temple, temple segmented)
    * masking:              (temple ruins, temple ruins masked); aimed at marking the missing areas
    * de-masking            (temple ruins masked, temple); aimed at reconstructing the marked areas
    * text_assisted         (temple ruins, temple description, temple)
"""

DATASET = 'color_assisted'
TEMPLES = [0, 9]
MODEL = ''
