"""
Visualize test-time augmentations
"""

from PIL import Image
import torchvision.transforms as tf
import torchvision.transforms.functional as TF
from functools import partial


img = Image.open("astronaut.jpg")


transformations = [
            TF.vflip,
            TF.hflip,
            partial(TF.rotate, angle=90, expand=True),
            partial(TF.rotate, angle=180, expand=True),
            partial(TF.rotate, angle=270, expand=True),
            tf.Compose([TF.vflip, partial(TF.rotate, angle=90, expand=True)]),
            tf.Compose([TF.vflip, partial(TF.rotate, angle=-90, expand=True)])
            ]

detransformations = [
            TF.vflip,
            TF.hflip,
            partial(TF.rotate, angle=-90, expand=True),
            partial(TF.rotate, angle=-180, expand=True),
            partial(TF.rotate, angle=-270, expand=True),
            tf.Compose([partial(TF.rotate, angle=-90, expand=True), TF.vflip]),
            tf.Compose([partial(TF.rotate, angle=90, expand=True), TF.vflip])
            ]

for i in range(len(transformations)):
    img_transf = transformations[i](img)
    img_transf.show(title="transf" + str(i))
    img_detransf = detransformations[i](img_transf)
    img_detransf.show(title="detransf" + str(i))
