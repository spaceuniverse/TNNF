# ---------------------------------------------------------------------#
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from numpy import *
from fDataWorkerCORE import *
# ---------------------------------------------------------------------#


class Graphic(object):
    @staticmethod
    def PicSaver(img, folder, name, color="L"):  # Saves picture to folder. Color "L" or "RGB"
        imsave = Image.fromarray(DataMutate.Normalizer(img))  # Normalizer(img).astype('uint8') for RGB
        imsave = imsave.convert(color)
        imsave.save(folder + name + ".jpg", "JPEG", quality=100)

# ---------------------------------------------------------------------#


class MultiWeights(object):

    def __init__(self, path='./', name='multi_weights.png'):


        self.pathToSave = path

        #Check last symbol to be '/'
        if self.pathToSave[-1] != '/':
            self.pathToSave += '/'

        #Filename to save
        if name != 'multi_weights.png':
            self.name = name + '.png'
        else:
            self.name = name

        #Pics list
        self.pictures = []

        #Pics location
        self.width = None
        self.height = None

    def defineOptimalPicLocation(self, n):

        self.width = int(np.ceil(np.sqrt(n) * 0.8))
        self.height = int(np.ceil(np.true_divide(n, self.width)))

    def add(self, p):

        #Add pictures to list
        self.pictures.append(p)

    def draw(self):
        #Number of picture
        numOfPictures = len(self.pictures)

        # Border between pictures
        border = 1

        #Picture's size
        picH, picW, picC = self.pictures[-1].shape

        #Define pictures location
        self.defineOptimalPicLocation(numOfPictures)

        #Prepare plate for weights
        plate = Image.new('RGBA',
                          (int(border + self.width * (picW + border)), int(border + self.height * (picH + border))),
                          (0, 0, 0, 255))

        #Plate's size
        plateW, plateH = plate.size

        #Iterate over picture's location
        count = 0
        for h in xrange(self.height):
            for w in xrange(self.width):

                offset = (border + w * (picW + border), border + h * (picH + border))
                #print self.pictures[count].shape
                plate.paste(Image.fromarray(DataMutate.Normalizer(self.pictures[count]).astype('uint8')), offset)

                count += 1
                if count == numOfPictures:
                    break

        plate.save(self.pathToSave + self.name)





