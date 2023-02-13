import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import skimage.io as io
from pycocotools.coco import COCO
import pylab
import os



datadir = str(os.getcwd() + '/' + 'dataset/')

def checking_TypeData(type):
    """
    It takes in a string, and returns a COCO object
    
    :param type: This is the type of data you want to check. It can be either train, val, or test
    :return: the coco object.
    """
    annFile = "{}/0COCO_{}_annos.json".format(datadir,type)
    coco = COCO(annFile)
    return coco

def check_category(type):
    """
    It takes in a type of data (train, val, or test) and returns the image and image id of the first
    image in the dataset that has a damage annotation
    
    :param type: the type of data you want to check
    :return: The image and the image id
    """
    coco = checking_TypeData(type)
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories for damages: \n{}\n'.format(', '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories for damages: \n{}'.format(', '.join(nms)))
    catIds = coco.getCatIds(catNms=['damage'])
    imgIds = coco.getImgIds(catIds=catIds )
    imgId = coco.getImgIds(imgIds = [4])
    img = coco.loadImgs(imgId)[0]
    return img, imgId

def visualImageNonDefect(type, img_dir):
    """
    This function takes in a type of defect and a directory of images and returns a visual of a
    non-defect image of that type
    
    :param type: the type of defect you want to see
    :param img_dir: the directory where the images are stored
    """
    img, imdId = check_category(type)
    I = io.imread(img_dir + '/' + img['file_name'])
    plt.axis('off')
    plt.imshow(I)
    plt.show()

def visualImageDefect(type, img_dir):
    """
    This function takes in a type of defect and an image directory and returns a visual representation
    of the defect in the image
    
    :param type: the type of defect you want to see
    :param img_dir: the directory where the images are stored
    """
    img, imgId = check_category(type)
    I = io.imread(img_dir + '/' + img['file_name'])
    coco = checking_TypeData(type)
    plt.imshow(I)
    plt.axis('on')
    annIds = coco.getAnnIds(imgIds=imgId,iscrowd=None)
    anns = coco.loadAnns(annIds)
    # plt.scatter([anns[0]["bbox"][0], anns[0]["bbox"][0] + anns[0]["bbox"][2]],[anns[0]["bbox"][1], anns[0]["bbox"][1] + anns[0]["bbox"][3]])
    # plt.annotate("top left ({},{})".format(anns[0]["bbox"][0], anns[0]["bbox"][1]), (anns[0]["bbox"][0], anns[0]["bbox"][1]))
    # plt.annotate("bottom right ({},{})".format(anns[0]["bbox"][0] + anns[0]["bbox"][2] , anns[0]["bbox"][1] + anns[0]["bbox"][3]), (anns[0]["bbox"][0] + anns[0]["bbox"][2] , anns[0]["bbox"][1] + anns[0]["bbox"][3]))
    coco.showAnns(anns, draw_bbox=True )