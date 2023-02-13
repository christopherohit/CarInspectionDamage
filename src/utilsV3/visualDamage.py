from RestructModelings.utils.visualizer import ColorMode, Visualizer

'''
List variable and datatype of variable
All output in this file
if get_image had to set the true it will return
out : numpy.ndarray
else get_image had to by default false it will retur
out : RestructModelings.utils.visualizer.VisImage
'''

#region It have function code to visual type of damage
def visualOnlyMaskDamage(image, damageLocation, metaData = None , get_image = False):

    v = Visualizer(image,
                    metadata= metaData,
                    scale= 1,
                    instance_mode= ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(damageLocation['instances'].to('cpu'))
    masks=damageLocation["instances"].get("pred_masks")
    masks=masks.to("cpu")
    v.overlay_instances(
        masks=masks,
        # boxes=None,
        # labels=None,
        # keypoints=None,
        # assigned_colors=None,
        alpha=0.3,
        )
    out = v.get_output()
    if get_image == False:
        return out
    else:
        out = out.get_image()
        return out

def visualOnlyBoxDamage(image, damageLocation, metaData = None, get_image = False):
    v = Visualizer(image,
                   metadata= metaData,
                   scale= 1,
                   instance_mode= ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(damageLocation['instances'].to('cpu'))
    for box in damageLocation['instances'].pred_boxes.to('cpu'):
        v.draw_box(box)
    out = v.get_output()
    if get_image == False:
        return out
    else:
        out = out.get_image()
        return out

def visualBothDamage(image, damageLocation, metaData = None, get_image = False):
    """
    It takes in an image, the damage location, and the metadata, and returns a visualized image with the
    damage location
    
    :param image: the image you want to visualize
    :param damageLocation: the output of the model
    :param metaData: This is the metadata for the dataset. It is used to get the names of the classes
    :param get_image: If True, returns an image, if False, returns a visualizer object, defaults to
    False (optional)
    :return: The output is a tensor of the image with the damage location overlayed.
    """
    v = Visualizer(image,
                   metadata= metaData,
                   scale= 1,
                   instance_mode= ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(damageLocation['instances'].to('cpu'))
    out = v.get_output()
    if get_image == False:
        return out
    else:
        out = out.get_image()
        return out
#endregion

#region
def visualBoxCarPart(image, partLocation, metaData = None, get_image  = False):
    """
    It takes an image, a dictionary of part locations, and a metadata object (which is optional) and
    returns a visualized image
    
    :param image: the image you want to draw on
    :param partLocation: the output of the model
    :param metaData: This is the metadata that is used to create the visualizer
    :param get_image: If True, returns an image, otherwise returns a Visualizer object, defaults to
    False (optional)
    :return: The image with the bounding boxes drawn on it.
    """
    v = Visualizer(image,
                   metadata= metaData,
                   scale= 1,
                   instance_mode= ColorMode.IMAGE)
    out = v.draw_instance_predictions(partLocation['instances'].to('cpu'))
    if get_image == False:
        return out
    else:
        out = out.get_image()
        return out
#endregion