import numpy as np
import cv2
from tqdm.auto import tqdm

def NumberDamage(outputs):
    """
    It takes the predicted masks from the model, and returns the contours of each mask, and the class of
    each mask.
    
    :param outputs: the output of the model, which is a dictionary
    :return: a list of contours and a list of classes.
    """
    # Extract the contour of each predicted mask and save it in a list
    List_Class = outputs['instances'].pred_classes.tolist()
    contours = []
    for pred_mask in outputs['instances'].pred_masks:
        # pred_mask is of type torch.Tensor, and the values are boolean (True, False)
        # Convert it to a 8-bit numpy array, which can then be used to find contours
        mask = pred_mask.cpu().numpy().astype('uint8')
        contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        contours.append(contour[0]) # contour is a tuple (OpenCV 4.5.2), so take the first element which is the array of contour points
    return contours, List_Class

def PolyArea(x,y):
    """
    The function takes two lists of x and y coordinates and returns the area of the polygon defined by
    those coordinates
    
    :param x: x coordinates of the polygon
    :param y: the y-coordinates of the polygon's vertices
    :return: The area of the polygon.
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def SplitLocation(contours):
    """
    It takes a list of lists of lists of lists of integers and returns a list of lists of integers and a
    list of lists of integers
    
    :param contours: list of contours
    :return: two lists of lists. The first list contains the x-coordinates of the contours, and the
    second list contains the y-coordinates of the contours.
    """
    all_point_x = []
    all_point_y = []
    for i in tqdm(range(len(contours))):
        x_all = []
        y_all = []
        for j in range(len(contours[i])):
            x = contours[i][j][0][0]
            y = contours[i][j][0][1]
            x_all.append(x)
            y_all.append(y)
        all_point_x.append(x_all)
        all_point_y.append(y_all)
    return all_point_x, all_point_y

def PixtoCm(listArea):
    """
    It takes a list of pixel values and converts them to cm^2
    
    :param listArea: list of areas in pixels
    :return: the list of values in cm^2.
    """
    Newlist = []
    for i in range(len(listArea)):
        CmValue = listArea[i] * 0.0264583333
        Newlist.append(CmValue)
    return Newlist

def DictClassArea():
    pass

'''
0 - mop_lom ----
1 - tray_son ----
2 - Rach ----
3 - mat_bo_phan ---
4 - thung ----
5 - Be_Den ----
6 - vo_kinh ----
'''