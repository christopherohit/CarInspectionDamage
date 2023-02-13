import os
import cv2
import datetime
from tqdm.auto import tqdm
import json
from via2coco import getArea
import numpy as np
import shutil


currentWorking = os.getcwd()

CAR_DATASET_ORIGINAL_IMG_PATH = '/content/drive/MyDrive/Detection/Car_Damage_Detection/preprocess_data/train/'
CAR_DATASET_TRAIN_IMG_PATH = '/content/drive/MyDrive/Detection/Car_Damage_Detection/preprocess_data/new_data/train/'
CAR_DATASET_VAL_IMG_PATH = '/content/drive/MyDrive/Detection/Car_Damage_Detection/preprocess_data/new_data/val/'
ANNOTATIONS_NAME = "0_via_train.json"


def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    """
    It creates a dictionary with the keys: id, file_name, width, height, date_captured, license,
    coco_url, flickr_url.
    
    :param image_id: An integer ID that uniquely identifies this image
    :param file_name: The name of the image file
    :param image_size: The size of the image
    :param date_captured: The date the image was captured
    :param license_id: The license under which the image is released, defaults to 1 (optional)
    :param coco_url: The URL of the image on the COCO website
    :param flickr_url: The URL of the image on Flickr
    :return: A dictionary with the following keys:
    id
    file_name
    width
    height
    date_captured
    license
    coco_url
    flickr_url
    """
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[1],
        "height": image_size[0],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }

    return image_info



def create_annotation_info(annotation_id, image_id, category_id, is_crowd,
                           area, bounding_box, segmentation):
    """
    It takes in an annotation_id, image_id, category_id, is_crowd, area, bounding_box, and segmentation,
    and returns a dictionary with the same information.
    
    :param annotation_id: An integer ID for the annotation
    :param image_id: The id of the image that the annotation is for
    :param category_id: The id of the category that this annotation belongs to
    :param is_crowd: set to 1 if the object is a crowd of objects
    :param area: The area of the bounding box
    :param bounding_box: [x,y,width,height]
    :param segmentation: [polygon]
    :return: A dictionary with the following keys:
        id
        image_id
        category_id
        iscrowd
        area
        bbox
        segmentation
    """
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,  # float
        "bbox": bounding_box,  # [x,y,width,height]
        "segmentation": segmentation  # [polygon]
    }
    return annotation_info
    

def get_segmenation(coord_x, coord_y):
    """
    It takes two lists of coordinates and returns a list of segments
    
    :param coord_x: x coordinates of the points
    :param coord_y: The y-coordinates of the points in the line
    :return: A list of lists.
    """
    seg = []
    for x, y in zip(coord_x, coord_y):
        seg.append(x)
        seg.append(y)
    return [seg]


def convert(VIA_ORIGINAL_ANNOTATIONS_NAME, imgdir, annpath):
    """
    The function takes in the original annotations file, the directory of the images, and the path of
    the annotations. It then creates a dictionary of the coco style, which you can dump into a json file
    
    :param VIA_ORIGINAL_ANNOTATIONS_NAME: The name of the VIA annotations file
    :param imgdir: directory for your images
    :param annpath: path for your annotations
    :return: coco_output is a dictionary of coco style which you could dump it into a json file
    """
    """
    :param imgdir: directory for your images
    :param annpath: path for your annotations
    :return: coco_output is a dictionary of coco style which you could dump it into a json file
    as for keywords 'info','licenses','categories',you should modify them manually
    """

    annotations = json.load(open(CAR_DATASET_ORIGINAL_IMG_PATH + VIA_ORIGINAL_ANNOTATIONS_NAME, encoding="utf-8"))
    # annotations = list(annotations.values())  # don't need the dict keys
    annotations = [a for a in annotations if a['regions']]
    name_supercategory_dict = {}
    for a in tqdm(annotations):
        names = [r['class'] for r in a['regions']]
        supercategories = ["damage" for r in a['regions']]
        for index, name in enumerate(names):
            if name not in name_supercategory_dict.keys():
                name_supercategory_dict[name] = supercategories[index]
                
    coco_output = {}
    coco_output['info'] = {
        "description": "Ezin Dataset",
        "url": "https://github.com/christopherohit",
        "version": "0.1.0",
        "year": 2022,
        "contributor": "christopherohit",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    coco_output['licenses'] = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    # coco_output['categories'] = [{
    #     'id': 1,
    #     'name': "中性杆状核粒细胞",
    #     'supercategory': "粒細胞系",
    #     },
    #     {
    #         'id': 2,
    #         'name': '中性中幼粒细胞',
    #         'supercategory': '粒細胞系',
    #     },
    #                 .
    #                 .
    #                 .
    # ]

    # get the coco category from dict
    coco_output['categories'] = []
    for i in tqdm(range(len(name_supercategory_dict))):
        category = {'id': i + 1,
                    'name': list(name_supercategory_dict)[i],
                    'supercategory': name_supercategory_dict[list(name_supercategory_dict)[i]]
                    }
        coco_output['categories'].append(category)
    coco_output['images'] = []
    coco_output['annotations'] = []
    ##########################################################################################################

    ann = json.load(open(annpath, encoding="utf-8"))
    # annotations id start from zero
    ann_id = 0
    # in VIA annotations, [key]['filename'] are image name
    for img_id, key in tqdm(enumerate(ann.keys())):
          filename = ann[key]['name']
          img = cv2.imread(imgdir + filename)
          # make image info and storage it in coco_output['images']
          image_info = create_image_info(img_id, os.path.basename(filename), img.shape[:2])
          coco_output['images'].append(image_info)

          regions = ann[key]["regions"]
          # for one image ,there are many regions,they share the same img id
          for region in regions:
              cate = region['class']
              # cate must in categories
              assert cate in [i['name'] for i in coco_output['categories']]
              # get the cate_id
              cate_id = 0
              for category in coco_output['categories']:
                  if cate == category['name']:
                      cate_id = category['id']
              ####################################################################################################

              iscrowd = 0
              points_x = region['all_x']
              points_y = region['all_y']
              area = 0

              min_x = min(points_x)
              max_x = max(points_x)
              min_y = min(points_y)
              max_y = max(points_y)
              box = [min_x, min_y, max_x - min_x, max_y - min_y]
              segmentation = get_segmenation(points_x, points_y)

              # make annotations info and storage it in coco_output['annotations']
              ann_info = create_annotation_info(ann_id, img_id, cate_id, iscrowd, area, box, segmentation)
              coco_output['annotations'].append(ann_info)
              ann_id = ann_id + 1

    return coco_output


def train_val_split(annos_name, original_dir, train_dir, val_dir, move):
    """
    It takes in the annotations file, the original directory, the train directory, the validation
    directory, and a boolean value that determines whether or not to move the images to the train and
    validation directories. 
    
    It then returns the train and validation annotations. 
    
    The function first loads the annotations file and then filters out the annotations that don't have
    any regions. 
    
    It then gets the images in the annotations and then randomly selects 20% of the images to be in the
    validation set. 
    
    It then creates the train and validation annotations and then moves the images to the train and
    validation directories if the move parameter is set to True. 
    
    If the move parameter is set to False, then it just creates the train and validation annotations.
    
    :param annos_name: the name of the annotation file
    :param original_dir: the directory where the original images and annotations are stored
    :param train_dir: The directory where the training images are stored
    :param val_dir: the directory where you want to save the validation images
    :param move: whether to move the images to the train and val folders
    :return: train_annos, val_annos
    """
    annotations = json.load(open(original_dir + annos_name, encoding="utf-8"))
    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annotations = [a for a in annotations if a['regions']]

    # get images in annotation
    total_images = [a['name'] for a in annotations]

    # image index that will move to val
    val_index = np.random.choice(len(annotations), size=len(annotations) // 6, replace=False).tolist()
    train_index = [i for i in range(len(annotations))]

    for i in val_index:
        train_index.remove(i)

    # create train, val annos
    val_annos = {}
    train_annos = {}
    # move images to train, val folder
    if move:
        shutil.rmtree(val_dir)
        os.mkdir(val_dir)
        shutil.rmtree(train_dir)
        os.mkdir(train_dir)
        for i in tqdm(val_index):
            shutil.copyfile(original_dir + total_images[i],
                            val_dir + total_images[i])
            val_annos[annotations[i]['name']] = annotations[i]
        for i in tqdm(train_index):
            shutil.copyfile(original_dir + total_images[i],
                            train_dir + total_images[i])
            train_annos[annotations[i]['name']] = annotations[i]
    # not move images to train, val folder
    else:
        for i in tqdm(val_index):
            val_annos[annotations[i]['name']] = annotations[i]
        for i in tqdm(train_index):
            train_annos[annotations[i]['name']] = annotations[i]

    return train_annos, val_annos




Train_via_annos, Val_via_annos = train_val_split(ANNOTATIONS_NAME,CAR_DATASET_ORIGINAL_IMG_PATH,
                                                     CAR_DATASET_TRAIN_IMG_PATH, CAR_DATASET_VAL_IMG_PATH, move=True)
with open(CAR_DATASET_TRAIN_IMG_PATH + '0Train_via_annos.json', 'w', encoding="utf-8") as outfile:
    json.dump(Train_via_annos, outfile, sort_keys=True, indent=4, ensure_ascii=False)

with open(CAR_DATASET_VAL_IMG_PATH + '0Val_via_annos.json', 'w', encoding="utf-8") as outfile:
    json.dump(Val_via_annos, outfile, sort_keys=True, indent=4, ensure_ascii=False)

COCO_train_annos = convert(ANNOTATIONS_NAME, CAR_DATASET_TRAIN_IMG_PATH,
                            CAR_DATASET_TRAIN_IMG_PATH + '0Train_via_annos.json')
COCO_val_annos = convert(ANNOTATIONS_NAME, CAR_DATASET_VAL_IMG_PATH,
                          CAR_DATASET_VAL_IMG_PATH + '0Val_via_annos.json')

# save COCO annotations
with open(CAR_DATASET_TRAIN_IMG_PATH + '0COCO_train_annos.json', 'w', encoding="utf-8") as outfile:
    json.dump(COCO_train_annos, outfile, sort_keys=True, indent=4, ensure_ascii=False)

with open(CAR_DATASET_VAL_IMG_PATH + '0COCO_val_annos.json', 'w', encoding="utf-8") as outfile:
    json.dump(COCO_val_annos, outfile, sort_keys=True, indent=4, ensure_ascii=False)