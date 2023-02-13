from __future__ import unicode_literals
import json

ANNOTATIONS_NAME_MERGE = "multiclass_via_region_data_merge.json"
ANNOTATIONS_NAME_1 = "train_multiclass.json"
ANNOTATIONS_NAME_2 = "val_multiclass.json"


def merge2json(annos_name_1, annos_name_2):
    """
    It takes two JSON files, merges them, and returns a dictionary of the merged JSON
    
    :param annos_name_1: The first JSON file you want to merge
    :param annos_name_2: the name of the json file you want to merge with the first one
    :return: A dictionary of the merged annotations.
    """
    annos_1 = json.load(open(annos_name_1, encoding="utf-8"))
    annos_2 = json.load(open(annos_name_2, encoding="utf-8"))
    annos_merge = {**annos_1, **annos_2}

    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annos_merge = list(annos_merge.values())
    annos_merge = [a for a in annos_merge if a['regions']]
    annos_out = {}
    for i in annos_merge:
        annos_out[i['filename']] = i

    return annos_out


annos_out = merge2json(ANNOTATIONS_NAME_1, ANNOTATIONS_NAME_2)
# annos_out = json.dumps(annos_out, sort_keys=True, indent=4, ensure_ascii=False)
with open(ANNOTATIONS_NAME_MERGE, 'w', encoding="utf-8") as outfile:
    json.dump(annos_out, outfile, sort_keys=True, indent=4, ensure_ascii=False)