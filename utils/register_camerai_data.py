import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

def get_camerai_dicts(img_dir, json_file):
    with open(json_file) as f:
        imgs_anns = json.load(f)
        image_infos = imgs_anns["images"]
        annotations = imgs_anns["annotations"]

    anno_idx = 0
    total_anno = len(annotations)
    dataset_dicts = []
    for dict in image_infos:
        record = {}

        record["file_name"] = os.path.join(img_dir, dict["file_name"])
        record["image_id"] = dict["id"]
        record["height"] = dict["height"]
        record["width"] = dict["width"]

        objs = []
        while anno_idx < total_anno:
            annotation = annotations[anno_idx]
            if annotation["image_id"] == dict["id"]:
                obj = annotation.copy()
                obj["category_id"] -= 1
                obj["bbox_mode"] = BoxMode.XYWH_ABS
                objs.append(obj)
                anno_idx += 1
            else:
                break

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def register_camerai():
    classes = ['aunt_jemima_original_syrup', 'band_aid_clear_strips', 'bumblebee_albacore',
               'cholula_chipotle_hot_sauce', 'crayola_24_crayons', 'hersheys_cocoa',
               'honey_bunches_of_oats_honey_roasted', 'honey_bunches_of_oats_with_almonds',
               'hunts_sauce', 'listerine_green', 'mahatma_rice', 'white_rain_body_wash', 'pringles_bbq',
               'cheeze_it', 'hersheys_bar', 'redbull', 'mom_to_mom_sweet_potato_corn_apple',
               'a1_steak_sauce', 'jif_creamy_peanut_butter', 'cinnamon_toast_crunch',
               'arm_hammer_baking_soda', 'dr_pepper', 'haribo_gold_bears_gummi_candy',
               'bulls_eye_bbq_sauce_original', 'reeses_pieces', 'clif_crunch_peanut_butter',
               'mom_to_mom_butternut_squash_pear', 'pop_trarts_strawberry', 'quaker_big_chewy_chocolate_chip',
               'spam', 'coffee_mate_french_vanilla', 'pepperidge_farm_milk_chocolate_macadamia_cookies',
               'kitkat_king_size', 'snickers', 'toblerone_milk_chocolate', 'clif_z_bar_chocolate_chip',
               'nature_valley_crunchy_oats_n_honey', 'ritz_crackers', 'palmolive_orange', 'crystal_hot_sauce',
               'tapatio_hot_sauce', 'nabisco_nilla', 'pepperidge_farm_milano_cookies_double_chocolate',
               'campbells_chicken_noodle_soup', 'frappuccino_coffee', 'chewy_dips_chocolate_chip',
               'chewy_dips_peanut_butter', 'nature_vally_fruit_and_nut', 'cheerios',
               'lindt_excellence_cocoa_dark_chocolate', 'hersheys_symphony', 'campbells_chunky_classic_chicken_noodle',
               'martinellis_apple_juice', 'dove_pink', 'dove_white', 'david_sunflower_seeds',
               'monster_energy', 'act_ii_butter_lovers_popcorn', 'coca_cola_bottle', 'twix']

    for d in ["train", "val"]:
        DatasetCatalog.register("camerai_" + d,
                                lambda d=d: get_camerai_dicts("/home/aistore17/Datasets/cocoformat_dataset/" + d,
                                                              f"/home/aistore17/Datasets/cocoformat_dataset/annotations/instances_{d}2017.json"))
        MetadataCatalog.get("camerai_" + d).set(thing_classes=classes)