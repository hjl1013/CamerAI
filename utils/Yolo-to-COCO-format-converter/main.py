from pathlib import Path

from create_annotations import (
    create_image_annotation,
    create_annotation_from_yolo_format,
    coco_format,
)
import cv2
import argparse
import json
import numpy as np
import imagesize

#################################################
# Change the classes depend on your own dataset.#
# Don't change the list name 'Classes'          #
#################################################

YOLO_DARKNET_SUB_DIR = "YOLO_darknet"

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


def get_images_info_and_annotations(opt):
    path = Path(opt.path)
    annotations = []
    images_annotations = []
    if path.is_dir():
        file_paths = sorted(path.rglob("*.jpg"))
        file_paths += sorted(path.rglob("*.jpeg"))
        file_paths += sorted(path.rglob("*.png"))
    else:
        with open(path, "r") as fp:
            read_lines = fp.readlines()
        file_paths = [Path(line.replace("\n", "")) for line in read_lines]

    image_id = 0
    annotation_id = 1  # In COCO dataset format, you must start annotation id with '1'

    for file_path in file_paths:
        # Check how many items have progressed
        print("\rProcessing " + str(image_id) + " ...", end='')

        # Build image annotation, known the image's width and height
        w, h = imagesize.get(str(file_path))
        image_annotation = create_image_annotation(
            file_path=file_path, width=w, height=h, image_id=image_id
        )
        images_annotations.append(image_annotation)

        label_file_name = f"{file_path.stem}.txt"
        if opt.yolo_subdir:
            annotations_path = file_path.parent / YOLO_DARKNET_SUB_DIR / label_file_name
        else:
            annotations_path = file_path.parent / label_file_name

        if not annotations_path.exists():
            continue  # The image may not have any applicable annotation txt file.

        with open(str(annotations_path), "r") as label_file:
            label_read_line = label_file.readlines()

        # yolo format - (class_id, x_center, y_center, width, height)
        # coco format - (annotation_id, x_upper_left, y_upper_left, width, height)
        for line1 in label_read_line:
            label_line = line1
            category_id = (
                int(label_line.split()[0]) + 1
            )  # you start with annotation id with '1'
            x_center = float(label_line.split()[1])
            y_center = float(label_line.split()[2])
            width = float(label_line.split()[3])
            height = float(label_line.split()[4])

            float_x_center = w * x_center
            float_y_center = h * y_center
            float_width = w * width
            float_height = h * height

            min_x = int(float_x_center - float_width / 2)
            min_y = int(float_y_center - float_height / 2)
            width = int(float_width)
            height = int(float_height)

            annotation = create_annotation_from_yolo_format(
                min_x,
                min_y,
                width,
                height,
                image_id,
                category_id,
                annotation_id,
                segmentation=opt.box2seg,
            )
            annotations.append(annotation)
            annotation_id += 1

        image_id += 1  # if you finished annotation work, updates the image id.

    return images_annotations, annotations


def debug(opt):
    path = opt.path
    color_list = np.random.randint(low=0, high=256, size=(len(classes), 3)).tolist()

    # read the file
    file = open(path, "r")
    read_lines = file.readlines()
    file.close()

    for line in read_lines:
        print("Image Path : ", line)
        # read image file
        img_file = cv2.imread(line[:-1])

        # read .txt file
        label_path = line[:-4] + "txt"
        label_file = open(label_path, "r")
        label_read_line = label_file.readlines()
        label_file.close()

        for line1 in label_read_line:
            label_line = line1

            category_id = label_line.split()[0]
            x_center = float(label_line.split()[1])
            y_center = float(label_line.split()[2])
            width = float(label_line.split()[3])
            height = float(label_line.split()[4])

            int_x_center = int(img_file.shape[1] * x_center)
            int_y_center = int(img_file.shape[0] * y_center)
            int_width = int(img_file.shape[1] * width)
            int_height = int(img_file.shape[0] * height)

            min_x = int_x_center - int_width / 2
            min_y = int_y_center - int_height / 2
            width = int(img_file.shape[1] * width)
            height = int(img_file.shape[0] * height)

            print("class name :", classes[int(category_id)])
            print("x_upper_left : ", min_x, "\t", "y_upper_left : ", min_y)
            print("width : ", width, "\t", "\t", "height : ", height)
            print()

            # Draw bounding box
            cv2.rectangle(
                img_file,
                (int(int_x_center - int_width / 2), int(int_y_center - int_height / 2)),
                (int(int_x_center + int_width / 2), int(int_y_center + int_height / 2)),
                color_list[int(category_id)],
                3,
            )

        cv2.imshow(line, img_file)
        delay = cv2.waitKeyEx()

        # If you press ESC, exit
        if delay == 27 or delay == 113:
            break

        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser("Yolo format annotations to COCO dataset format")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Absolute path for 'train.txt' or 'test.txt', or the root dir for images.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Visualize bounding box and print annotation information",
    )
    parser.add_argument(
        "--output",
        default="train_coco.json",
        type=str,
        help="Name the output json file",
    )
    parser.add_argument(
        "--yolo-subdir",
        action="store_true",
        help="Annotations are stored in a subdir not side by side with images.",
    )
    parser.add_argument(
        "--box2seg",
        action="store_true",
        help="Coco segmentation will be populated with a polygon "
        "that matches replicates the bounding box data.",
    )
    args = parser.parse_args()
    return args


def main(opt):
    output_name = opt.output
    output_path = "output/" + output_name

    print("Start!")

    if opt.debug is True:
        debug(opt)
        print("Debug Finished!")
    else:
        (
            coco_format["images"],
            coco_format["annotations"],
        ) = get_images_info_and_annotations(opt)

        for index, label in enumerate(classes):
            categories = {
                "supercategory": "Defect",
                "id": index + 1,  # ID starts with '1' .
                "name": label,
            }
            coco_format["categories"].append(categories)

        with open(output_path, "w") as outfile:
            json.dump(coco_format, outfile, indent=4)

        print("Finished!")


if __name__ == "__main__":
    options = get_args()
    main(options)
