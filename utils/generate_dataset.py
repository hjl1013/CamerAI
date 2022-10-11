import numpy as np
import cv2
import random
from pathlib import Path

def combine_image_with_background(image, background):
    mask = (np.sum(image <= 5, axis=2) == 0)[:, :, None]
    return background * (1 - mask) + image * mask

def generate_image_file(fout, fnames, background_fname):
    image_result = cv2.imread(background_fname)

    for f in fnames:
        image = cv2.imread(f + '.jpg')
        image_result = combine_image_with_background(image, image_result)

    cv2.imwrite(fout, image_result)
    print('generated ' + fout)

def generate_label_file(fout, fnames):
    lines = []
    for fname in fnames:
        with open(fname + '.txt', 'r') as fr:
            line = fr.readline()
            if len(line) < 3:
                return False
            lines.append(line.rstrip('\n'))

    with open(fout, 'w') as fw:
        for line in lines:
            fw.write(line)
            fw.write('\n')

    print('generated ' + fout)
    return True

def generate_dataset():

    num_datas = 100000
    png_root = '/home/aistore17/Datasets/2.backsub_images_100'
    background_root = '/home/aistore17/Datasets/3.Background_Videos/images'
    out_root = '/home/aistore17/Datasets/NewDataset'
    # out_root = '/home/aistore17/Datasets/tmp'
    dir_names = ['15.redbull', '25.clif_crunch_peanut_butter', '17.a1_steak_sauce', '30.coffee_mate_french_vanilla', '36.nature_valley_crunchy_oats_n_honey', '51.campbells_chunky_classic_chicken_noodle', '8.hunts_sauce', '24.reeses_pieces', '50.hersheys_symphony', '7.honey_bunches_of_oats_with_almonds', '41.nabisco_nilla', '55.david_sunflower_seeds', '28.quaker_big_chewy_chocolate_chip', '13.cheeze_it', '43.campbells_chicken_noodle_soup', '3.cholula_chipotle_hot_sauce', '0.aunt_jemima_original_syrup', '2.bumblebee_albacore', '37.ritz_crackers', '52.martinellis_apple_juice', '19.cinnamon_toast_crunch', '44.frappuccino_coffee', '9.listerine_green', '42.pepperidge_farm_milano_cookies_double_chocolate', '1.band_aid_clear_strips', '29.spam', '4.crayola_24_crayons', '45.chewy_dips_chocolate_chip', '38.palmolive_orange', '22.haribo_gold_bears_gummi_candy', '58.coca_cola_bottle', '27.pop_trarts_strawberry', '40.tapatio_hot_sauce', '47.nature_vally_fruit_and_nut', '6.honey_bunches_of_oats_honey_roasted', '39.crystal_hot_sauce', '31.pepperidge_farm_milk_chocolate_macadamia_cookies', '46.chewy_dips_peanut_butter', '11.white_rain_body_wash', '34.toblerone_milk_chocolate', '33.snickers', '16.mom_to_mom_sweet_potato_corn_apple', '59.twix', '35.clif_z_bar_chocolate_chip', '18.jif_creamy_peanut_butter', '23.bulls_eye_bbq_sauce_original', '32.kitkat_king_size', '57.act_ii_butter_lovers_popcorn', '49.lindt_excellence_cocoa_dark_chocolate', '10.mahatma_rice', '26.mom_to_mom_butternut_squash_pear', '5.hersheys_cocoa', '48.cheerios', '12.pringles_bbq', '56.monster_energy', '53.dove_pink', '21.dr_pepper', '14.hersheys_bar', '20.arm_hammer_baking_soda', '54.dove_white']

    for i in range(num_datas):
        image_file_name = out_root + '/' + str(i) + '.jpg'
        label_file_name = out_root + '/' + str(i) + '.txt'
        n = random.randint(1, 3)
        background_file_name = str(random.choice(list(Path(background_root).iterdir())))

        file_names = []
        for j in range(n):
            dir = random.choice(dir_names)

            while True:
                file = str(random.choice(list(Path(png_root + '/' + dir).iterdir()))).split('.jpg')[0].split('.txt')[0]

                if not (file in file_names):
                    break

            file_names.append(str(file))

        print(file_names)
        ret = generate_label_file(label_file_name, file_names)
        if ret:
            generate_image_file(image_file_name, file_names, background_file_name)
        print()

if __name__ == '__main__':
    generate_dataset()