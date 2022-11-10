import sys
sys.path.append('/home/aistore17/CamerAI_hjl')
sys.path.append('/home/aistore17/CamerAI_hjl/yolov7')

import argparse
import torch
import time
import numpy as np
import cv2
from pathlib import Path

from models.experimental import attempt_load
from utils.general import check_imshow, check_img_size, non_max_suppression
from utils.datasets import LoadStreams, LoadImages


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

# style of text when writing onto image
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.3
fontColor = (255,255,255)
thickness = 1
lineType = 2


def main(args):
    source, weights, device_name, imgsz = args.source, args.weights, args.device, args.img_size
    save_path = '/home/aistore17/CamerAI_hjl/yolov7/results/result2.mp4'

    # only implementing with device GPU
    assert device_name.split(':')[0] == 'cuda', f'{device_name} not implemented'
    device = torch.device(device_name)
    webcam = source == '0'

    # load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    model.half()

    if webcam:
        raise NotImplementedError
    else:
        # define dataset from paths of videos
        cam_vid_paths = [str(path) for path in Path(source).iterdir()]
        datasets = [LoadImages(vid, img_size=imgsz, stride=stride) for vid in cam_vid_paths]

        # run detection
        vid_writer = None
        for datas in zip(datasets[0], datasets[1], datasets[2], datasets[3], datasets[4]):
            start = time.time()

            # list results saves the number of each product
            # TODO: stack images
            results = np.zeros(60)
            for data in datas:
                result_tmp = np.zeros(60)

                # get prediction from model(yolov7)
                with torch.no_grad():
                    path, img, im0s = data[:-1]
                    img = torch.from_numpy(img).to(device)
                    img = img.half()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = model(img)[0]
                    pred = non_max_suppression(pred, args.conf_thres, args.iou_thres)

                # calculating union of products
                for det in pred[0]:
                    det = det.to('cpu')
                    result_tmp[int(det[-1])] += 1
                results = np.maximum(results, result_tmp)

            # saving result as a video
            _, _, img, vid_cap = datas[0]
            if not isinstance(vid_writer, cv2.VideoWriter):
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            for i, num in enumerate(results):
                bottomLeftCornerOfText = (5 + (i % 10) * 50, 10 * (i // 10 + 1))
                cv2.putText(img, f'{classes[i][0:4]}:{int(num):02d}',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
            vid_writer.write(img)

            print(time.time()-start)
        vid_writer.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/aistore17/results/yolov7_200ep_best.pt', help='yolo model to use')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to run')
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--source', type=str, default='/home/aistore17/Datasets/4.TestVideosSample', help='0 for webcams, else directory path to videos')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    args = parser.parse_args()

    main(args)