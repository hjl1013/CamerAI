import argparse
import torch
import time
import numpy as np
import cv2
import sys
from pathlib import Path
import torch.multiprocessing as mp
sys.path.append(str(Path(__file__).absolute().parent.parent))
sys.path.append(str(Path(__file__).absolute().parent.parent / 'yolov7'))

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
    source, weights, device_name, imgsz = args.source, args.weights, f'cuda:{args.gpu_id}', args.img_size
    save_path, num_frames_to_avg = args.save_path, args.num_frames_to_avg

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
        start = time.time()
        inputs = []
        results = np.zeros(60)
        for cnt, datas in enumerate(zip(datasets[0], datasets[1], datasets[2], datasets[3], datasets[4])):
            print()

            # get predictions from 5 cameras averages results of {num_frames_to_avg} frames
            tmp = time.time()
            images = [data[1] for data in datas]
            if inputs is None:
                inputs = images
            else:
                inputs.extend(images)
            print(f'image preparing time: {time.time() - tmp}')

            if cnt % num_frames_to_avg == num_frames_to_avg - 1:
                # get prediction from model
                tmp = time.time()
                with torch.no_grad():
                    inputs = torch.from_numpy(np.array(inputs))
                    inputs = inputs.to(device)
                    inputs = inputs.half()
                    inputs /= 255.0

                    preds = model(inputs)[0] # tuple of predictions in all 5 cameras
                    print(f'model prediction time: {time.time() - tmp}')
                    preds = non_max_suppression(preds, args.conf_thres, args.iou_thres)
                print(f'model total prediction time: {time.time() - tmp}')

                # calculating final result
                results = np.zeros(60)
                tmp = time.time()
                for i in range(num_frames_to_avg):
                    result = np.zeros(60)

                    # calculating union of 5 cameras in one frame
                    for pred in preds[5 * i: 5 * (i+1)]:
                        result_tmp = np.zeros(60)
                        pred = pred.to('cpu')
                        for det in pred:
                            result_tmp[int(det[-1])] += 1
                        result = np.maximum(result, result_tmp)

                    # averageing result
                    results += result / num_frames_to_avg
                results = results.round().astype(np.int32)

                print(f'results calculating time: {time.time() - tmp}')
                inputs = None

            # saving result as a video
            tmp = time.time()
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
            print(f'video saving time: {time.time() - tmp}')
            print(time.time()-start)
        vid_writer.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/aistore17/results/yolov7_200ep_best.pt', help='yolo model to use')
    parser.add_argument('--gpu-id', type=str, default='0', help='device to run')
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--source', type=str, default='/home/aistore17/Datasets/4.TestVideosSample', help='0 for webcams, else directory path to videos')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--save-path', type=str, default='../results/result.mp4', help='path to save resulting video')
    parser.add_argument('-nf', '--num-frames-to-avg', type=int, default=5, help='number of frames to average')
    args = parser.parse_args()

    main(args)