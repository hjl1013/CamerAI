import argparse
import os
from subprocess import call

def run(args):
    cnt = 0
    while os.path.exists(os.path.join(args.out_path, f"exp{cnt}")):
        cnt += 1

    out_path = os.path.join(args.out_path, f"exp{cnt}")
    for i in range(1, 7):
        call(["python3", "main.py", "--weights", f"{args.weights}", "--iou-thres", f"{args.iou_thres}",
              "--source", f"{args.source_root}/{i}", "--img-size", f"{args.img_size}", "--out-path", f"{out_path}",
              "--save-vid", f"{args.save_vid}", "--save-csv", f"{args.save_csv}", "-cfc", f"{args.conf_thres_center}",
              "-cfs", f"{args.conf_thres_side}", "-msc", f"{args.max_stable_count}"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/aistore17/results/yolov7_400ep.pt',
                        help='yolo model to use')
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--source-root', type=str, default='/home/aistore17/Datasets/4.TestVideosSample',
                        help='0 for webcams, else directory path to videos')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--out-path', type=str, default='../../results', help='path to save resulting video')
    parser.add_argument('--save-vid', type=bool, default=True, help='whether to save video as output')
    parser.add_argument('--save-csv', type=bool, default=True, help='whether to save csv file as output')
    parser.add_argument('-cfc', '--conf-thres-center', type=float, default=0.93)
    parser.add_argument('-cfs', '--conf-thres-side', type=float, default=0.6)
    parser.add_argument('-nfa', '--num-frames-to-avg', type=int, default=5, help='number of frames to average')
    parser.add_argument('-msc', '--max-stable-count', type=int, default=6,
                        help='maximum counts to stay stable. nfa * msc number of frames should stay stable')
    args = parser.parse_args()

    run(args)