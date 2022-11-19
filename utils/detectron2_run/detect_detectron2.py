import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[2]))

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from utils.data.register_camerai_data import register_camerai

def detect(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_thres  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("camerai_val", output_dir=args.output_dir)
    val_loader = build_detection_test_loader(cfg, "camerai_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default="/home/aistore17/CamerAI_hjl/detectron2/configs/CamerAI/CamerAI_retinanet.yaml",
                        help='config file')
    parser.add_argument('--weights', type=str, required=True, help='path to model')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--output-dir', type=str, default="./output/test", help='output to save result')
    args = parser.parse_args()

    register_camerai()
    detect(args)