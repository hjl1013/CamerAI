import os
import argparse

from detectron2.config import get_cfg, LazyConfig
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from utils.data.register_camerai_data import register_camerai

def train(args):
    # get config file and modify
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    if args.weights is None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x")
    else:
        cfg.MODEL.WEIGHTS = args.weights
    if args.learning_rate is not None:
        cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.OUTPUT_DIR = args.out_dir

    # train
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="/home/aistore17/CamerAI_hjl/detectron2/configs/CamerAI/CamerAI_retinanet.yaml",
                        help='config file')
    parser.add_argument('-w', '--weights', type=str, default=None, help='model weights')
    parser.add_argument('-lr', '--learning-rate', type=float, default=None, help='learning rate')
    parser.add_argument('-mp', '--max-epochs', type=int, default=270000, help='maximum epoch')
    parser.add_argument('--out-dir', type=str, default='./output/detectron2', help='save path')
    args = parser.parse_args()

    # register camerai format data
    register_camerai()
    train(args)