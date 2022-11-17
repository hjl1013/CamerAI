# CamerAI
### How to run `utils/final/main.py` 
    CUDA_VISIBLE_DEVICES=6 python3 main.py --conf-thres 0.8 --gpu-id 6 -nfa 10 -msc 3
#### This is the main python file of the project. 
The `main.py` has some arguments 
--weights : type=str, default = /results/yolov7_200ep_best.pt', yolo model to use
--gpu-id : type=str, default='0', help='device to run'
--conf-thres : type=float, default=0.25
--iou-thres : type=float, default=0.45
--source : type=str, default='/home/aistore17/Datasets/4.TestVideosSample', help='0 for webcams, else directory path to videos'
--img-size : type=int, default=640, help='inference size (pixels)'
--save-path : type=str, default='../results/result.mp4', help='path to save resulting video'
-nf, --num-frames-to-avg : type=int, default=5, help='number of frames to average'
