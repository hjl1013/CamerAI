# CamerAI
## How to run our model

First, `cd` into `/home/aistore17/Final_submission/utils/final`. Then run

    python3 main.py --weights /home/aistore17/results/yolov7_400ep.pt -cfc 0.93 -cfs 0.8 -nfa 10 -msc 3 \
    --out-path ../../results --source /home/aistore17/Datasets/4.TestVideosSample-2/1

Running this will save a video with change log written on top of center camera video
and a csv file of change log with its corresponding time. Default parameters output the
best results.

Change --source argument to `/home/aistore17/Datasets/4.TestVideosSample-2/{2~6}` to run video sample 2~6.
This folder has to have only 5 camera videos in it with the format of `#_center.mp4`, `#_left_back.mp4`,
`#_left_front.mp4`, `#_right_back.mp4`, `#_right_front.mp4`. # indicates the number of video.

The output file path is `/home/aistore17/Final_submission/results`

There are some more arguments you can use to run `main.py`

* --weights : type=str, default = '/home/aistore17/results/yolov7_400ep.pt', yolo model to use
* --iou-thres : type=float, default=0.45
* --source : type=str, default='/home/aistore17/Datasets/4.TestVideosSample-2/1', help='0 for webcams, else directory path to videos'
* --img-size : type=int, default=640, help='inference size (pixels)'
* --output-path : type=str, default='../../results', help='path to save results'
* --save-vid : type=bool, default=True, help='whether to save video as output'
* --save-csv : type=bool, default=True, help='whether to save csv file as output'
* -cfc, --conf-thres-center : type=float, default=0.25, confidence threshold of center camera
* -cfs, --conf-thres-side : type=float, default=0.25, confidence threshold of side cameras
* -nfa, --num-frames-to-avg : type=int, default=5, help='number of frames to average'
* -msc, --max-stable-count : type=int, default=5, help='maximum counts for results to stay stable'
