# VideoABC
Created by [Wenliang Zhao](https://wl-zhao.github.io/), [Yongming Rao](https://raoyongming.github.io/), [Yansong Tang](https://andytang15.github.io/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1)

This repository contains PyTorch implementation for VideoABC: A Real-World Video Dataset for Abductive Visual Reasoning (TIP 2022).

[[IEEE Xplore]](https://ieeexplore.ieee.org/document/9893026)

## Dataset
VideoABC is built from the COIN dataset. Please download the videos from [COIN](https://coin-dataset.github.io/), and place them in the `data` folder.
```
├── configs
├── data
│   ├── ...
│   ├── A129SM9S54A
│   ├── a158bYDFSwU
│   ├── A1AjNjJFGs4
│   └── ...
├── metadata
├── README.md
├── slowfast
├── tools
└── ...
```
The `metadata` folder contains the VideoABC question/choice pairs as well as the train/test split.

## Training
```
python tools/run_net.py --cfg configs/VideoABC/SLOWFAST_HDR.yaml
```

## Evaluation
```
python tools/run_net.py --cfg configs/VideoABC/SLOWFAST_HDR.yaml TEST.CHECKPOINT_FILE_PATH path/to/checkpoint TEST.ENABLE True
```

## Acknowledgement
Our code is based on [SlowFast](https://github.com/facebookresearch/SlowFast). Our dataset is based on [COIN](https://coin-dataset.github.io/).
## Citation
If you find our work useful in your research, please consider citing:
```
@article{zhao2022videoabc,
  title={VideoABC: A Real-World Video Dataset for Abductive Visual Reasoning},
  author={Zhao, Wenliang and Rao, Yongming and Tang, Yansong and Zhou, Jie and Lu, Jiwen},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={6048--6061},
  year={2022},
  publisher={IEEE}
}
```
