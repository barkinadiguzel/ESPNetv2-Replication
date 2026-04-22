class ESPNetV2Config:
    num_classes = 1000
    in_channels = 3

    K = 4
    groups = 4

    stem_channels = 16
    stage_channels = [32, 64, 128, 256]
    stage_repeats = [1, 3, 7, 3]
    head_channels = 1024
