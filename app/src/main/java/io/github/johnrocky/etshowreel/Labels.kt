package io.github.johnrocky.etshowreel

/** COCO 2017 categories in the contiguous 0-79 order the DETR heads emit. */
val COCO_LABELS =
    listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
    )

/** Cityscapes 19-class training labels, in the order PIDNet emits. */
val CITYSCAPES_LABELS =
    listOf(
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
        "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
        "motorcycle", "bicycle",
    )

/** The official Cityscapes colours, so the output reads the same as published benchmarks. */
val CITYSCAPES_PALETTE =
    intArrayOf(
        0x804080, 0xF423E8, 0x464646, 0x66339C, 0xBE9999, 0x999999, 0xFAAA1E, 0xDCDC00,
        0x6B8E23, 0x98FB98, 0x4682B4, 0xDC143C, 0xFF0000, 0x00008E, 0x000046, 0x003C64,
        0x005064, 0x0000E6, 0x770B20,
    )
