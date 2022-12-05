#task1 NpuFusedSGD/NpuFusedAdam
# 原始代码
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)   


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, momentum=args.momentum)   

# 优化后代码
optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), lr=args.lr, momentum=args.momentum)   



#task2 amp.initialize(*combine_grad=True)

model, optimizer = amp.initialize(model, optimizer, opt_level='O2', loss_scale=32.0)

model, optimizer = amp.initialize(
    model, 
    optimizer, 
    opt_level='O2', 
    loss_scale=32.0
)


model, optimizer = amp.initialize(model, optimizer, opt_level='O2', loss_scale=32.0, combine_grad=True)

model, optimizer = amp.initialize(
    model,
    optimizer, 
    opt_level='O2', 
    loss_scale=32.0, 
    combine_grad=True
)

# task3 算子替换
iou

ptiou

# nms

# single_level_responsible_flags

# yolo_bbox_coder

# delta_xywh_bbox_coder

# channel_shuffle

# Prefetcher

# Dropout

# LabelSmoothingCrossEntropy

# ROIAlign

# DCNv2

# LSTM


