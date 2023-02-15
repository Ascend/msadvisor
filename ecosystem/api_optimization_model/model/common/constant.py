#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from enum import Enum

#数据类型
class aclvencChannelDescParamType(Enum):
    ACL_VENC_THREAD_ID_UINT64=0,
    ACL_VENC_CALLBACK_PTR=1,
    ACL_VENC_PIXEL_FORMAT_UINT32=2,
    ACL_VENC_ENCODE_TYPE_UINT32=3,
    ACL_VENC_PIC_WIDTH_UINT32=4,
    ACL_VENC_PIC_HEIGHT_UINT32=5,
    ACL_VENC_KEY_FRAME_INTERVAL_UINT32=6,
    ACL_VENC_BUF_ADDR_PTR=7,
    ACL_VENC_BUF_SIZE_UINT32=8,
    ACL_VENC_RC_MODE_UINT32=9,
    ACL_VENC_SRC_RATE_UINT32=10,
    ACL_VENC_MAX_BITRATE_UINT32=11,
    ACL_VENC_MAX_IP_PROP_UINT32=12