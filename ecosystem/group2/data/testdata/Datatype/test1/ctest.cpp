enum aclvencChannelDescParamType {
    ACL_VENC_THREAD_ID_UINT64 = 0,      //回调线程ID，该属性类型对应的值为uint64_t类型
    ACL_VENC_CALLBACK_PTR=1,              //回调函数，该属性类型对应的值是内存指针
    ACL_VENC_PIXEL_FORMAT_UINT32=2,       //输入图像格式，该属性类型对应的值为uint32_t类型
    ACL_VENC_ENCODE_TYPE_UINT32=3,        //视频编码协议，该属性类型对应的值为uint32_t类型
    ACL_VENC_PIC_WIDTH_UINT32=4,          //输入图片宽度，该属性类型对应的值为uint32_t类型
    ACL_VENC_PIC_HEIGHT_UINT32=5,         //输入图片高度，该属性类型对应的值为uint32_t类型
    ACL_VENC_KEY_FRAME_INTERVAL_UINT32=5, //关键帧间隔，该属性类型对应的值为uint32_t类型
    ACL_VENC_BUF_ADDR_PTR=6,              //编码输出缓存地址
    ACL_VENC_BUF_SIZE_UINT32=7,           //编码输出缓存大小，该属性类型对应的值为uint32_t类型
    ACL_VENC_RC_MODE_UINT32=8,            //码率控制模式，该属性类型对应的值为uint32_t类型
    ACL_VENC_SRC_RATE_UINT32=9,           //输入码流帧率，该属性类型对应的值为uint32_t类型
    ACL_VENC_MAX_BITRATE_UINT32=10,        //输出码率，该属性类型对应的值为uint32_t类型
    ACL_VENC_MAX_IP_PROP_UINT32=11       //一个GOP内单个I帧bit数和单个P帧bit数的比例，该属性类型对应的值为uint32_t类型
};