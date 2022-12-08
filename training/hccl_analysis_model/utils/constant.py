

class Constant:
    """
    Constant of HcclAnalysisTool
    """
    # communication link
    RDMA = "RDMA"
    PCIE = "PCIE"
    HCCS = "HCCS"
    SDMA = "SDMA"
    # pattern for hccl info and step trace file
    HCCL_DIR_FORMAT = "hccl_info_"
    STEP_TRACE_FILE = "step_trace_raw_*_detail_time.csv"
    # HcclAnalysisTool Optional List
    SDMA_TRANSIT_ITEMS = ["Memcpy", "Reduce Inline"]
    TRANSIT_TYPE = ["RDMA", "PCIE", "HCCS"]
    TRANSPORT_TYPE_DICT = {"None": 0, "HCCS": 1, "PCIE": 2, "RDMA": 3}
    TRANSPORT_TYPE_BW_DICT = {"SDMA": 1000000, "HCCS": 18, "PCIE": 20, "RDMA": 12.5}
    COMMUNICATIONS_OPS = ["AllReduce", "ReduceScatter", "AllGather", "Broadcast", "NeighborExchange"]

    LOCAL = "LOCAL"
    NOTIFY_WAIT = "Notify Wait"
    REDUCE_TBE = "Reduce TBE"
    RDMA_SEND = "RDMASend"

    # HcclAnalysisTool hyperparameter
    RDMA_TRANSIT_OP_NUM = 5
    WAIT_TIME_THRESHOLD = 0.2
    # ANALYSIS_STEP_NUM = 1
    DEFAULT_STEP_NUM = 1
    LINK_BANDWIDTH = {"HCCS": 18, "PCIE": 20, "RDMA": 12.5}
    MESSAGE_SIZE_THRESHOLD = {"HCCS": 32, "PCIE": 32, "RDMA": 0.5}   # size:MB HCCS、PCIE阈值可能需要调小一点
    LARGE_MESSAGE_RATE = 0.8
    BANDWIDTH_THRESHOLD = 0.8

    RANK_NUM_PER_SERVER = 8
    RANK_NUM_PER_OS = 4

    # Unit conversion
    B_TO_G = 1024 ** 3
    US_TO_MS = 1000

    # communication link info keys
    TRANSPORT_TYPE_ = "transport_type"
    TRANSIT_SIZE = "transit_size"
    TRANSIT_TIME = "transit_time"
    WAIT_TIME = "wait_time"

    # profiling data keys
    # hccl iter*.trace keys
    DEVICE_ID = 'device id'
    ITERATION = 'iteration'
    TRACEEVENTS = 'traceEvents'
    # traceEvents keys
    TID = 'tid'
    PID = 'pid'
    TS = 'ts'
    DUR = 'dur'
    PH = 'ph'
    NAME = 'name'
    ARGS = 'args'
    # args keys
    NOTIFY_ID = 'notify id'
    DUR_EST = 'duration estimated'
    STAGE = 'stage'
    STEP = 'step'
    BANDWIDTH = 'bandwidth'
    STREAM_ID = 'stream id'
    TASK_ID = 'task id'
    TASK_TYPE = 'task type'
    SRC_RANK = 'src rank'
    DST_RANK = 'dst rank'
    TRANSPORT_TYPE = 'transport type'
    SIZE = 'size'
    # check data type
    NULL = 'NULL'
    # timeline task type keys
    AICPU = 'aicpu'
    AICORE = 'aicore'
    HOSTCPU = 'hostcpu'
    COMMUNICATION = 'communication'

    # matrix type
    BANDWIDTH_UTILIZATION = "bandwidth_utilization"
    # return value
    HCCL_ANALYSIS_ERROR = -1
    HCCL_ANALYSIS_OK = 1
    DATA_PARSE_OK = 1
    DATA_PARSE_ERROR = -1

    # msadvisor type of identified result
    SUCCESS = 'success'
    MODEL = 'model'
    OP = 'op'
    CLASS_TYPE = {'op': '0', 'model': '1'}
    ERROR_CODE = {'success': '0', 'optimized': '1'}
    EXTEND_TYPE = {'list': '0', 'table': '1', 'sourcedata': '2'}
    EXTEND_DATA_TYPE = {'str': '0', 'int': '1', 'double': '2'}
    # extend datatype
    STR = 'str'
    DOUBLE = 'double'
    # extend type
    LIST = 'list'
    TABLE = 'table'
