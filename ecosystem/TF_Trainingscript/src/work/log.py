from advisor import Advisor
from work.flag import Type

def log(plogdata,result):
    if plogdata != []:
        advisor = Advisor('plog', plogdata[0][0])
        # EVENT�����plog�в�������Use fusion library value��
        advisor.work('Use fusion library value', 'Gradient segmentation', Type.LOG, plogdata[0][1], result)