from en import Advice as En
from zh import Advice as Zh

Advice = None


def set_locale(locale):
    global Advice
    if locale == 'zh':
        Advice = Zh
    else:
        Advice = En
