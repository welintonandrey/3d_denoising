# based on: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
# Thx StackOverflow ;)

import time
import sys
import numpy as np

def printProgressBar(count, total, bar_len = 30, prefix = 'Progress: '):
    filled_len = int(round(bar_len * count / float(total)))

    percents = (100.0 * count / float(total))
    bar = '#' * filled_len + ' ' * (bar_len - filled_len)

    sys.stdout.write('%s[%s] %.2lf%s\r' % (prefix, bar, percents, '%'))
    sys.stdout.flush()
