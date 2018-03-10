import time

class Timer(object):
    def __init__(self, msg):
        super(Timer, self).__init__()
        self.msg = '[{0:.3f} sec] ' + msg

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, type, value, traceback):
        t = time.clock() - self.start
        print(self.msg.format(t))
