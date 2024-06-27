import threading as th

class LayerConnection():
    def __init__(self, a, b):
        self.type_ = "connection"
        self.weight = 1
        self.a = a
        self.b = b

        th.Thread(target=self.loop).start()

    def loop(self, a=None):
        if a == None:
            a = self.a
            
        else:
            if 

class LayerPoint():
    def __init__(self):
        self.type_ = "point"
        self.bias = 0

    def 

class Input():
