from pyo import *
import numpy as np

s = Server().boot()
s.start()

ir = SndTable('data/IR.BottleHall.wav')

x = []
x.append(Osc(SawTable(), [100, 101]))
x.append(Convolve(x[-1], ir, 2200))
x[-1].out()

s.gui(locals())
