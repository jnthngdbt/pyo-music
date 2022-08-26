
from pyo import *

s = Server().boot()
s.start()

m = Cloud(density=6, poly=5).play()
m.ctrl()

t = ExpTable([(0, 0), (100, 1), (8192, 0)])
t.graph()

m0 = 109+5
f = midiToHz([m0, m0+5, m0+7, m0+12])
print (f)

tr = TrigEnv(m, t, dur=4, mul=.01)
a = Sine(freq=f, mul=tr)
a.ctrl()

b = Freeverb(a, size=[.88, .92], bal=.9).out()
b.ctrl()

s.gui(locals())