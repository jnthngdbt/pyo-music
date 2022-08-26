
from pyo import *

s = Server().boot()
s.start()

a = LFO(freq=100)
a.ctrl()

Scope(a)

s.gui(locals())