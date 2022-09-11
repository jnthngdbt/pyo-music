from attr import has
from pyo import *
import numpy as np

s = Server().boot()
s.start()

t = Sig(0.5)
t.ctrl()

# ma = t 
# mb = (1.0-t)
# print(ma) # Sig
# print(ma.value)

# ma = t * 1.0
# mb = (1.0-t)
# print(ma) # dummy, constant value
# # print(ma.value) # error

# ma = t * Sig(1.0)
# mb = (Sig(1.0)-t)
# print(ma) # dummy, constant value
# # print(ma.value) # error

# ma = Sig(t * 1.0)
# mb = Sig(1.0-t)
# print(ma) # Sig, but value contains dummy
# print(ma.value)

# ma = Sig(t.value * 1.0)
# mb = Sig(1.0-t.value)
# print(ma) # Sig, but constant value
# print(ma.value)


# ma = Sub(t, 0.0, mul=1.0) # (ma = t * 1.0) to multiply, sub 0 and use mul
# mb = Sub(1.0, t) # (mb = 1-t) sub
# print(ma) # Sig, works
# # print(ma.value) # Sub has no attribute 'value'

ma = Sig(1., mul=t) # (ma = t * 1.0) to multiply, sub 0 and use mul
mb = Sig(1., add=-t) # (mb = 1-t) sub
print(ma) # Sig, works
print(mb) # Sig, works
print(ma.value)
print(mb.value)

a = Sine(300, mul=ma)
b = Sine(500, mul=mb)

m = Mix([a,b], 2, mul=0.1).out()

s.gui(locals())
