"""
Cos waveshaping synthesis.

"""
from pyo import *
import numpy as np

s = Server().boot()

# Sets fundamental frequency and highest harmonic.
freq = 278/8
high = 20

# Generates lists for frequencies and amplitudes
harms = [freq * i for i in range(1, high)]
amps = [0.33 / i for i in range(1, high)]

# Creates a square wave by additive synthesis.
a = Sine(freq=harms, mul=amps)#.out()
a.ctrl()
print("Number of Sine streams: %d" % len(a))

# Mix down the number of streams of "a" before computing the Chorus.
b = Chorus(a.mix(1), depth=2, feedback=0.1)#.out()
b.ctrl()

d = Delay(b)
d.ctrl()

# b = Chorus(b, feedback=0.5)
# b = Chorus(b, feedback=0.5)
# b = Chorus(b, feedback=0.5)
# b = Chorus(b, feedback=0.5)

c = SPan(d).out()

scope = Scope(c).ctrl()

s.gui(locals())