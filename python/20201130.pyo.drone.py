from pyo import *
import numpy as np

s = Server().boot()

# Sets fundamental frequency and highest harmonic.
freq = 278/4
high = 12

# Generates lists for frequencies and amplitudes
harms = [freq * i for i in range(1, high)]
amps = [1 / i for i in range(1, high)]
# amps = [0, 0.18, 0.25, 0.33, 0.36, .33, .28, .2, .12, .01, 0, 0]

a = Sine(freq=harms, mul=amps)#.out()
a.ctrl()

b = Chorus(a.mix(1), depth=1, feedback=0.1, bal=0.5)#.out()
b.ctrl()

c = SPan(b).out()

fd = Fader(fadein=4, fadeout=4, dur=15).play() # DOES NOT WORK

scope = Scope(fd).ctrl()

s.gui(locals())


# # Rich frequency spectrum as stereo input source.
# amp = Fader(fadein=0.25, mul=0.5).play()
# src = PinkNoise(amp).mix(2)

# # Flanger parameters                        == unit ==
# middelay = 0.005                            # seconds

# depth = Sig(0.99)                           # 0 --> 1
# depth.ctrl(title="Modulation Depth")
# lfospeed = Sig(0.2)                         # Hertz
# lfospeed.ctrl(title="LFO Frequency in Hz")
# feedback = Sig(0.5, mul=0.95)               # 0 --> 1
# feedback.ctrl(title="Feedback")

# # LFO with adjusted output range to control the delay time in seconds.
# lfo = Sine(freq=lfospeed, mul=middelay*depth, add=middelay)

# # Dynamically delayed signal. The source passes through a DCBlock
# # to ensure there is no DC offset in the signal (with feedback, DC
# # offset can be fatal!).
# flg = Delay(DCBlock(src), delay=lfo, feedback=feedback)

# # Mix the original source with its delayed version.
# # Compress the mix to normalize the output signal.
# cmp = Compress(src+flg, thresh=-20, ratio=4).out()
