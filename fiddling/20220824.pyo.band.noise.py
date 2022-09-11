from pyo import *
import numpy as np

s = Server().boot()
s.start()

# https://mynoise.net/NoiseMachines/peakNoiseGenerator.php?l=03030303039903030303
# https://mynoise.net/NoiseMachines/whiteNoiseGenerator.php?l=02020202029902020202

# Best band noise 1 (almost like myNoise): 2 brown noise streams, biquadx 2 stages 1.3q

freq = midiToHz(85) # C# 73, like slider 5/10 on myNoise

noise1 = BrownNoise()
noise2 = BrownNoise()
band = Biquadx([noise1, noise2], freq=freq, q=1.3, type=2, stages=2, mul=.2).out()

band.ctrl()

Spectrum(band)

s.gui(locals())
