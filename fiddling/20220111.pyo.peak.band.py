from pyo import *
import numpy as np

s = Server().boot()
s.start()

# https://mynoise.net/NoiseMachines/peakNoiseGenerator.php?l=03030303039903030303
# https://mynoise.net/NoiseMachines/whiteNoiseGenerator.php?l=02020202029902020202

# Best peak noise 1 (almost like myNoise): 2 brown noise streams, biquadx 2 stages 10q

freq = midiToHz(73) # C# 85, like slider 6/10 on myNoise

noise1 = BrownNoise()
noise2 = BrownNoise()
band = Biquadx([noise1, noise2], freq=freq, q=2.3, type=2, stages=3, mul=.3)
peak = Biquadx([noise1, noise2], freq=freq, q=26,  type=2, stages=2, mul=2.)

band.ctrl()
peak.ctrl()

p = band + peak
p.out()

Spectrum(p)

s.gui(locals())
