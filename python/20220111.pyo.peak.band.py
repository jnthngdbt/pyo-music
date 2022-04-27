from pyo import *
import numpy as np

s = Server().boot()
s.start()

# https://mynoise.net/NoiseMachines/peakNoiseGenerator.php?l=03030303039903030303
# https://mynoise.net/NoiseMachines/whiteNoiseGenerator.php?l=02020202029902020202

# Best peak noise 1 (almost like myNoise): 2 pink noise streams, biquadx 2 stages 10q

freq = midiToHz(85) # C# 85, like slider 6/10 on myNoise

noise1 = PinkNoise()
noise2 = PinkNoise()
band = Biquadx([noise1, noise2], freq=freq, q=10, type=2, stages=2, mul=.5)

band.out()

s.gui(locals())
