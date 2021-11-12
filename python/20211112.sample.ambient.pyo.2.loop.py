from pyo import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

name = "./data/03 Mission Two.m4a.aiff" # 72*0.05, 88*0.05
# name = "./data/04 Mission Three.m4a.aiff" # 24*0.05, 38*0.05, 234*0.05
# name = "./data/07 Mission Six.m4a.aiff" # 331*0.05, 545*0.05, 1760*0.05
# name = "./data/11 Mission Ten.m4a.aiff" # 494*0.05, 727*0.05
# name = "./data/Big Rock.1.m4a.aiff"
# name = "./data/Alone.3.m4a.aiff"
# name = "./data/Jump.12.m4a.aiff"
# name = "./data/Press.5.m4a.aiff"
# name = "./data/Late.06.m4a.aiff"
# name = "./data/Sam Sung 3.m4a.aiff"
# name = "./data/Aly Wood 2.m4a.aiff"
# name = "./data/Beverly Aly Hills 5.m4a.aiff" # t: 3.55, 7.5, 12.6
# name = "./data/insects.m4a.aiff"
# name = "./data/smallthings.m4a.aiff" # t: 2.85, 16.5, 31.55, 47.95
# name = "./data/Background noise with voice.m4a.aiff" # 43*0.05
# name = "./data/Tron Ouverture.m4a.aiff"

fftn = 2**13 #int(0.5 * T*fs) 

s = Server().boot()

snd = SndTable(name)

fs = snd.getSamplingRate()
T = 0.5

loop = Looper(snd, start=6, dur=T, startfromloop=True, xfade=0, mode=1, xfadeshape=0, interp=1)
loop.ctrl()

Fc = FFT(loop, size=fftn, overlaps=8)

# Randomize the phase.
Fp = CarToPol(Fc["real"], Fc["imag"])
FpM = Fp["mag"]
FpP = 2.0 * np.pi * Xnoise()

Fc2 = PolToCar(FpM, FpP)

x = IFFT(Fc2["real"], Fc2["imag"], size=fftn, overlaps=8).out()

s.gui(locals())