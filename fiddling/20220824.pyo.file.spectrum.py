from pyo import *
import numpy as np

s = Server().boot()
s.start()

# A3: 220Hz, A4: 440Hz

# t = SndTable("C:\\Users\\jgodbout\\Documents\\git\\sandbox\\python\\songs\\slider.spectrum.sample.wav")
# t = SndTable("C:\\Users\\jgodbout\\OneDrive\\Music\\myNoise\\WhitePeak\\3a.ogg") # C# 250 Hz
# t = SndTable("C:\\Users\\jgodbout\\OneDrive\\Music\\myNoise\\WhitePeak\\4a.ogg") # C# 500 Hz
# t = SndTable("C:\\Users\\jgodbout\\OneDrive\\Music\\myNoise\\Northern\\5a.ogg") # peaks: 141, 191
t = SndTable("C:\\Users\\jgodbout\\OneDrive\\Music\\myNoise\\OakSongRiikka\\9a.ogg") # peaks: 240(C), 287(D), 381(G), 430(A)

f = s.getSamplingRate() / t.size
x = Osc(t, freq=f).out()

Spectrum(x)

s.gui(locals())