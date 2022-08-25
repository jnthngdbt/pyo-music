from pyo import *
import numpy as np

s = Server().boot()
s.start()

# A3: 220Hz, A4: 440Hz

# t = SndTable("C:\\Users\\jgodbout\\Documents\\git\\sandbox\\python\\songs\\slider.spectrum.sample.wav")
# t = SndTable("C:\\Users\\jgodbout\\OneDrive\\Music\\myNoise\\WhitePeak\\3a.ogg") # C# 250 Hz
# t = SndTable("C:\\Users\\jgodbout\\OneDrive\\Music\\myNoise\\WhitePeak\\4a.ogg") # C# 500 Hz
t = SndTable("C:\\Users\\jgodbout\\OneDrive\\Music\\myNoise\\WhitePeak\\5a.ogg") # C# 1000 Hz

f = s.getSamplingRate() / t.size
x = Osc(t, freq=f).out()

Spectrum(x)

s.gui(locals())