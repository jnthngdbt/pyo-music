from pyo64 import *
import numpy as np

s = Server().boot()
s.start()

N = 1024

a = Noise(.25).mix(2)
fin = FFT(a, size=N, overlaps=4, wintype=2)

d = Sig(1)
d.ctrl([SLMap(0., 10., 'lin', "value", d.value)], "D")

t = DataTable(512, init=np.random.rand(512).tolist())
amp = TableIndex(t, fin["bin"])

re = fin["real"] * amp
im = fin["imag"] * amp
fout = IFFT(re, im, size=N, overlaps=4, wintype=2, mul=0.2).mix(2).out()

Spectrum(fout, size=4096)

def changeSpectrum():
  if hasattr(d.value, 'value'):
    v = d.value.value * np.random.rand(512)
    t.replace(v.tolist())

clk = Pattern(changeSpectrum, 1).play()

s.gui(locals())
