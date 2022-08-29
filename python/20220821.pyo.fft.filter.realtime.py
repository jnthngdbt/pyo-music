from pyo64 import *
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt

from scipy import signal

s = Server().boot()
s.start()

# # Minimum FFT length requirement.
# fs = 44100 # Hz
# minFreq = 20 # Hz
# minFreqPeriod = 1/minFreq # .05 s
# minFreqNbCycles = 2
# minFftPeriod = minFreqNbCycles * minFreqPeriod # 2*.05=.1 > 4410
# minFftN = fs * minFftPeriod

N = 2**14 # 2**14=16k
Nh = int(N/2)

a = Noise(.25).mix(2)
fin = FFT(a, size=N, overlaps=4, wintype=2)

def getSigVal(sig):
  return sig.value.value if hasattr(sig.value, 'value') else sig.value

note = midiToHz(61)

p0 = Sig(int(N * note / s.getSamplingRate()))
p0.ctrl([SLMap(5., 500., 'lin', "value", p0.value)], "P0")

bw = Sig(8)
bw.ctrl([SLMap(5., 50., 'lin', "value", bw.value)], "BW")

bws = Sig(.7)
bws.ctrl([SLMap(.1, 2.0, 'lin', "value", bws.value)], "BWS")

dmp = Sig(.52)
dmp.ctrl([SLMap(.1, 10., 'lin', "value", dmp.value)], "DMP")

def createSpectrum():
  s = np.zeros(Nh)
  for i in range(16):
    Nw = int((i * getSigVal(bws) + 1) * getSigVal(bw))
    if Nw < 3: Nw = 3 # minimum length
    if Nw % 2 == 0: Nw += 1 # make sure odd
    Nwh = int(Nw/2)
    damping = np.exp(-i*getSigVal(dmp))
    w = 10 * damping * signal.windows.hann(Nw)
    p = int(getSigVal(p0) * (i + 1))

    a = max(0, p-Nwh)
    b = min(Nh-1, p + Nwh + 1)
    Nab = b-a
    s[a:b] += w[0:Nab] # slicing: [a,b[
  return s

v = createSpectrum()
t = DataTable(Nh, init=v.tolist())
amp = TableIndex(t, fin["bin"])

# plt.plot(v)
# plt.show()

re = fin["real"] * amp
im = fin["imag"] * amp
fout = IFFT(re, im, size=N, overlaps=4, wintype=2).mix(2)

p = []
p.append(fout)
# p.append(Freeverb(p[-1], size=[.5,.6])), p[-1].ctrl()
p[-1].out()

Spectrum(fout, size=N)

def changeSpectrum():
  tm = time.time()
  v = createSpectrum()
  t.replace(v.tolist())
  print("{}s".format(time.time() - tm))

clk = Pattern(changeSpectrum, .1).play()

s.gui(locals())
