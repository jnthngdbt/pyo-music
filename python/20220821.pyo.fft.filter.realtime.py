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

def getSigVal(sig):
  if not hasattr(sig, 'value'):
    return sig
  return sig.value.value if hasattr(sig.value, 'value') else sig.value

class SpectrumEditor:
  def __init__(self) -> None:
    pass

class PeakPadSpectrum:
  def __init__(self) -> None:
    self.size = 2**14 # 2**14=16k
    self.halfsize = int(self.size/2)

    self.freq = Sig(midiToHz(73))
    self.freq.ctrl([SLMap(20., 5000., 'lin', "value", self.freq.value)], "Frequency")

    self.qp = Sig(9.6)
    self.qp.ctrl([SLMap(.1, 50., 'lin', "value", self.qp.value)], "Q Peak")

    self.qb = Sig(.67)
    self.qb.ctrl([SLMap(.1, 50., 'lin', "value", self.qb.value)], "Q Noise")

    self.mulp = Sig(10)
    self.mulp.ctrl([SLMap(.1, 50., 'lin', "value", self.mulp.value)], "Amplitude Peak")

    self.mulb = Sig(2)
    self.mulb.ctrl([SLMap(.1, 50., 'lin', "value", self.mulb.value)], "Amplitude Noise")

    self.damp = Sig(.56)
    self.damp.ctrl([SLMap(.01, 3., 'lin', "value", self.damp.value)], "Damp")

    self.nbHarms = Sig(1)
    self.nbHarms.ctrl([SLMap(1, 64, 'lin', "value", self.nbHarms.value, res='int')], "Harmonics")

    self.shape = Sig(.323)
    self.shape.ctrl([SLMap(0, 1, 'lin', "value", self.shape.value)], "Shape")

    self.std = Sig(.012)
    self.std.ctrl([SLMap(0.005, .2, 'lin', "value", self.std.value)], "Sigma")

    self.magnitudes = np.zeros(self.halfsize)
    self.generate()

  def generate(self):
    self.magnitudes = np.zeros(self.halfsize)

    self.addPeaks(
      getSigVal(self.freq), 
      getSigVal(self.nbHarms), 
      getSigVal(self.mulp), 
      getSigVal(self.qp), 
      getSigVal(self.damp), 
      getSigVal(self.shape),
      getSigVal(self.std))

    self.addPeaks(
      getSigVal(self.freq), 
      getSigVal(self.nbHarms), 
      getSigVal(self.mulb), 
      getSigVal(self.qb), 
      getSigVal(self.damp),
      getSigVal(self.shape),
      getSigVal(self.std))

  def addPeaks(self, freq, nbHarms, scale, q, damp, shape, std):
    for i in range(nbHarms):
      fi = freq * (i + 1)
      bandwidth = (fi / q)
      
      winSize = int(self.size * bandwidth / s.getSamplingRate())
      if winSize < 3: winSize = 3 # minimum length
      if winSize % 2 == 0: winSize += 1 # make sure odd

      winHalfsize = int(.5 * winSize)

      damping = np.exp(-i * damp)
      w = scale * damping * signal.windows.general_gaussian(winSize, shape, std*winSize)

      freqPos = int(self.size * fi / s.getSamplingRate())

      a = max(0, freqPos - winHalfsize)
      b = min(self.size-1, freqPos + winHalfsize + 1)
      Nab = b-a

      self.magnitudes[a:b] += w[0:Nab] # slicing: [a,b[

class PeakPadSynth:
  def __init__(self) -> None:
    self.spectrum = PeakPadSpectrum()

    self.noise = Noise(.25).mix(2)
    self.fft = FFT(self.noise, size=self.spectrum.size, overlaps=4, wintype=2)

    self.table = DataTable(self.spectrum.halfsize, init=self.spectrum.magnitudes.tolist())
    self.amp = TableIndex(self.table, self.fft["bin"])

    self.real = self.fft["real"] * self.amp
    self.imag = self.fft["imag"] * self.amp
    self.ifft = IFFT(self.real, self.imag, size=self.spectrum.size, overlaps=4, wintype=2).mix(1)

    self.out = Delay(self.ifft, [.0, .2])

  def updateSpectrum(self):
    tm = time.time()
    self.spectrum.generate()
    self.table.replace(self.spectrum.magnitudes.tolist())
    print("{}s".format(time.time() - tm))


x = PeakPadSynth()

p = []
p.append(x.out)
p[-1].out()

Spectrum(p[-1])

clk = Pattern(x.updateSpectrum, .1).play()

s.gui(locals())
