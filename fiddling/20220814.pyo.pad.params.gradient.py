from pyo import *

s = Server().boot()
s.start()

padSize = 262144 * 2
padFreq = 440
padRatio = s.getSamplingRate() / padSize

f = Sig(midiToHz(48))
f.ctrl([SLMap(midiToHz(24), midiToHz(72), 'lin', "value", f.value)], "Freq")

freqRatio = padRatio * f / padFreq

t = Sig(1.)
t.ctrl([SLMap(0., 1., 'lin', "value", t.value)], "Gradient")


x = [
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=50, bwscl=1.0, damp=.7, nharms=64), freq=[freqRatio,freqRatio]),
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=30, bwscl=1.0, damp=.7, nharms=64), freq=[freqRatio,freqRatio]),
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=10, bwscl=1.0, damp=.7, nharms=64), freq=[freqRatio,freqRatio]),
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=70, bwscl=1.0, damp=.7, nharms=64), freq=[freqRatio,freqRatio]),
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=90, bwscl=1.0, damp=.7, nharms=64), freq=[freqRatio,freqRatio]),
  #
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=30, bwscl=1.0, damp=.9, nharms=64), freq=[freqRatio,freqRatio]),
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=30, bwscl=1.0, damp=.7, nharms=64), freq=[freqRatio,freqRatio]),
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=30, bwscl=1.0, damp=.5, nharms=64), freq=[freqRatio,freqRatio]),
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=30, bwscl=1.0, damp=.3, nharms=64), freq=[freqRatio,freqRatio]),
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=30, bwscl=1.0, damp=.1, nharms=64), freq=[freqRatio,freqRatio]),
  #
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=30, bwscl=1.0, damp=.9, nharms=64), freq=[freqRatio,freqRatio]),
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=30, bwscl=1.0, damp=.9, nharms=32), freq=[freqRatio,freqRatio]),
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=30, bwscl=1.0, damp=.9, nharms=16), freq=[freqRatio,freqRatio]),
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=30, bwscl=1.0, damp=.9, nharms=8), freq=[freqRatio,freqRatio]),
  #
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=40, bwscl=2., damp=.5, nharms=64), freq=[freqRatio,freqRatio]),
]

n = len(x)

for i in range(n):
  p = i / n
  m = Clip(1. - n * Abs(t-p), min=0, max=1)
  x[i].mul = m

pipeline = []
pipeline.append(Mix(x, 2))
# pipeline.append(Compress(pipeline[-1], thresh=-20., ratio=8.)), pipeline[-1].ctrl()
# pipeline.append(Freeverb(pipeline[-1], size=.99, damp=.5, bal=0.5)), pipeline[-1].ctrl()
# pipeline.append(Delay(pipeline[-1], delay=[.15, .2], feedback=.0))
# pipeline.append(Pan(pipeline[-1], outs=2, pan=self.lfoPan, mul=self.lfoMul))
# pipeline.append(MoogLP(pipeline[-1], freq=self.cutoff, res=0.0))
pipeline[-1].out()

s.gui(locals())
