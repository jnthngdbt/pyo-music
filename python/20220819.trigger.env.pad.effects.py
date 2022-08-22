from pyo import *

s = Server().boot()
s.start()

padSize = 262144 * 2
padFreq = 440
padRatio = s.getSamplingRate() / padSize

def f(f):
  return padRatio * f / padFreq

T = 16
t = Trig()
a = HannTable()
ta = TrigEnv(t, a, T)

x = [
  Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=50, bwscl=1.0, damp=1., nharms=64), freq=[f(400), f(405)], mul=ta),
]



p = []
p.append(Mix(x, 2))
# p.append(Compress(p[-1], thresh=-20., ratio=8.)), p[-1].ctrl()
# p.append(Freeverb(p[-1], size=.99, damp=.5, bal=0.5)), p[-1].ctrl()
p.append(Delay(p[-1], delay=[.39, .38], feedback=.9)), p[-1].ctrl()
# p.append(Pan(p[-1], outs=2, pan=self.lfoPan, mul=self.lfoMul))
p.append(WGVerb(p[-1], feedback=.9)), p[-1].ctrl()
p.append(MoogLP(p[-1])), p[-1].ctrl()
p.append(EQ(p[-1], freq=100))                               , p[-1].ctrl()
p.append(EQ(p[-1], freq=500))                               , p[-1].ctrl()
p.append(EQ(p[-1], freq=1000))                               , p[-1].ctrl()
p[-1].out()

b = Pattern(t.play, T + 4).play()

Spectrum(p[-1])

s.gui(locals())
