from pyo import *
import numpy as np

s = Server().boot()
s.start()

# ---------------------------------------------------------
# Actual Northern lights analysis
# - modes at: 208Hz (G#), 278Hz (C#)

# -----------------------------------------------------------------------------------------------

class PeakSynth:
    def __init__(self, freq, qFactor, damp, mulp):
        self.noise1 = BrownNoise()

        Nh = 16

        f = [i * freq for i in range(1,Nh)]
        mb = [.22*Exp(-i*damp) for i in range(1,Nh)]
        mp = [mulp*Exp(-i*damp) for i in range(1,Nh)]

        self.band = Biquadx(self.noise1, freq=f, q=qFactor*2.3, type=2, stages=3, mul=mb)
        self.peak = Biquadx(self.noise1, freq=f, q=qFactor*26,  type=2, stages=2, mul=mp)
        self.out = self.band + self.peak

mulp = Sig(9)
mulp.ctrl([SLMap(0., 10., 'lin', "value", mulp.value)], "Peak volume") # NOTE: the 'name' must be the name of attribute

qFactor = Sig(3.5)
qFactor.ctrl([SLMap(0.01, 10., 'lin', "value", qFactor.value)], "Qf", ) # NOTE: the 'name' must be the name of attribute

damp = Sig(.1)
damp.ctrl([SLMap(0.01, 1., 'lin', "value", damp.value)], "Damp", ) # NOTE: the 'name' must be the name of attribute

root = 73
oscs = [
    PeakSynth(midiToHz(root-7), qFactor, damp, mulp),
    PeakSynth(midiToHz(root-5), qFactor, damp, mulp),
    PeakSynth(midiToHz(root  ), qFactor, damp, mulp),
]

p = []
p.append(Mix([osc.out for osc in oscs], 1, mul=1.))
p.append(Delay(p[-1], delay=[.00,.20]))
p.append(Biquadx(p[-1], midiToHz(root+12), stages=2))
p[-1].ctrl()
p[-1].out()

Spectrum(p[-1])

# # ---------------------------------------------------------

# def expand(notes=[48, 52, 55], octaves=[0,1,2]):
#   x = []
#   notes = np.array(notes)
#   for i in octaves:
#     x = np.concatenate([x, i * 12 + notes], axis=0)
#   return x.tolist()

# -----------------------------------------------------------

# class Peak:
#   def __init__(self, note, mulp, mulb, qp, qb) -> None:
#     freq = midiToHz(note)
#     self.noise1 = BrownNoise()
#     self.noise2 = BrownNoise()
#     self.band = Biquadx([self.noise1, self.noise2], freq=freq, q=qb, type=2, stages=3, mul=mulb)
#     self.peak = Biquadx([self.noise1, self.noise2], freq=freq, q=qp, type=2, stages=2, mul=mulp)

#     self.out = self.band + self.peak

# mulp = Sig(2.)
# mulp.ctrl([SLMap(0., 4., 'lin', "value", mulp.value)], "Peak volume") # NOTE: the 'name' must be the name of attribute

# mulb = Sig(.3)
# mulb.ctrl([SLMap(0., 4., 'lin', "value", mulb.value)], "Noise volume") # NOTE: the 'name' must be the name of attribute

# qp = Sig(26)
# qp.ctrl([SLMap(0., 50., 'lin', "value", qp.value)], "Qp") # NOTE: the 'name' must be the name of attribute

# qb = Sig(2.3)
# qb.ctrl([SLMap(0., 50, 'lin', "value", qb.value)], "Qb") # NOTE: the 'name' must be the name of attribute

# root = 85
# xs = [
#   Peak(root-7, mulp, mulb, qp, qb),
#   Peak(root-5, mulp, mulb, qp, qb),
#   Peak(root  , mulp, mulb, qp, qb),
# ]

# p = []
# p.append(Mix([x.out for x in xs], 2))
# p.append(MoogLP(p[-1], freq=20000, res=1.)); p[-1].ctrl()
# # p.append(MoogLP(p[-1], freq=Osc(TriangleTable(), freq=0.005, phase=.9).range(1000, 5000), res=.1)); # LFO controlled cutoff
# p[-1].out()

# Spectrum(p[-1])

# -----------------------------------------------------------
# padFreq = 440
# padSize = 2**19
# padRatio = s.getSamplingRate() / padSize
# t = PadSynthTable(basefreq=padFreq, size=padSize, damp=0.7, bw=50, bwscl=1, spread=1)

# root = 49 # c#
# intervals = [0, 9, 16]
# amp = [1.,.6, .1]
# octaves = [0]

# notes = [x + root for x in intervals]
# # freq = expand(octaves=octaves, notes=)
# freq = padRatio * np.array(midiToHz(notes)) / padFreq

# o = Osc(t, freq=freq.tolist(), mul=amp)
# d = Delay(o, delay=[.00, .05], feedback=0)
# # c = Chorus(d, depth=1, feedback=0.25, bal=0.5)
# # c.ctrl()
# b = Biquadx(d, freq=midiToHz(85), q=1, type=2, stages=2)

# mix = 4. * b

# p = Pan(mix, outs=2, pan=.5, spread=0)
# p.out()
# -----------------------------------------------------------

# root = 85 # c#
# amp = [1., .5, .4]
# notes = [0, 5, 7]

# freq = expand(octaves=[0], notes=[x + root for x in notes])

# noise1 = PinkNoise()
# noise2 = PinkNoise()
# b = Biquadx([noise1, noise2], freq=midiToHz(freq), mul=amp, q=40, type=2, stages=2)
# p = Pan(b, outs=2, pan=0.5)

# mix = 2.0 * p
# mix.out()


# -----------------------------------------------------------

# Peaks, expanded, bandpass


# class Peak:
#   def __init__(self, note=48, mul=1.0) -> None:
#     q = 30 # 10 + 0.3*note
#     self.noise1 = PinkNoise()
#     self.noise2 = PinkNoise()
#     self.band = Biquadx([self.noise1, self.noise2], freq=midiToHz(note), q=q, type=2, stages=2, mul=mul)
#     # self.band.ctrl()

#   def out(self):
#     self.band.out()

# root = 49 # c#
# intervals = [0, 5]
# octaves = [0,1,2,3,4]

# notes = [x + root for x in intervals]
# notes = expand(notes, octaves)

# # peaks = [
# #   # Peak(note=66, mul=0.5),
# #   Peak(note=73, mul=0.4),
# #   Peak(note=78, mul=0.6),
# #   Peak(note=85, mul=0.3),
# #   Peak(note=92, mul=0.2),
# #   # Peak(note=95, mul=0.2),
# # ]

# peaks = [Peak(n) for n in notes]
# pk = [peak.band for peak in peaks]

# p = Pan(pk, outs=2, pan=0.5)

# # p.out()

# # lfoLpTable = TriangleTable()
# # lfoLp = Osc(lfoLpTable, freq=.05, phase=0.25).range(1000, 4000)
# # lp = MoogLP(p, freq=lfoLp, res=1)
# # # Reson
# # lp.out()

# rs = Biquadx(p, freq=2000, q=10, stages=2, type=2)
# # rs = Resonx(p, freq=2000, q=10, stages=2)
# rs.out()
# rs.ctrl()

# -----------------------------------------------------------

s.gui(locals())
