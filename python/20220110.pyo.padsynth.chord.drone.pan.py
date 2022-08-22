from pyo import *
import numpy as np

s = Server().boot()
s.start()

class Chord:
    def __init__(self, root, scale, octaves, bw, damp, mul, cutoff=20000):

        padSize = 262144 * 2
        padFreq = 440
        padRatio = s.getSamplingRate() / padSize

        # Expand notes across octaves.
        self.notes = np.nonzero(scale)[0]
        for i in octaves:
            self.notes = np.concatenate([self.notes, i * 12 + self.notes], axis=0)

        nbNotes = len(self.notes)

        self.table = PadSynthTable(
          basefreq=padFreq,
          size=padSize,
          spread=1, # 1: def strings, 2:shallower/pure, in between: creepy (near 1, slight dissonnance) // freq = basefreq * pow(n, spread)
          bw=bw, # 20: org, 50: def strings, 70: dreamy more highs
          bwscl=1, # 1: def string, 2: dreamy, 3: flute/wind
          damp=damp, # 0.7: def, 1: big high synth, 0.5: mellow // amp = pow(damp, n)
          nharms=64, # 64: def
        )

        def randRange(a,b):
            r = b - a
            return a + r * np.random.rand()

        self.oscLfos = [None] * nbNotes
        self.oscs = [None] * nbNotes

        self.panLfos = [None] * nbNotes
        self.pans = [None] * nbNotes

        # Create note streams with LFOs.
        for i, note in enumerate(self.notes):
            f = midiToHz(note + root)
            freqRatio = padRatio * f / padFreq

            self.oscLfos[i] = Sine(freq=randRange(0.01, 0.02), phase=0.75).range(0, 1)
            self.oscs[i] = Osc(self.table, freq=freqRatio, mul=self.oscLfos[i])

            self.panLfos[i] = Sine(freq=randRange(0.01, 0.02), phase=np.random.rand()).range(.2, .8)
            self.pans[i] = Pan(self.oscs[i], outs=2, pan=self.panLfos[i])

        self.p = []
        self.p.append(Mix(self.pans, 2, mul=mul))
        self.p.append(Delay(self.p[-1], delay=[.38, .39], feedback=.5))                       , self.p[-1].ctrl()
        # self.p.append(Freeverb(self.p[-1], size=[.79,.80], damp=.9, bal=.3))                  , self.p[-1].ctrl()
        self.p.append(WGVerb(self.p[-1]))                  , self.p[-1].ctrl()
        self.p.append(MoogLP(self.p[-1], freq=cutoff, res=0.0))                               , self.p[-1].ctrl()
        self.p.append(EQ(self.p[-1], freq=100))                               , self.p[-1].ctrl()
        self.p.append(EQ(self.p[-1], freq=500))                               , self.p[-1].ctrl()
        self.p.append(EQ(self.p[-1], freq=1000))                               , self.p[-1].ctrl()
        self.p[-1].out()

        Spectrum(self.p[-1])


# Notes to play. Put selected profile at the end to be active.
#        0  1  2  3  4  5  6  7  8  9  10 11
scale = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] # power
scale = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # peak
scale = [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0] # riff
scale = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0] # major
scale = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0] # minor small
scale = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0] # major small
scale = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0] # minor
scale = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # note

root = 72
dampFactor = 1.0
cutoff = 20000
volume = .35

x = Chord(mul=volume*.7, root=root, scale=scale, octaves=[], bw=40, damp=dampFactor*0.7, cutoff=cutoff)

s.gui(locals())
