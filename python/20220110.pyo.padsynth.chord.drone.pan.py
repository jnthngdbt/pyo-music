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

            self.oscLfos[i] = Sine(freq=randRange(0.01, 0.04), phase=0.75).range(0, 1)
            self.oscs[i] = Osc(self.table, freq=freqRatio, mul=self.oscLfos[i])

            self.panLfos[i] = Sine(freq=randRange(0.01, 0.04), phase=np.random.rand()).range(0.3, 0.7)
            self.pans[i] = Pan(self.oscs[i], outs=2, pan=self.panLfos[i])

        # Mix all note streams.
        self.x = 0.0 * Sine()
        for sig in self.pans:
            self.x = self.x + sig

        self.d = Delay(self.x, delay=np.arange(0.15, 0.3, 0.05).tolist(), feedback=.5)
        self.d.ctrl()

        self.r = Freeverb(self.d, size=[.79,.8], damp=.9, bal=.3, mul=mul)
        self.r.ctrl()

        self.lp = MoogLP(self.r, freq=cutoff, res=0.0)
        self.lp.out()

# #        0  1  2  3  4  5  6  7  8  9  10 11
# scale = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0] # major
# scale = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0] # minor small
# scale = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0] # minor
# scale = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] # power
# scale = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # peak
# scale = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0] # major small
# scale = [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0] # riff
# scale = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # note

scale = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # note
a = Chord(mul=.7, root=24, scale=scale, octaves=[0,1], bw=40, damp=0.9)

scale = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] # power
b = Chord(mul=.4, root=60, scale=scale, octaves=[0], bw=50, damp=0.5)

scale = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # note
c = Chord(mul=.15, root=84, scale=scale, octaves=[0], bw=50, damp=0.4)

s.gui(locals())
