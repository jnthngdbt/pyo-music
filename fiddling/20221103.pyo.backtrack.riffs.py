from pyo import *
import numpy as np
from typing import List

s = Server().boot()

maxNote = 100 # set high note to deactivate this fonctionality

gain = .042
root = 25 # below 24 is too low // A:21, C:24, E:28
bpm = 48

beatDur = 60 / bpm # beatPerSec = bpm / 60, so secPerBeat = 60 / bpm
beatPerBar = 4 # 4:4
barDur = beatDur * beatPerBar

class Track:
    def __init__(self, name="Track", root=49, note=[0], beat=[1], dur=.2, prob=.6, speed=1, mul=1):
        self.name = name
        self.root = root
        self.beat = beat
        self.note = note
        self.dur = dur
        self.prob = prob
        self.speed = speed

        self.mul = Sig(mul)
        self.mul.ctrl(title=name + " Gain")

class Maestro:
    def __init__(self, track: Track, riffs: List[List[int]], time=0.25, nbSectionsSwitch=0):
        self.track = track
        self.riffs = riffs
        self.time = time
        self.nbSectionsSwitch = nbSectionsSwitch

        self.beatIdx = 0

        self.mul = self.track.mul
        self.sectionSize = max([len(self.track.note), len([self.track.beat])])

        self.clock = Pattern(self.tick, time=self.time).play()

        print('Riff initial: {}'.format(self.track.note))

    def tick(self):
        doSwitch = (self.nbSectionsSwitch > 0) and (self.beatIdx > 0) and (self.beatIdx % (self.sectionSize * self.nbSectionsSwitch) == 0)
        if doSwitch:
            self.switchRiff()

        beat = self.track.beat[self.beatIdx % len(self.track.beat)]
        note = self.track.note[self.beatIdx % len(self.track.note)] + self.track.root

        if note > maxNote:
            note = note - 12

        if beat:
            self.track.play(note)

        self.beatIdx = self.beatIdx + 1

    def switchRiff(self):
        idx = random.randint(0, len(self.riffs)-1)
        self.track.note = self.riffs[idx]
        print('Riff #{}: {}'.format(idx, self.riffs[idx]))


class Bass (Track):
    def __init__(self, adsr=[.01,.01,.8,.02], tone=[.8, 1., .8, .5, .3, .2, .1], reverb=0., **kwargs):
        super().__init__(**kwargs)

        self.env = Adsr(attack=adsr[0], decay=adsr[1], sustain=adsr[2], release=adsr[3], dur=self.dur, mul=self.mul)
        # self.env.ctrl()

        self.table = HarmTable(tone)
        self.osc = Osc(table=self.table, freq=midiToHz(25), mul=self.env, phase=[0,.4])
        self.p = []
        self.p.append(self.osc)
        self.p.append(Freeverb(self.p[-1], size=0.8, damp=0.5, bal=reverb))
        self.p.append(Chorus(self.p[-1]))
        
        self.p[-1].out()

    def play(self, note=0):
        self.osc.freq = midiToHz(note)
        self.env.play()

riffs = [
  [ 9, 5, 0, 7],
  [ 9, 5, 0, 0],
  [ 9, 7, 0, 5],
  [ 9, 7, 5, 0],
  [ 5, 0, 9, 7],
  [ 5, 0, 7, 9],
  [ 0, 7, 9, 5],
  [ 5, 7, 9, 9],
  [ 5, 7, 9, 5],
  [ 5, 0, 7, 7],
  [ 0, 7, 0, 5],
  [ 0, 5, 0, 7],
  [ 0, 5, 0, 5],
  [ 4, 5, 0, 7],
  [ 7, 9, 5, 0],
  [ 0, 0, 0, 5],
  [ 0, 0, 0, 5],
  [ 0, 2, 4, 5],
]

M = Maestro(time=barDur, nbSectionsSwitch=2, riffs=riffs, track=
    Bass(name="Bass", mul = gain, root = root, adsr=[.01, .01, 1, .01], dur=barDur, tone=[1, 1, .6, .2, .1],
        #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
        beat = [ 1 ],
        note = [ 0, 7, 9, 5],
    )
)

s.start()
s.gui(locals())
