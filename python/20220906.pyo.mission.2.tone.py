
from pyo import *

s = Server().boot()
s.start()

class Horn:
  def __init__(self, note, tone, mul=1., env=1, envf=1) -> None:
    self.env = env
    self.envf = envf
    self.table = HarmTable(tone) # sub 0, main, fifth, octave
    self.osc = Osc(self.table, midiToHz(note))
    self.p = []
    self.p.append(self.osc.mix(1))
    self.p.append(Chorus(self.p[-1], [1, 1.2], feedback=0)) # adds some movement
    self.p.append(Freeverb(self.p[-1], .8)) # adds some movement
    self.p.append(ButLP(self.p[-1], freq=20000*self.envf))
    self.p[-1].mul = self.env * mul
    self.p[-1].out()

    if hasattr(self.env, 'ctrl'): self.env.ctrl()
    if hasattr(self.envf, 'graph'): self.envf.graph()

  def playNote(self, note):
    if hasattr(self.envf, 'play'): self.envf.play()
    if hasattr(self.env, 'play'): self.env.play()
    self.osc.freq = self.envf * midiToHz(note)

  def getOsc(self):
    return self.p[-1]

class BassString:
  def __init__(self, note, mul) -> None:
    f = midiToHz(note)
    padSize = 262144 * 2
    padFreq = 440
    padRatio = s.getSamplingRate() / padSize
    freqRatio = padRatio * f / padFreq
    self.p = []
    self.p.append(Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=50, bwscl=1.05, damp=.7, nharms=64), freq=freqRatio))
    self.p.append(EQ(self.p[-1], f, q=10, boost=-40)) # adds some movement
    # self.p.append(EQ(self.p[-1], f*2, q=10, boost=-10)) # adds some movement
    self.p[-1].ctrl()
    self.p.append(Freeverb(self.p[-1], [.78, .80])) # adds some movement
    self.p[-1].mul = mul
    self.p[-1].out()

  def getOsc(self):
    return self.p[-1]

class NoiseBaseline:
  def __init__(self, mul) -> None:
    self.p = []
    self.p.append(BrownNoise())
    self.p.append(Delay(self.p[-1], delay=[0, .1])) # adds some movement
    self.p[-1].mul = mul
    self.p[-1].out()

  def getOsc(self):
    return self.p[-1]

class InstrumentPlayer:
  def __init__(self, instrument, root=49, note=[0], beat=[1], time=0.25, interval=0, wait=0):
    self.instrument = instrument
    self.beat = beat
    self.root = root
    self.note = note
    self.time = time
    self.interval = interval
    self.wait = wait

    self.size = max([len(beat), len(note)])
    self.tickIdx = 0
    self.isInterval = False
    self.isWaiting = wait > 0
    
    self.clock = Pattern(self.tick, time=self.time).play()

  def tick(self):
    if self.isWaiting:
      if self.tickIdx < self.wait:
        self.tickIdx = self.tickIdx + 1
        return # wait
      else:
        self.tickIdx = 0
        self.isWaiting = False
    elif self.isInterval:
      if self.tickIdx < self.interval:
        self.tickIdx = self.tickIdx + 1
        return # wait
      else:
        self.tickIdx = 0
        self.isInterval = False

    beat = self.beat[self.tickIdx % len(self.beat)]
    note = self.note[self.tickIdx % len(self.note)]
    if beat:
        self.instrument.playNote(self.root + note)
    self.tickIdx = self.tickIdx + 1

    if self.tickIdx == self.size:
      self.tickIdx = 0
      self.isInterval = True

  def getOsc(self):
    return self.instrument.getOsc()

x = [
  BassString(32, mul=1),
  Horn([56], mul=.02, tone=[1., .5, .25]),
  InstrumentPlayer(
    Horn(63, mul=.04, tone=[1., .5, .25], env=Adsr(.04, 0, 1, 0)),
    beat=[1],
    note=[0],
    root=63, time=.2, wait=4, interval=40
  ),
  InstrumentPlayer(
    Horn(56, mul=.1, tone=[1., .5, .25], env=Adsr(.03, 120, 0, 0), envf=Linseg([(0,0.4),(.01,1)])),
    beat=[ 1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    note=[-5,0,2,0,0,0,0,0,4,5,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # beat=[ 1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0, 1,0,1],
    # note=[-5,0,2,0,0,0,0,0,4,5,4,0,0,0,0,0,0,0,0,0,0,0,0,2,0,-2,0,0],
    root=56, time=.2, interval=60, wait=20
  ),
  NoiseBaseline(mul=.01),
]

Spectrum([xi.getOsc() for xi in x], size=8192)

s.gui(locals())