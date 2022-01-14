from pyo import *
import numpy as np

s = Server().boot()
s.start()

class Pad:
  def __init__(self):
    self.size = 262144
    self.basefreq = 440
    self.ratio = s.getSamplingRate() / self.size
    self.table = PadSynthTable(basefreq=self.basefreq, size=self.size, 
      spread=1, # for slight dissonnance
      bw=50, # pure > noisy
      bwscl=1, # breathing
      damp=0.7) # mellow/bright

  def freq(self, midi=48):
    f = midiToHz(midi)
    f = np.array(f)
    f = self.ratio * f / self.basefreq
    return f.tolist()

class Instrument:
    def __init__(self):
      self.maxchnls = 16 # seems that number of channels (notes) stays fixed from instantiation
      self.pad = Pad()
      self.osc = Osc(self.pad.table, freq=np.zeros(self.maxchnls).tolist()).stop()
      self.pan = Pan(self.osc, outs=2, pan=0.5, spread=0.5).stop()
      self.delay = Delay(self.pan, delay=[0.15, 0.2, 0.25, 0.3], feedback=0.5).stop()
      self.env = Fader(fadein=2, fadeout=2, dur=1).stop()

      self.mix = 0.1 * self.delay * self.env

    def play(self, notes=[48, 55], dur=8, delay=0):
        notes = self.expand(notes=notes)
        self.osc.freq = self.pad.freq(notes)

        self.env.dur = dur

        self.osc.play(dur=dur, delay=delay)
        self.pan.play(dur=dur, delay=delay)
        self.delay.play(dur=dur, delay=delay)
        self.env.play(dur=dur, delay=delay)

        self.mix.out(dur=dur, delay=delay)

    def expand(self, notes=[48, 52, 55], octaves=[0,1]):
      x = []
      notes = np.array(notes)
      for i in octaves:
        x = np.concatenate([x, i * 12 + notes], axis=0)
      return x.tolist()

k = Instrument()
k.play(notes=[45, 48, 52], dur=8)


s.gui(locals())