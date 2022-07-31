from pyo import *
import numpy as np

s = Server().boot()
s.start()

def randRange(a,b):
    r = b - a
    return a + r * np.random.rand()

class Base:
    def __init__(self, lfo=0.015, cutoff=20000, mul=1.):
        self.lfo = lfo
        self.cutoff = cutoff

        self.amp = Sig(mul)
        self.amp.ctrl()

        self.lfoMul = self.amp * Sine(freq=self.lfo * randRange(0.6, 1.5), phase=0.75).range(0, 1)


class Pad (Base):
    def __init__(self, note, bw, damp, reverb=.3, **kwargs):
        super().__init__(**kwargs)

        padSize = 262144 * 2
        padFreq = 440
        padRatio = s.getSamplingRate() / padSize

        self.note = note

        self.table = PadSynthTable(
          basefreq=padFreq,
          size=padSize,
          spread=1, # 1: def strings, 2:shallower/pure, in between: creepy (near 1, slight dissonnance) // freq = basefreq * pow(n, spread)
          bw=bw, # 20: org, 50: def strings, 70: dreamy more highs
          bwscl=1, # 1: def string, 2: dreamy, 3: flute/wind
          damp=damp, # 0.7: def, 1: big high synth, 0.5: mellow // amp = pow(damp, n)
          nharms=64, # 64: def
        )

        f = midiToHz(note)
        freqRatio = padRatio * f / padFreq

        self.lfoPan = Sine(freq=self.lfo * randRange(1.0, 2.0), phase=np.random.rand()).range(0.3, 0.7)

        self.pipeline = []
        self.pipeline.append(Osc(self.table, freq=freqRatio))
        self.pipeline.append(Compress(self.pipeline[-1], thresh=-30., ratio=2.))
        self.pipeline.append(Pan(self.pipeline[-1], outs=2, pan=self.lfoPan, mul=self.lfoMul))
        self.pipeline.append(Delay(self.pipeline[-1], delay=[.15, .2], feedback=.0))
        self.pipeline.append(Freeverb(self.pipeline[-1], size=[.98,.99], damp=.5, bal=reverb))
        self.pipeline.append(MoogLP(self.pipeline[-1], freq=self.cutoff, res=0.0))
        self.out = self.pipeline[-1]

        self.out.out()

class Recorded (Base):
    def __init__(self, soundPath, reverb=.3, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.table = SndTable(soundPath)
        self.freq = self.table.getRate()

        self.pipeline = []
        self.pipeline.append(Osc(table=self.table, freq=self.freq, mul=self.lfoMul))
        self.pipeline.append(Freeverb(self.pipeline[-1], size=[.79,.8], damp=.9, bal=reverb))
        self.pipeline.append(MoogLP(self.pipeline[-1], freq=self.cutoff, res=0.0))
        self.out = self.pipeline[-1]

        self.out.out()

class RecordedGuit (Base):
    def __init__(self, soundPath, reverb=.3, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.table = SndTable(soundPath)
        self.freq = self.table.getRate()

        self.pipeline = []
        self.pipeline.append(Osc(table=self.table, freq=self.freq, mul=self.lfoMul))
        self.pipeline.append(Freeverb(self.pipeline[-1], size=[.79,.8], damp=.9, bal=reverb))
        self.pipeline.append(Delay(self.pipeline[-1], delay=[.05, .06], feedback=.5))
        self.pipeline.append(MoogLP(self.pipeline[-1], freq=self.cutoff, res=0.0))
        self.out = self.pipeline[-1]

        self.out.out()

class ToneBeatSimple (Base):
    def __init__(self, note, dur, decay=.1, sustain=.2, reverb=.3, tone=[1, 0.3, 0.1, 0.02, 0.005], **kwargs) -> None:
        super().__init__(**kwargs)

        f = midiToHz(note)

        self.table = HarmTable(tone)

        durValue = dur
        if hasattr(dur, 'value'):
            if hasattr(dur.value, 'value'):
                durValue = dur.value.value
            else:
                durValue = dur.value

        self.env = Adsr(attack=.01, decay=decay, sustain=sustain, release=.01, dur=durValue)
        # self.env.ctrl()

        self.lfoPan = Sine(freq=self.lfo * randRange(1.0, 2.0), phase=np.random.rand()).range(0.3, 0.7)

        self.pipeline = []
        self.pipeline.append(Osc(table=self.table, freq=[f, f*1.01]))
        self.pipeline.append(Pan(self.pipeline[-1], outs=2, pan=self.lfoPan))
        self.pipeline.append(Freeverb(self.pipeline[-1], size=[.98,.99], damp=.5, bal=reverb))
        # self.pipeline.append(Delay(self.pipeline[-1], delay=[.05, .06], feedback=.5))
        self.out = self.pipeline[-1]

        self.out = self.lfoMul * self.env * self.out
        self.out.out()

        self.pat = Pattern(self.play, dur)
        self.pat.play()

    def play(self):
        self.env.play()

root = 27
dampFactor = 1.0
cutoff = 10000
volumeA = 1.0
volumeB = 1.0
dt = Sig(.2)
dt.ctrl()

# c = [SLMap(.0, 5., 'lin', "Volume", 1.0)]
# volumeB.ctrl(c, "Beat Volume")

oscs = [
    Pad(note=root     , bw=40, damp=dampFactor*0.9, cutoff=cutoff, reverb=.9, mul= volumeA * 0.800), #  0.546   0.000
    Pad(note=root+12  , bw=40, damp=dampFactor*0.8, cutoff=cutoff, reverb=.9, mul= volumeA * 0.800), #  0.542   0.000
    Pad(note=root+24  , bw=40, damp=dampFactor*0.8, cutoff=cutoff, reverb=.9, mul= volumeA * 0.326), #  0.204   0.000
    Pad(note=root+24+7, bw=40, damp=dampFactor*0.8, cutoff=cutoff, reverb=.9, mul= volumeA * 0.123), #  0.077   0.000
    Pad(note=root+36  , bw=50, damp=dampFactor*0.6, cutoff=cutoff, reverb=.9, mul= volumeA * 0.179), #  0.112   0.065
    Pad(note=root+36+7, bw=50, damp=dampFactor*0.6, cutoff=cutoff, reverb=.9, mul= volumeA * 0.056), #  0.035   0.023
    Pad(note=root+48  , bw=50, damp=dampFactor*0.6, cutoff=cutoff, reverb=.9, mul= volumeA * 0.080), #  0.050   0.073
    Pad(note=root+48+7, bw=50, damp=dampFactor*0.5, cutoff=cutoff, reverb=.9, mul= volumeA * 0.030), #  0.019   0.042
    Pad(note=root+60  , bw=50, damp=dampFactor*0.4, cutoff=cutoff, reverb=.9, mul= volumeA * 0.067), #  0.042   0.027
    Pad(note=root+60+7, bw=50, damp=dampFactor*0.3, cutoff=cutoff, reverb=.9, mul= volumeA * 0.019), #  0.012   0.023
    Pad(note=root+72  , bw=50, damp=dampFactor*0.3, cutoff=cutoff, reverb=.9, mul= volumeA * 0.040), #  0.042   0.027
    Pad(note=root+72+7, bw=50, damp=dampFactor*0.3, cutoff=cutoff, reverb=.9, mul= volumeA * 0.010), #  0.012   0.023
    # ToneBeatSimple(note=root, dur=64*dt, decay=12., sustain=.05,    reverb=.3, mul= volumeB * 0.110, tone=[.2, 1., 0.8, 0.6, 0.2, .05]), #  0.069   0.000
    # ToneBeatSimple(note=root, dur=dt, decay=.2, sustain=.0,    reverb=.3, mul= volumeB * 0.110, tone=[.8, 1., 0.8, 0.6, 0.2, .05]), #  0.069   0.000
    ToneBeatSimple(note=root   , dur=dt, decay=.8*dt.value, sustain=.0,    reverb=.9, mul= volumeB * 0.040, tone=[1., .8, .4, .2, .05]), #  0.069   0.000
    ToneBeatSimple(note=root+12, dur=dt, decay=.8*dt.value, sustain=.0,    reverb=.9, mul= volumeB * 0.040, tone=[1., .8, .4, .2, .05]), #  0.069   0.000
    ToneBeatSimple(note=root+24, dur=dt, decay=.8*dt.value, sustain=.0,    reverb=.9, mul= volumeB * 0.032, tone=[1., .8, .4, .2, .05]), #  0.069   0.000
    ToneBeatSimple(note=root+36, dur=dt, decay=.8*dt.value, sustain=.0,    reverb=.9, mul= volumeB * 0.010, tone=[1., .4, .2, .1, .05]), #  0.069   0.000
    # ToneBeatSimple(note=root+36+7, dur=dt, decay=.2, sustain=.0,    reverb=.8, mul= volume * 0.010, tone=[1., .8, .4, .2, .05]), #  0.069   0.000
    # ToneBeatSimple(note=root+36, dur=.5*dt, decay=.1, sustain=.2,   reverb=.3, mul= volume * 0.070, tone=[1, 0.3, 0.1, 0.02, 0.005]), #  0.123   0.000
    # # Recorded(soundPath="./data/Clean Combo#03.wav", cutoff=cutoff, mul=volume*4.0, reverb=0.),
    # # Recorded(soundPath="./data/Clean Combo#01.wav", cutoff=cutoff, mul=volume*2.0, reverb=0.5),
    # Recorded(soundPath="./data/Clean Combo#05.wav", cutoff=cutoff, mul=volume*3.0, reverb=0.5),
    # RecordedGuit(soundPath="./data/AmbientE Loops.Guit.1.wav", cutoff=cutoff, mul=volume*0.06, lfo=.003, reverb=0.5),
    # RecordedGuit(soundPath="./data/AmbientE Loops.Guit.3.wav", cutoff=cutoff, mul=volume*0.06, lfo=.003, reverb=0.5),
    # RecordedGuit(soundPath="./data/AmbientE Loops.Guit.4.wav", cutoff=cutoff, mul=volume*0.06, lfo=.003, reverb=0.5),
]

def printAmps():
    [print("{:.3f}".format(osc.amp.value.value)) for osc in oscs if hasattr(osc.amp.value, 'value')]
    print("-")

pool = Pattern(printAmps, 5.)
pool.play()

s.gui(locals())
