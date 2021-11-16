# 20210823.sample.ambient.generator.py

import audiosegment
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
import scipy.io.wavfile
import winsound
import os
import math
from scipy import signal
from scipy import ndimage
from threading import Thread

matplotlib.style.use(['dark_background'])
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['figure.facecolor'] = '#222222'
plt.rcParams['axes.facecolor'] = '#222222'
plt.rcParams["font.size"] = "11"
plt.rcParams["font.family"] = "consolas"

plt.close('all')

## -------------------------------------------------------
def discretize(x, volume=1.0):
  A = volume * (2**15-1) # int16 scale factor; 2^16/2, since signed, -1 to avoid saturation
  y = x.copy()
  y *= A/np.max(np.abs(x)) # normalize the signal to span the int16 domain
  return y.astype(np.int16)

def write(x, name, volume=1.0):
  scipy.io.wavfile.write(name, fs, discretize(x, volume))

def loop(s, f):
  play(s, f, loop=True)

def play(s, f, volume=1.0, loop=False):
  f = "./songs/" + f
  write(s, f, volume=volume)
  play = True
  while (play):
    winsound.PlaySound(f, 0)
    play = loop

## -------------------------------------------------------
def interpolateFft(F, k):
  Nf = F.shape[0]
  Ng = int(Nf * k)

  Gr = np.zeros((Ng, 2))
  Gi = np.zeros((Ng, 2))
  G = np.zeros((Ng, 2))

  f = np.fft.fftfreq(Nf)
  f = np.fft.fftshift(f)
  F = np.fft.fftshift(F, axes=0)

  g = np.linspace(f[0], f[-1], num=Ng)

  for i in np.arange(F.shape[1]):
    Gr[:,i] = np.interp(g, f, F[:,i].real)
    Gi[:,i] = np.interp(g, f, F[:,i].imag)

  G = Gr + 1j * Gi
  G = np.fft.fftshift(G)

  return G

def randomizePhase(F):
  F = np.abs(F) * np.exp(1j * np.random.rand(F.shape[0], F.shape[1]) * 2.0 * np.pi)
  return F

def applyWindow(x, w):
  Nw = int(0.5 * len(w))
  Nx = x.shape[0]

  for i in np.arange(x.shape[1]):
    x[0:Nw, i] = x[0:Nw, i] * w[0:Nw]
    x[Nx-Nw:, i] = x[Nx-Nw:, i] * w[Nw:]

  return x

## -------------------------------------------------------
def spectrogram(x, fs, Ts, Tw):
  stepLen = int(Ts * fs)
  windowLen = int(Tw * fs)

  maxPos = x.shape[0] - windowLen - 1
  nbSteps = int(maxPos / stepLen) - 1
  nbChans = x.shape[1]

  S = np.zeros((windowLen, nbSteps, nbChans)) * np.exp(1j * np.zeros((windowLen, nbSteps, nbChans)))
  
  for i in np.arange(nbSteps):
    print('{0}/{1}'.format(i, nbSteps))
    pos = i * stepLen
    xi = x[pos: pos + windowLen, :]
    S[:, i, :] = np.fft.fft(xi, axis=0)

  return S

def computeFft(x, i, Tw):
  Nw = int(Tw * fs)
  xi = x[i: i + Nw, :]
  F = np.fft.fft(xi, axis=0)
  return F

def crossFadeWindow(n): # for equal power fading
  hn = int(0.5 * n)
  t = np.arange(hn)
  x = np.sqrt(t)
  x = np.concatenate([x, np.flipud(x)])
  x = x / np.max(x)

  w = np.zeros(n)
  w[0:len(x)] = x # deal with odd window lenght, may miss 1 point
  return w

def inverseFft(F, w):
  x = np.real(np.fft.ifft(F, axis=0))
  if w > 0.0:
    x = applyWindow(x, signal.windows.hann(int(w * x.shape[0])))
  return x

def filterSound(x, fc, fs):
  sos = signal.butter(6, fc, 'lp', fs=fs, output='sos')
  x = signal.sosfilt(sos, x, axis=0)
  return x

def boostBass(x, k, fc, fs):
  sos = signal.butter(2, fc, 'lp', fs=fs, output='sos')
  return x + k * signal.sosfilt(sos, x, axis=0)

def exportCompressed(x, name, album, fs, format = "mp3"):
  y = audiosegment.from_numpy_array(discretize(x), fs)
  info = {"title": name, "album": album, "artist": "Jonathan Godbout", "genre": "Space Ambient"}
  y.export("./songs/" + name + "." + format, format=format, tags=info) # NOTE: folder must exist

def playSound(x):
  tmpWav = "temp.wav"
  loop(x, tmpWav)

def mixSignal(s, x, n):
  if len(s) <= 0:
    return x, np.arange(x.shape[0])
  else:
    Ns = s.shape[0] 
    Nx = x.shape[0]
    Ny = Ns + Nx - n
    y = np.zeros((Ny, s.shape[1]))
    y[0:Ns, :] = s
    y[Ns-n:,:] += x
    t = np.arange(Ny)
    return y, t[Ns-n:]

def notch(F, f, fn, ti, lfo, phase):
  fi = argmax(f > fn)
  Nhw = int(0.5 * fi) # sample based half-bandwidth

  win = signal.windows.tukey(2 * Nhw)
  win = duplicate1dArrayToChannels(F.shape[1])

  notch = np.zeros(Fi.shape)
  i = fi-Nhw
  j = fi+Nhw

  notch[i:j, :] = win
  notch[-j:-i, :] = win # symetrical negative frequency part

  amp = np.sin(2.0 * np.pi * ti * lfo + phase)
  amp = np.clip(1.0 * amp, -1.0, 1.0)
  amp = 0.5 + 0.5 * amp

  notch = 1.0 - amp * notch

  return F * notch

def getBandWindow(f, f0, idx):
  currFreq = f0 * 2.**(idx)
  prevFreq = f0 * 2.**(idx-1)
  nextFreq = f0 * 2.**(idx+1)
  f1 = currFreq - 0.5 * (currFreq - prevFreq)
  f2 = currFreq + 0.5 * (nextFreq - currFreq)
  i1 = int(1.0 * argmax(f > f1)) # expand for overlap, % or prev frequency
  i2 = int(1.0 * argmax(f > f2)) # expand for overlap, % or next frequency

  Nw = i2-i1

  w = np.zeros(len(f))
  w[i1:i2] = signal.windows.boxcar(Nw) # not ideal, will be scaled

  return w

def showBands(f, bands):
  sum = np.zeros(f.shape)
  plt.figure()
  for i in np.arange(len(bands)):
    w = getBandWindow(f, bands[0], i)
    sum += w
    plt.plot(f, w)
  plt.plot(f, sum)

def duplicate1dArrayToChannels(x, nbChannels):
  return np.tile(x, (nbChannels, 1)).T # copy as 2 equal columns

def modulateSin(s, t):
  lfoFreq = 0.01 + 0.03 * np.random.rand()
  lfoPhase = 2.0 * np.pi * np.random.rand()
  amp = np.sin(2.0 * np.pi * t * lfoFreq + lfoPhase)
  amp = 0.5 + 0.5 * amp
  amp = duplicate1dArrayToChannels(amp, s.shape[1])
  return s * amp

def delay(x, n):
  x[n:,:] += x[0:-n]

def applyDelays(x, fs, nbDelays, delaySec):
  for _ in np.arange(nbDelays):
    jitter = 0.1 * np.random.rand()
    delay(x, int(delaySec * (1.0 + jitter) * fs))
  return x

## -------------------------------------------------------
# NOTE: if file not found error:
#       - pip install audiosegment, then ffmpeg (may need to go through choco)
#       - must run all thoses commands as admin, must relaunch vscode

# name = "03 Mission Two"                # 72*0.05, 88*0.05
# name = "04 Mission Three"              # 24*0.05, 38*0.05, 234*0.05
# name = "07 Mission Six"                # 78.55 85.8 88.45 120.15 // 331*0.05, 545*0.05, 1760*0.05
# name = "09 Mission Eight"                # 17.6, 49.05, 51.85, 54.35
name = "11 Mission Ten"                # 3.35, 24.65, 26.1, 29.9, 32.35, 727*0.05
# name = "Big Rock.1"                    # 127.5
# name = "Alone.3"         
# name = "Jump.12"                       # 12.05 35.95 50.45 56.15 68.7
# name = "Press.5"         
# name = "Late.06"         
# name = "Sam Sung 3"                    # 51.8
# name = "Aly Wood 2"                    # 52 (dude)
# name = "Beverly Aly Hills 5"           # t: 3.55, 7.5/7.9, 10.9, 12.8, 15.8/17.75 // 23.7, 27.0, 30.1, 32.5, 35.15
# name = "insects"   
# name = "smallthings"                   # t: 1.85, 2.45, 2.85, 16.6, 17, 21.85, 33, 43.1,  44.8, 45.5, 47.1, 47.9, 50.1
# name = "Background noise with voice"   # 43*0.05, 1.75
# name = "Tron Ouverture"                # 697*0.05, 843*0.05, 55.4 58.9, 107.9, 125.25  130.9 135.7

nameIn = "./data/" + name + ".m4a"

seg = audiosegment.from_file(nameIn)

x = seg.to_numpy_array()
x = x / x.max()

## -------------------------------------------------------
Tw = 1.0 # sample duration
Ts = 60 # desired final song duration
songWindowFactor = 0.2
doBoostBass = False # when using a recording

timePosSec = 3.35

firstBandFreq = 512
nbBands = 3 # (use as low pass filter) 0:64, 1:128, 2:256, 3:512, 4:1024, 5:2048, 6:4096, 7:8192

nbDelays = 5
delaySec = 1.0

## -------------------------------------------------------

fs = seg.frame_rate
bands = [firstBandFreq * 2**i for i in np.arange(nbBands)]
Nw = int(Tw*fs) # sample window
Ns = int(Ts*fs) # final song
Nc = x.shape[1] # number of channels
f = np.fft.fftfreq(Nw, 1 / fs)

nameInfo = '{}.filters.'.format(name)
nameParams = '{}.bands.Tw{}ms.Ts{}ms.f0{}Hz.{}bands'.format(name, int(Tw*1000), int(Ts*1000), firstBandFreq, nbBands)
## -------------------------------------------------------

if x.ndim == 1:
  x = x.reshape(len(x), 1)
  x = np.concatenate([x, x], axis=1)

print(bands)
showBands(f, bands)

if doBoostBass:
  x = boostBass(x, 5, 500, fs)

exportCompressed(x, "sample.ambient.input.song", name, fs)

tx = np.linspace(0, x.shape[0] / fs, num=x.shape[0])
ti = argmax(tx > timePosSec) # sample index to play

ts = np.linspace(0, Ts, num=Ns)

F = computeFft(x, ti, Tw)
F = interpolateFft(F, Ts / Tw)
F = randomizePhase(F)

f = np.fft.fftfreq(F.shape[0], 1 / fs)

Fs = inverseFft(F, songWindowFactor)
exportCompressed(Fs, "sample.ambient.generated.spectrum", name, fs)
exportCompressed(Fs, nameInfo + "spectrum." + nameParams, name, fs)

s = np.zeros((Ns, Nc))
for i in np.arange(len(bands)):
  print('{0}/{1}'.format(i, len(bands)))
  
  w = getBandWindow(f, bands[0], i)
  w = duplicate1dArrayToChannels(w, Nc)

  Fi = w * F
  Fi = randomizePhase(Fi)

  si = inverseFft(Fi, songWindowFactor)
  si = modulateSin(si, ts)

  exportCompressed(si, "sample.ambient.band{}".format(i), name, fs)

  s = s + si

# s = applyDelays(s, fs, nbDelays, delaySec)
s = applyWindow(s, signal.windows.hann(int(songWindowFactor * Ns)))

exportCompressed(s, nameInfo + nameParams, name, fs)
exportCompressed(s, "sample.ambient.generated.song", name, fs)

music_thread = Thread(target=lambda: playSound(s))
music_thread.start()

plt.show()
