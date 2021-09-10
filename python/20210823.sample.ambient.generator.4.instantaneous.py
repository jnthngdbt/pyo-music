# 20210823.sample.ambient.generator.py

import audiosegment
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
import scipy.io.wavfile
import winsound
import os
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
  x *= A/np.max(np.abs(x)) # normalize the signal to span the int16 domain
  return x.astype(np.int16)

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

  F = np.abs(F)
  G = np.zeros((Ng, 2))

  f = np.fft.fftfreq(Nf)
  f = np.fft.fftshift(f)
  F = np.fft.fftshift(F, axes=0)

  g = np.linspace(f[0], f[-1], num=Ng)

  for i in np.arange(F.shape[1]):
    G[:,i] = np.interp(g, f, F[:,i])

  G = np.fft.fftshift(G)

  return G

def randomizePhase(F):
  F = np.abs(F) * np.exp(1j * np.random.rand(F.shape[0], F.shape[1]) * np.pi)
  return F

def applyWindow(x, w):
  Nw = len(w)
  Nh = int(0.5 * Nw)
  Nx = x.shape[0]

  for i in np.arange(x.shape[1]):
    x[0:Nh, i] = x[0:Nh, i] * w[0:Nh]
    x[Nx-Nh:, i] = x[Nx-Nh:, i] * w[Nw-Nh:]

  return x

## -------------------------------------------------------
def spectrogram(x, fs, Ts, Tw):
  stepLen = int(Ts * fs)
  windowLen = int(Tw * fs)

  maxPos = x.shape[0] - windowLen - 1
  nbSteps = max(1, int(maxPos / stepLen) - 1)
  nbChans = x.shape[1]

  S = np.zeros((windowLen, nbSteps, nbChans)) * np.exp(1j * np.zeros((windowLen, nbSteps, nbChans)))
  
  for i in np.arange(nbSteps):
    print('{0}/{1}'.format(i, nbSteps))
    pos = i * stepLen
    xi = x[pos: pos + windowLen, :]
    S[:, i, :] = np.fft.fft(xi, axis=0)

  return np.abs(S)

def smooth(x, n, a):
  return ndimage.convolve1d(x, np.ones(n), axis=a)

def crossFadeWindow(n): # for equal power fading
  hn = int(0.5 * n)
  t = np.arange(hn)
  x = np.sqrt(t)
  x = np.concatenate([x, np.flipud(x)])
  x = x / np.max(x)

  w = np.zeros(n)
  w[0:len(x)] = x # deal with odd window lenght, may miss 1 point
  return w

def crossFadeWindowParabola(n): # for equal power fading
  t = np.linspace(-1, 1, num=n)
  x = t ** 2
  x = 1 - x / np.max(x)
  return x

def crossFadeWindowCosine(n): # for equal power fading
  t = np.arange(n)
  x = np.sin(np.pi * t / n)
  return x

def reconstructSample(F, k, w, phases):
  F = interpolateFft(F, k)

  # Use random phases when cross-fading to avoid interferences and hear beats due to similar signals.
  if True: #len(phases) == 0: 
    phases = np.random.rand(F.shape[0], F.shape[1]) * 2.0 * np.pi

  F = np.abs(F) * np.exp(1j * phases)

  x = np.real(np.fft.ifft(F, axis=0))

  if w > 0:
    x = applyWindow(x, crossFadeWindow(int(w * x.shape[0])))

  return x, phases

def filterSound(x, fc, fs):
  sos = signal.butter(6, fc, 'lp', fs=fs, output='sos')
  x = signal.sosfilt(sos, x, axis=0)
  return x

def boostBass(x, k, fc, fs):
  sos = signal.butter(2, fc, 'lp', fs=fs, output='sos')
  return x + k * signal.sosfilt(sos, x, axis=0)

def exportCompressed(x, name, fs):
  y = audiosegment.from_numpy_array(discretize(x), fs)
  y.export(name)

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

## -------------------------------------------------------
# name = "03 Mission Two"
# name = "04 Mission Three"
# name = "05 Mission Four"
# name = "07 Mission Six"
# name = "08 Mission Seven"
# name = "09 Mission Eight"
# name = "11 Mission Ten"
# name = "Big Rock.1"
# name = "Alone.3"
# name = "Jump.12"
# name = "Press.5"
# name = "Late.06"
# name = "Sam Sung 3" # wow
# name = "Aly Wood 2"
# name = "Beverly Aly Hills 5"
# name = "insects"
# name = "smallthings" # t: 31
# name = "Sam Buca - Outdoor Tone Car"
# name = "Sam Buca - Outdoor Tone"
# name = "Tone night 21h Aug"
# name = "Tone outside 17h August"
# name = "Sam Buca - Outdoor Water"
# name = "Sam Buca - Indoor Water Mild"
# name = "Sam Buca - Indoor Water"
# name = "Tron Ouverture"
# name = "Background noise with voice"
# name = "TC RS212 vs SVT15E"
# name = "bass recording jam miche 1"
name = "Ambient input sing"

nameIn = "./data/" + name + ".m4a"

seg = audiosegment.from_file(nameIn)

fs = seg.frame_rate
x = seg.to_numpy_array()
x = x / x.max()

if x.ndim == 1:
  x = x.reshape(len(x), 1)
  x = np.concatenate([x, x], axis=1)

## -------------------------------------------------------
Ts = 0.05 # step duration
Tw = 0.25 # sample duration
lowPass = 12000
doBoostBass = False # when using a recording

x = filterSound(x, lowPass, fs)
if doBoostBass:
  x = boostBass(x, 5, 500, fs)

exportCompressed(x, "./songs/sample.ambient.generated.song.m4a", fs)

S = spectrogram(x, fs, Ts, Tw)
# S = smooth(S, 3, 1)

P = np.squeeze(np.sum(np.log(S, where=S>0), axis=2))

t = np.linspace(0, x.shape[0] / fs, num=S.shape[1])
f = np.fft.fftfreq(S.shape[0], 1 / fs)

## -------------------------------------------------------

# C = np.corrcoef(P.T)
# c = np.squeeze(np.sum(C, axis=0))
# ci = np.argsort(c)

## -------------------------------------------------------
Tk = 4.0 # desired final sample duration (slow down factor)
winRatio = 0.6
crossFadeRatio = 0.9 #1.5 * winRatio

nbFfts = S.shape[1]

plt.figure()

s = []
p = []
for i in np.arange(nbFfts):
  print('{0}/{1}'.format(i, nbFfts))
  Si = np.squeeze(S[:,i,:])
  si, p = reconstructSample(Si, Tk / Tw, winRatio, p)

  s, ti = mixSignal(s, si, int(crossFadeRatio * Tk * fs))

  if i in [0.5 * nbFfts - 1, 0.5 * nbFfts, 0.5 * nbFfts + 1]: # debug plots
    plt.plot(ti, si[:,0], alpha=0.6)

plt.plot(s[:,0], alpha=0.3)

# s = smooth(s, 100, 0)

# pathOut = "C:/Users/jgodbout/OneDrive/Documents/music/sample.ambient/"
pathOut = "./songs/"
nameOut = '{}{}.Ts{}ms.Tw{}ms.Tk{}ms.LP{}Hz.win{}.crossfade{}.m4a'.format(pathOut, name, int(Ts*1000), int(Tw*1000), int(Tk*1000), int(lowPass), int(winRatio*100), int(crossFadeRatio*100))
exportCompressed(s, "./songs/sample.ambient.generated.sample.m4a", fs)
exportCompressed(s, nameOut, fs)

# music_thread = Thread(target=lambda: playSound(s))
# music_thread.start()

## ------------------------------------------------------
maxFreq = 6000

fmax = argmax(f > maxFreq)
P0 = P[:fmax, :]
f0 = f[:fmax]
plt.figure()
plt.pcolormesh(t, f0, P0)

# plt.figure()
# plt.pcolormesh(t, t, C)

plt.figure()
plt.plot(crossFadeWindowCosine(500))

plt.show()
