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

def reconstructSample(F, k, w):
  F = interpolateFft(F, k)
  F = randomizePhase(F)

  x = np.real(np.fft.ifft(F, axis=0))
  # x = applyWindow(x, crossFadeWindow(int(w * x.shape[0])))
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

def duplicate1dArrayToChannels(x, nbChannels):
  return np.tile(x, (nbChannels, 1)).T # copy as 2 equal columns
  
def computeFft(x, i, Tw):
  Nw = int(Tw * fs)
  w = signal.windows.hann(Nw)
  w = duplicate1dArrayToChannels(w, x.shape[1])
  xi = x[i: i + Nw, :]
  xw = w * xi
  F = np.fft.fft(xw, axis=0)
  return F

def delay(x, n):
  x[n:,:] += x[0:-n]

## -------------------------------------------------------
# NOTE: if file not found error:
#       - pip install audiosegment, then ffmpeg (may need to go through choco)
#       - must run all thoses commands as admin, must relaunch vscode

# name = "03 Mission Two"                # 72*0.05, 88*0.05
# name = "04 Mission Three"              # 24*0.05, 38*0.05, 234*0.05
# name = "07 Mission Six"                # 78.55 85.8 88.45 120.15 // 331*0.05, 545*0.05, 1760*0.05
# name = "09 Mission Eight"              # 17.6, 49.05, 51.85, 54.35
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
# name = "TC RS212 vs SVT15E" # 13.25
# name = "bass recording jam miche 1"

nameIn = "./data/" + name + ".m4a"

seg = audiosegment.from_file(nameIn)

fs = seg.frame_rate
x = seg.to_numpy_array()
x = x / x.max()

if x.ndim == 1:
  x = x.reshape(len(x), 1)
  x = np.concatenate([x, x], axis=1)

## -------------------------------------------------------
positions = [24.65, 26.1, 29.9, 32.35]

Tw = 0.25 # sample duration
Tk = 16.0 # desired final sample duration (slow down factor)

Ns = 1 # number of times the sample is delayed
Np = len(positions)

sampleWinRatio = 0.9
crossFadeRatio = 0.1

lowPass = 6000
doBoostBass = False # when using a recording

## -------------------------------------------------------

x = filterSound(x, lowPass, fs)
if doBoostBass:
  x = boostBass(x, 5, 500, fs)

exportCompressed(x, "sample.ambient.input.song", name, fs)

# S = spectrogram(x, fs, Ts, Tw)
# S = smooth(S, 3, 1)

# P = np.squeeze(np.sum(np.log(S, where=S>0)q, axis=2))

# t = np.linspace(0, x.shape[0] / fs, num=S.shape[1])
# f = np.fft.fftfreq(S.shape[0], 1 / fs)

## -------------------------------------------------------

plt.figure()

t = np.linspace(0, x.shape[0] / fs, num=x.shape[0])

s = []
p = []
ii = 0
for i in np.arange(Np):

  for j in np.arange(Ns): 
    ii += 1
    print('{0}/{1}'.format(ii, Np * Ns))

    ti = argmax(t > positions[i]) # sample index to play
    print(ti)
    print(positions[i])

    F = computeFft(x, ti, Tw)
    si = reconstructSample(F, Tk / Tw, sampleWinRatio)
    s, ti = mixSignal(s, si, int(crossFadeRatio * Tk * fs))

plt.plot(s[:,0], alpha=0.3)

for i in np.arange(1, 100):
  delay(s, int(i * fs * 0.05))


nameOut = '{}.Tw{}ms.Tk{}ms.LP{}Hz.win{}.crossfade{}.m4a'.format(name, int(Tw*1000), int(Tk*1000), int(lowPass), int(sampleWinRatio*100), int(crossFadeRatio*100))
exportCompressed(s, "sample.ambient.generated.song.m4a", name, fs)
exportCompressed(s, nameOut, name, fs)

music_thread = Thread(target=lambda: playSound(s))
music_thread.start()

## ------------------------------------------------------
maxFreq = 6000

# fmax = argmax(f > maxFreq)
# P0 = P[:fmax, :]
# f0 = f[:fmax]
# plt.figure()
# plt.pcolormesh(t, f0, P0)

# plt.figure()
# plt.pcolormesh(t, t, C)

plt.figure()
plt.plot(crossFadeWindowCosine(500))

plt.show()
