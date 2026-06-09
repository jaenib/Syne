"""Synthesize a reactive test track with clear, varied musical events:
intro kick groove -> build (riser + faster hats) -> drop -> outro, with chord
changes. Designed to make a visualizer's reactivity obvious and measurable."""
import numpy as np, soundfile as sf, sys

sr = 22050
bpm = 120.0
beat = 60.0 / bpm                      # 0.5s
def env(n, a=0.002, d=0.12):           # percussive AD envelope
    t = np.arange(n)/sr
    return np.minimum(t/a, 1.0) * np.exp(-t/d)

def kick(dur=0.18, f0=110, f1=45):
    n=int(dur*sr); t=np.arange(n)/sr
    f=f1+(f0-f1)*np.exp(-t/0.03)
    return np.sin(2*np.pi*np.cumsum(f)/sr)*env(n,0.001,0.10)*1.0

def hat(dur=0.05):
    n=int(dur*sr); return np.random.default_rng().standard_normal(n)*env(n,0.0005,0.02)*0.35

def tone(freqs, dur, amp=0.25):
    n=int(dur*sr); t=np.arange(n)/sr
    s=sum(np.sin(2*np.pi*f*t) for f in freqs)/len(freqs)
    a=np.minimum(t/0.01,1.0)*np.minimum((dur-t)/0.05,1.0)
    return s*a*amp

chords=[[130.81,164.81,196.0],[110.0,130.81,164.81],[87.31,110.0,130.81],[98.0,123.47,196.0]]
total=14.0; y=np.zeros(int(total*sr))
def add(sig, t0):
    i=int(t0*sr); j=min(i+len(sig),len(y)); y[i:j]+=sig[:j-i]

# section layout (seconds): 0-4 groove, 4-7 build, 7-7.4 drop gap, 7.4-11 full, 11-14 outro
b=0.0; bi=0
while b < total:
    sec = 0 if b<4 else (1 if b<7 else (2 if b<7.4 else (3 if b<11 else 4)))
    if sec==2:                          # drop gap: near silence
        b+=beat; bi+=1; continue
    # kick on every beat (full sections punch harder)
    amp = 1.2 if sec in (3,) else (0.9 if sec in (0,1) else 0.0)
    if sec!=4: add(kick()*amp, b)
    else: 
        if bi%2==0: add(kick()*0.7, b)
    # hats on 8ths (faster in build)
    sub = 4 if sec==1 else 2
    for k in range(sub):
        add(hat(), b+k*beat/sub)
    # chord per 2 beats
    if bi%2==0:
        ch=chords[(bi//2)%len(chords)]
        add(tone(ch, beat*2, amp=0.22 if sec!=2 else 0), b)
    # build riser
    if sec==1:
        n=int(beat*sr); t=np.arange(n)/sr
        add(np.sin(2*np.pi*(300+ (b-4)/3*1200)*t)*0.05*(t/beat), b)
    b+=beat; bi+=1

y=y/np.max(np.abs(y)+1e-9)*0.95
sf.write(sys.argv[1] if len(sys.argv)>1 else '/tmp/reactive.wav', y.astype(np.float32), sr)
print("wrote", sys.argv[1] if len(sys.argv)>1 else '/tmp/reactive.wav', f"{total}s")
