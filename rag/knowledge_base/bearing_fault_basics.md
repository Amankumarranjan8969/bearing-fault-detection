# Rolling bearing faults: basics

A rolling bearing has an outer race, an inner race, rolling elements (balls or rollers) and a cage. A local defect
on one of these parts produces a short impact every time a rolling element passes over it. Each impact excites a
structural resonance, so the vibration signal shows repetitive bursts at a fault-specific rate.

Outer-race faults are stationary relative to the load zone, so impacts have a fairly constant amplitude. Inner-race
faults rotate with the shaft, so impact amplitude is modulated at the shaft frequency and sidebands appear around the
fault frequency. Rolling-element faults often appear at twice the ball spin frequency, because a damaged ball
contacts both races in one revolution, and they are usually modulated by the cage frequency.

Kurtosis and crest factor rise when a signal contains sharp impacts, which makes them early indicators of a
localised defect. RMS rises later, when damage is widespread. Envelope analysis (band-pass around a resonance,
Hilbert envelope, then FFT) reveals the repetition rate of the impacts even when they are hidden in broadband noise.
