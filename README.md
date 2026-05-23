# Campana

A browser-based bell synthesizer with additive synthesis, Freeverb reverb, stereo delay, and gapless streaming.

![Campana UI](screenshots/main%20ui.png)

## Features
- 4 bell textures: Tubular, Church, Singing Bowl, Crystal
- Freeverb global reverb (8 comb + 4 allpass)
- Stereo feedback delay with ping-pong
- Time scatter  float notes freely off the beat grid
- Octave spread, humanize, density controls
- Gapless chunk streaming via Web Audio API
- Dark / light theme

## Run
\\\
pip install -r requirements.txt
python main.py --gui
\\\
Then open http://127.0.0.1:8081
