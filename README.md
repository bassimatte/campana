# Campana

A browser-based bell synthesizer with additive synthesis, Freeverb reverb, stereo delay, and gapless streaming.

![Campana UI](screenshots/main%20ui.png)

Production API: https://campana-production.up.railway.app/

## Features
- 6 bell textures: Tubular, Church, Singing Bowl, Crystal, Bronze, Handbell
- Preset-specific generative grammars, including monumental tolling and agile carillon patterns
- Freeverb global reverb (8 comb + 4 allpass)
- Stereo feedback delay with ping-pong
- Time scatter: float notes freely off the beat grid
- Octave spread, humanize, density controls
- Gapless chunk streaming via Web Audio API
- Dark / light theme

## Run
```sh
pip install -r requirements.txt
python main.py --gui
```
Then open http://127.0.0.1:8081

## Deploy

Campana deploys on Railway using the repository `Dockerfile`. Railway injects
the `PORT` environment variable; the container falls back to port `10000` for
local Docker runs.

### Export resource controls

Background exports are rendered and converted through temporary files rather
than retained as encoded byte strings in process memory. Completed, failed, and
abandoned jobs expire automatically, downloaded files are deleted after the
response finishes, and concurrent rendering is limited independently from the
total job-record safeguard.

The defaults can be tuned with Railway environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `CAMPANA_RENDER_MAX_CONCURRENT` | `2` | Maximum simultaneous export renders |
| `CAMPANA_RENDER_JOB_TTL` | `600` | Job and temporary-file lifetime in seconds |
| `CAMPANA_RENDER_CLEANUP_INTERVAL` | `30` | Periodic cleanup interval in seconds |
| `CAMPANA_RENDER_JOBS_MAX` | `20` | Maximum retained job records |

When render capacity is reached, `/api/render` returns HTTP `429` with a
`Retry-After` header instead of starting another memory-intensive worker.

## Analytics

The canonical GitHub Pages app supports privacy-safe Umami product analytics.
See [ANALYTICS.md](ANALYTICS.md) for the event schema, privacy boundary, and
activation instructions.
