# Campana Umami Analytics

## Purpose

Measure whether visitors successfully listen to Campana, which official presets
they play, how long they remain, and whether exports complete. Analytics is for
aggregate product improvement, not user identification or operational API
monitoring.

## Privacy boundary

The Umami tracker loads only on the canonical GitHub Pages path:

```text
bassimatte.github.io/campana/
```

It remains disabled on local installations, repository forks, previews, and the
Railway backend. Search parameters and URL fragments are excluded, and the
tracker respects the browser's Do Not Track setting.

Campana does not send:

- Seeds or exact control values
- Filenames, audio, or generated content
- Raw errors or tracebacks
- Free-form text
- User or session identifiers supplied by Campana

All custom events and properties pass through explicit allowlists. Do not enable
session replay or heatmaps for Campana.

## Events

| Event | Meaning | Allowed properties |
|---|---|---|
| `campana_audio_started` | The first audio chunk was decoded and scheduled | `preset`, `texture`, latency bucket |
| `campana_audio_failed` | Initial playback or continued streaming failed | `stage`, coarse `reason` |
| `campana_listening_reached` | Active playback reached 30 seconds, 2 minutes, or 5 minutes | `preset`, `duration` |
| `campana_export_completed` | The rendered download was prepared successfully | `preset`, `format`, `duration` |
| `campana_export_failed` | Export start, render, or download failed | `stage`, `format` |

Preset analytics use the public values `sera`, `tempio`, `cristallo`,
`cattedrale`, `deriva`, `notte`, `festa`, and `aurora`. The legacy internal ID
`giardino` is mapped to `festa` before an event is sent.

## Shared Umami website

The Umami Cloud website limit is already reached, so Campana reuses the existing
`bassimatte.github.io` website used by Mantice. Every Campana pageview and event
has the Umami tag `campana`, and its URL path starts with `/campana/`. Custom
event names also start with `campana_`, making them identifiable in Umami's
Events tab even when no tag filter is active. These fields isolate Campana data
without consuming another website slot.

The shared Website ID is public tracker configuration, not an API key. No Umami
API key is stored in the application.

After deployment:

1. Open the shared `bassimatte.github.io` website in Umami.
2. Apply the filter **Tag = campana**. A **Path = /campana/** filter is an
   additional cross-check.
3. Open the deployed Campana page and verify a tagged page view in realtime.
4. Start a preset and verify a `campana_audio_started` event with its `preset`
   property.

## Suggested reports

Primary funnel:

```text
Page view
  -> campana_audio_started
  -> campana_listening_reached (30s)
  -> campana_listening_reached (2m)
  -> campana_export_completed
```

Keep the **Tag = campana** filter applied, then break down
`campana_audio_started`, `campana_listening_reached`, and
`campana_export_completed` by the `preset` property to compare actual preset
usage and depth of listening.
