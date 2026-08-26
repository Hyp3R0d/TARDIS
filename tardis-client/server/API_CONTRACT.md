# TARDIS Studio proxy contract

The browser talks only to the local Express server. The proxy keeps the TARDIS
inference credential private and forwards the documented asynchronous video API.

## Create

`POST /api/generations`

```json
{
  "prompt": "A cinematic ...",
  "imageData": "data:image/png;base64,...",
  "settings": {
    "quality": "speed",
    "size": "1280x720",
    "fps": 30,
    "duration": 5,
    "withAudio": false
  }
}
```

`prompt` is required and is limited to 512 characters. `imageData` is optional,
but when present it must be a PNG/JPEG Data URL whose decoded bytes are at most
5 MB. The proxy maps this to the TARDIS service's reference-image field. The response is a
small client-facing envelope containing `id`, `requestId`, `taskStatus`, and
`model`.

The upstream request is sent to the configured TARDIS inference base URL at
`/tardis/v1/videos` with `model: tardis`. The bearer credential is read only
from `TARDIS_INFERENCE_API_KEY` (or an equivalent secret-manager entry) in the
server environment; it is never accepted from the browser.

## Poll

`GET /api/generations/:id`

The proxy calls `GET /tardis/v1/videos/{id}` on the configured TARDIS inference
service.
The normalized response is:

```json
{
  "id": "...",
  "status": "PROCESSING|SUCCESS|FAIL",
  "videoUrl": "https://.../video.mp4",
  "coverUrl": "https://.../cover.jpg",
  "error": null,
  "model": "tardis"
}
```

`videoUrl` and `coverUrl` are populated from the first item of upstream
`video_result` after `SUCCESS`. These are temporary provider URLs, so a client
that needs durable history should download/copy the asset to local storage.
