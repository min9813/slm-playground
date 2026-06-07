"""Realtime speech-to-speech over WebRTC, powered by fastrtc.

This mounts a fastrtc `Stream` onto the existing FastAPI app (no Gradio). The
browser opens a WebRTC connection to `/webrtc/offer`, streams mic audio, and the
model's spoken reply streams back over the same connection. Reply text is pushed
to the client via Server-Sent Events at `/api/rtc/outputs`.
"""
from __future__ import annotations

import json

from backend.tts_engine import CHAT_SYSTEM_PROMPT, get_engine


def _handler(audio):
    """fastrtc ReplyOnPause handler: one user turn in, streamed reply out."""
    from fastrtc import AdditionalOutputs

    engine = get_engine()
    for kind, payload in engine.stream_reply(audio):
        if kind == "audio":
            yield payload  # (sample_rate, int16[1, N])
        else:  # ("text", partial_transcript)
            yield AdditionalOutputs({"role": "assistant", "content": payload})


def build_stream():
    from fastrtc import ReplyOnPause, Stream

    return Stream(
        handler=ReplyOnPause(_handler),
        modality="audio",
        mode="send-receive",
        # No STUN/TURN needed for direct LAN / Tailscale connections. Add ICE
        # servers here if reaching the server across NATs on the open internet.
    )


def mount_rtc(app) -> bool:
    """Mount the WebRTC routes + the SSE text endpoint. Returns False (and stays
    out of the way) if fastrtc isn't installed, so the rest of the app still runs."""
    try:
        from starlette.responses import StreamingResponse
    except Exception:
        return False

    try:
        stream = build_stream()
    except Exception as exc:  # fastrtc missing / import error
        print(f"[rtc] realtime chat disabled: {exc}")
        return False

    stream.mount(app)  # adds POST /webrtc/offer, WS /websocket/offer, telephone/*

    @app.get("/api/rtc/outputs")
    async def rtc_outputs(webrtc_id: str):
        async def event_stream():
            async for output in stream.output_stream(webrtc_id):
                data = output.args[0]
                yield f"event: output\ndata: {json.dumps(data)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/rtc/status")
    def rtc_status():
        return {"available": True, "system_prompt": CHAT_SYSTEM_PROMPT}

    print("[rtc] realtime speech-to-speech enabled at /webrtc/offer")
    return True
