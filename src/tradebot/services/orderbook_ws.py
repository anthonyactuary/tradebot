from __future__ import annotations

"""Optional: WebSocket orderbook updates.

Kalshi supports real-time orderbook updates (snapshot + deltas) over WebSockets.
This module is intentionally minimal and not yet wired into the main loop.

Why optional?
- WebSocket auth is required (API key during handshake).
- For an MVP, polling is simpler and still works.

If you want this wired in next, we can add:
- subscribe message format per docs
- local orderbook state + delta application
- reconnect + keepalive
"""

# Stub left intentionally.
