from __future__ import annotations

"""New-wallet / new-participant detection.

Kalshi's public trades endpoint includes ticker/price/count/time and taker_side,
but does not expose "wallet" addresses like on-chain venues.

If you mean "new Kalshi participant" or "new counterparty", we need a data
source that includes participant identifiers (often not public).

Current status: placeholder.
"""


class NewWalletDetectorNotSupported(RuntimeError):
    pass
