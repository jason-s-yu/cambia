#!/usr/bin/env python3
"""Deploy smoke test for the Cambia game service.

Plays a complete 2-player game (host "A" + guest "B") against a running
Cambia service instance end to end: guest auth, public lobby creation,
WebSocket handshake, ready-up, countdown, pre-game reveal, and a full round
of legal play (always draw from the stockpile, discard the drawn card, skip
every special ability, never snap) until the server ends the game.

Protocol notes (read from service source, not guessed):
  - Guest auth: GET /user/guest mints/returns an auth_token cookie
    (service/internal/handlers/user.go EnsureEphemeralUser /
    GuestHandler). Two independent httpx.Client cookie jars give the two
    players distinct identities.
  - Lobby: POST /lobby/create defaults to a *private* lobby
    (lobby.NewLobbyWithDefaults), but the host is never added to
    lob.Users on creation, and the WS handshake's private-lobby gate
    (service/internal/handlers/ws.go) checks lob.Users before adding
    anyone -- so a private lobby's own host cannot open a WS connection to
    it without a separate invite path. This script uses type "public"
    instead: for public lobbies the WS handshake itself adds the
    connecting user to lob.Users unconditionally, so no REST join call is
    required to open the socket (the join endpoint is still exercised for
    the guest, for protocol coverage).
  - WS: GET-upgrade at /ws/{lobbyId}, subprotocol "cambia"
    (handlers.HubWSHandler / wsopts.AcceptOptions). Client frames are
    {"type": ..., "body": {...}, "last_seq": N}; server frames are
    {"seq": N, "type": ..., "payload": {...}} (internal/hub/messages.go,
    internal/hub/connection.go ReadPump).
  - Ready/countdown/start: "ready" with all joined players ready
    auto-starts a countdown (lobby.MarkUserReadyUnsafe -> hub.beginCountdown),
    which after CountdownDuration (default 3s) creates the game and fires
    "game_started", followed by a *fixed* 10s pre-game reveal
    (game.BeginPreGame) before the first "game_player_turn"
    (internal/game/game.go).
  - Actions: action_draw_stockpile (no body), action_discard
    ({"id": <drawn-card-uuid>}, id must equal the card revealed by the
    prior private_draw_stockpile event), action_special
    ({"special": "skip"} always accepted regardless of triggering rank --
    internal/game/special_actions.go ProcessSpecialAction), action_cambia
    (no body, legal only at the start of a turn before drawing --
    internal/game/game.go HandlePlayerAction). Replace is never used here
    (AllowReplaceAbilities defaults to false).
  - Game end: engine ends the round exactly one full turn after a Cambia
    call (RULES.md SS C); "game_end" broadcasts
    {"scores": {uuid: int}, "winner": uuid, ...} (internal/game/game.go
    endGame).

Usage:
  python3 cambia_smoke.py --base-url https://cambia.jasonyu.io [--timeout 180] [--verbose]

Exit code 0 on a fully verified game; nonzero with a printed reason otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.parse
from collections import deque
from typing import Any

import httpx
from websockets.asyncio.client import connect as ws_connect

CAMBIA_SUBPROTOCOL = "cambia"
STALL_TIMEOUT_SEC = 30.0
# Global turn index (game.CambiaGame.TurnID, 0-based, alternates host/guest)
# at which the host calls Cambia instead of drawing. TurnID 8 is the host's
# 5th turn (0, 2, 4, 6, 8 with host seated first).
CAMBIA_AT_TURN = 8


class SmokeError(Exception):
    """A failed assertion or unrecoverable protocol condition."""


def log(verbose: bool, name: str, msg: str) -> None:
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] [{name}] {msg}", file=sys.stderr)


def ws_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urllib.parse.urlunsplit((scheme, parsed.netloc, "", "", ""))


def remaining(deadline: float) -> float:
    return deadline - time.monotonic()


class Player:
    """One WS-connected game participant with its own cookie jar and identity."""

    def __init__(self, name: str, base_url: str, verbose: bool, deadline: float):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.verbose = verbose
        self.deadline = deadline
        self.http = httpx.Client(base_url=self.base_url)
        self.user_id: str | None = None
        self.auth_token: str | None = None
        self.raw_set_cookie: str | None = None
        self.lobby_id: str | None = None
        self.ws: Any = None
        self.last_seq = 0
        self.event_counts: dict[str, int] = {}
        self.recent_events: deque[dict] = deque(maxlen=5)
        self.turns_taken = 0
        self.last_turn_seen = -1
        self.final_payload: dict | None = None
        self.done = asyncio.Event()

    def _log(self, msg: str) -> None:
        log(self.verbose, self.name, msg)

    def _http_timeout(self) -> float:
        rem = remaining(self.deadline)
        if rem <= 0:
            raise SmokeError(f"{self.name}: global timeout exceeded before REST call")
        return min(rem, 30.0)

    def guest_login(self) -> None:
        resp = self.http.get("/user/guest", timeout=self._http_timeout())
        resp.raise_for_status()
        self.user_id = resp.json()["id"]
        set_cookie_values = resp.headers.get_list("set-cookie")
        auth_cookie = next(
            (v for v in set_cookie_values if v.lower().startswith("auth_token=")), None
        )
        self.raw_set_cookie = auth_cookie
        self.auth_token = self.http.cookies.get("auth_token")
        if not self.auth_token:
            raise SmokeError(f"{self.name}: /user/guest did not set an auth_token cookie")
        self._log(f"guest login ok, user_id={self.user_id}")

    def create_public_lobby(self) -> str:
        resp = self.http.post(
            "/lobby/create", json={"type": "public"}, timeout=self._http_timeout()
        )
        resp.raise_for_status()
        data = resp.json()
        self.lobby_id = data["id"]
        self._log(f"created public lobby {self.lobby_id}")
        return self.lobby_id

    def join_lobby(self, lobby_id: str) -> None:
        resp = self.http.post(f"/lobby/{lobby_id}/join", timeout=self._http_timeout())
        resp.raise_for_status()
        self.lobby_id = resp.json()["lobby_id"]
        self._log(f"joined lobby {self.lobby_id}")

    async def connect_ws(self, base_ws_url: str) -> None:
        if not self.lobby_id or not self.auth_token:
            raise SmokeError(f"{self.name}: connect_ws called before lobby/auth were set up")
        url = f"{base_ws_url}/ws/{self.lobby_id}"
        rem = remaining(self.deadline)
        if rem <= 0:
            raise SmokeError(f"{self.name}: global timeout exceeded before WS connect")
        self.ws = await asyncio.wait_for(
            ws_connect(
                url,
                subprotocols=[CAMBIA_SUBPROTOCOL],
                additional_headers={"Cookie": f"auth_token={self.auth_token}"},
                open_timeout=min(rem, 30.0),
                close_timeout=5.0,
            ),
            timeout=min(rem, 30.0),
        )
        if self.ws.subprotocol != CAMBIA_SUBPROTOCOL:
            raise SmokeError(
                f"{self.name}: WS handshake did not negotiate the '{CAMBIA_SUBPROTOCOL}' "
                f"subprotocol (got {self.ws.subprotocol!r})"
            )
        status = getattr(getattr(self.ws, "response", None), "status_code", None)
        if status is not None and status != 101:
            raise SmokeError(f"{self.name}: WS handshake returned status {status}, expected 101")
        self._log(f"WS connected (subprotocol={self.ws.subprotocol!r}, status={status})")

    async def send(self, msg_type: str, body: dict | None = None) -> None:
        frame: dict[str, Any] = {"type": msg_type, "last_seq": self.last_seq}
        if body is not None:
            frame["body"] = body
        await self.ws.send(json.dumps(frame))
        self._log(f"-> {msg_type} {body if body else ''}")

    async def recv(self) -> dict:
        rem = remaining(self.deadline)
        if rem <= 0:
            raise SmokeError(f"{self.name}: global timeout exceeded while waiting for an event")
        try:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=min(rem, STALL_TIMEOUT_SEC))
        except asyncio.TimeoutError:
            dump = "\n".join(json.dumps(e) for e in self.recent_events)
            raise SmokeError(
                f"{self.name}: stalled {STALL_TIMEOUT_SEC:.0f}s with no event. "
                f"Last {len(self.recent_events)} events:\n{dump}"
            )
        env = json.loads(raw)
        self.last_seq = max(self.last_seq, env.get("seq", 0))
        etype = env.get("type", "?")
        self.event_counts[etype] = self.event_counts.get(etype, 0) + 1
        self.recent_events.append(env)
        if etype == "error":
            print(f"[{self.name}] SERVER ERROR EVENT: {env}", file=sys.stderr)
        self._log(f"<- {etype} seq={env.get('seq')}")
        return env

    async def play(self, is_host: bool) -> None:
        """Drive one player through ready -> countdown -> full game -> game_end."""
        await self.send("ready")
        while True:
            env = await self.recv()
            etype = env.get("type")
            payload = env.get("payload") or {}

            if etype == "error":
                continue

            # The hub discards any client message whose last_seq lags the hub's
            # sequence (a concurrent broadcast is enough) and answers with
            # sync_state instead of applying it. A human re-clicks Ready; this
            # script re-asserts it whenever a lobby snapshot shows us un-ready
            # in an open/ready_check phase. recv() has already advanced
            # last_seq past the snapshot, so the retry carries a fresh seq.
            if etype in ("lobby_state", "sync_state", "ready_update"):
                if payload.get("phase") in ("open", "ready_check", None):
                    users = (payload.get("lobby_status") or {}).get("users") or []
                    me = next((u for u in users if u.get("id") == self.user_id), None)
                    if me is not None and not me.get("is_ready"):
                        await self.send("ready")
                continue

            if etype == "game_end":
                self.final_payload = payload
                self.done.set()
                return

            if etype == "private_draw_stockpile":
                card = payload.get("card") or {}
                card_id = card.get("id")
                if not card_id:
                    raise SmokeError(f"{self.name}: private_draw_stockpile missing card.id")
                await self.send("action_discard", {"id": card_id})
                continue

            if etype == "player_special_choice":
                user = payload.get("user") or {}
                if user.get("id") == self.user_id:
                    await self.send("action_special", {"special": "skip"})
                continue

            if etype == "game_player_turn":
                user = payload.get("user") or {}
                if user.get("id") != self.user_id:
                    continue
                turn_num = (payload.get("payload") or {}).get("turn", -1)
                self.last_turn_seen = turn_num
                self.turns_taken += 1
                if is_host and turn_num >= CAMBIA_AT_TURN:
                    await self.send("action_cambia")
                else:
                    await self.send("action_draw_stockpile")
                continue

            # lobby_state, ready_update, phase_change, game_started,
            # private_initial_cards, player_draw_stockpile,
            # player_discard, player_snap_*, private_sync_state, etc.:
            # no client action required.

    def close(self) -> None:
        self.http.close()


async def run_smoke(base_url: str, timeout: float, verbose: bool) -> tuple[Player, Player]:
    deadline = time.monotonic() + timeout
    ws_url_base = ws_base_url(base_url)

    host = Player("A(host)", base_url, verbose, deadline)
    guest = Player("B(guest)", base_url, verbose, deadline)

    try:
        host.guest_login()
        guest.guest_login()

        scheme = urllib.parse.urlsplit(base_url).scheme
        for p in (host, guest):
            if not p.raw_set_cookie:
                raise SmokeError(f"{p.name}: no Set-Cookie: auth_token=... header observed")
            lowered = p.raw_set_cookie.lower()
            has_secure = "secure" in lowered
            has_lax = "samesite=lax" in lowered
            if scheme == "https":
                if not has_secure:
                    raise SmokeError(f"{p.name}: auth_token cookie missing Secure attribute (https deploy)")
                if not has_lax:
                    raise SmokeError(f"{p.name}: auth_token cookie missing SameSite=Lax attribute (https deploy)")
            else:
                log(verbose, p.name, f"http base-url: Secure/SameSite not enforced (raw={p.raw_set_cookie})")

        lobby_id = host.create_public_lobby()
        guest.join_lobby(lobby_id)

        await asyncio.gather(host.connect_ws(ws_url_base), guest.connect_ws(ws_url_base))

        await asyncio.gather(host.play(is_host=True), guest.play(is_host=False))

        return host, guest
    finally:

        async def _close_ws(p: Player) -> None:
            try:
                if p.ws is not None:
                    await p.ws.close()
            except Exception:
                pass

        await asyncio.gather(_close_ws(host), _close_ws(guest))
        host.close()
        guest.close()


def merged_event_counts(host: Player, guest: Player) -> dict[str, int]:
    merged: dict[str, int] = {}
    for counts in (host.event_counts, guest.event_counts):
        for k, v in counts.items():
            merged[k] = merged.get(k, 0) + v
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="e.g. https://cambia.jasonyu.io or http://localhost:8088")
    parser.add_argument("--timeout", type=float, default=180.0, help="whole-run bound in seconds (default 180)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    start = time.monotonic()
    try:
        host, guest = asyncio.run(run_smoke(args.base_url, args.timeout, args.verbose))
    except SmokeError as e:
        print(f"SMOKE FAILED: {e}", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as e:
        print(f"SMOKE FAILED: HTTP error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 - top-level catch-all for a clear nonzero exit
        print(f"SMOKE FAILED: unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - start

    if not host.final_payload or not guest.final_payload:
        print("SMOKE FAILED: game_end payload missing on one or both connections", file=sys.stderr)
        return 1

    winner = host.final_payload.get("winner")
    counts = merged_event_counts(host, guest)
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    turns_played = max(host.last_turn_seen, guest.last_turn_seen) + 1

    print(
        f"SMOKE OK: turns_played={turns_played} host_turns={host.turns_taken} "
        f"guest_turns={guest.turns_taken} winner={winner} elapsed={elapsed:.1f}s "
        f"scores={host.final_payload.get('scores')} events=[{counts_str}]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
