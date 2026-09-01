# Socket command actions

## Game/Turns

The following prefixes are used by server-emitted messages:

| `type` Prefix | Target(s) | Meaning               |
| ------------- | --------- | --------------------- |
| `player_*`    | Lobby     | Player made an action |
| `private_*`   | Player    | Private message       |
| `game_*`      | Lobby     | Administrative update |

The following prefixes are emitted by clients to the server:

| `type` Prefix | Meaning                        |
| ------------- | ------------------------------ |
| `action_*`    | Player wants to make an action |

### Command Summary

| Client Main Actions         | Type String               | Special Action String | Payload | Notes |
|-----------------------------|---------------------------|-----------------------|---------|-------|
| Draw card from stockpile    | `action_draw_stockpile`   | n/a                   |         |       |
| Draw card from discard pile | `action_draw_discardpile` | n/a                   |         |       |
| Discard drawn card          | `action_discard`          | n/a                   |         |       |
| Replace card in hand        | `action_replace`          | n/a                   |         |       |
| 7/8 Peek at self            | `action_special`          | `peek_self`           |         |       |
| 9/10 Peek at other          | `action_special`          | `peek_other`          |         |       |
| J/Q Blind swap              | `action_special`          | `blind_swap`          |         |       |
| K Peek and swap             | `action_special`          | `peek_swap`           |         |       |
| Call "Cambia"               | `action_special`          | n/a                   |         |       |
| Snap card                   | `action_snap`             | n/a                   |         |       |

### Snap Action

Client sent payload:

```json
{
  "type": "action_snap",
  "card": {
    "id": "{uuid}"
  }
}
```

Upon a successful snap, the server should broadcast:

```json
{
  "type": "player_snap_success",
  "user": {
    "id": "{uuid}"
  },
  "card": {
    "id": "{uuid}",
    "rank": "Ace",
    "suit": "Spades",
    "value": 0,
    "idx": 0
  }
}
```

If fail:
```json
{
  "type": "player_snap_fail",
  "user": {
    "id": "{uuid}"
  },
  "card": {
    "id": "{uuid}",
    "rank": "Ace",
    "suit": "Spades",
    "value": 0,
    "idx": 0
  }
}
```

additionally, the `penalizeSnapFail()` function in `internal/game/game.go` calls `drawCard()` twice. Each `drawCard()` call should emit from the server to all clients, to notify them that a player is drawing new cards:

```json
{
  "type": "player_snap_penalty",
  "player": {
    "id": "{uuid}"
  },
  "card": {
    "id": "{uuid}"
  },
  "payload": {
    "count": 1,
    "total": 2,
    "stockpileSize": 29,
    "discardSize": 4
  }
}
```

`stockpileSize` and `discardSize` are the pile sizes after this penalty card was drawn, including
any reshuffle the draw forced. Clients set their displayed counts from these rather than
subtracting one per event: a penalty draw is not always one card off the stockpile.

and privately ONLY to the player being penalized:

```json
{
  "type": "private_snap_penalty",
  "card": {
    "id": "{uuid}",
    "idx": 0
  }
}
```

Note that no card details are to be revealed, just the new cards.

### Draw Action
      
When a draw action is taken, a client emits the following payload. {location} is either "stockpile" or "discardpile" - the latter is only allowed if the house rule flag "AllowDrawFromDiscardPile" is true.

```json
{
  "type": "action_draw_{location}"
}
```

After receiving this message, the server handles the action appropriately. If a re-shuffle of the discard pile is necessary (if the stockpile only has one card left at any point, the discard pile is reshuffled and set to the stockpile, all face down), then the server must emit this message:

```json: server -> all clients
{
  "type": "game_reshuffle_stockpile",
  "stockpileSize": 30 // where this number is the new size of the stockpile (num cards)
}
```

The server responds with a message to all players after the draw:
```json: server -> all clients
{
  "type": "player_draw_stockpile",
  "user": {
    "id": "{uuid}"
  },
  "card": {
    "id": "{uuid}"
  },
  "stockpileSize": 29
}
```

and messages ONLY to the player drawing the card:
```json
{
  "type": "private_draw_stockpile",
  "card": {
    "id": "{uuid}",
    "rank": "Ace",
    "suit": "Spades",
    "value": 1
  }
}
```

The player who has drawn the card can then decide to swap or discard:

1. if immediately discarding, the client will send this payload to the server:

    ```json
    {
      "type": "action_discard",
      "card": {
        "id": "{uuid}"
      }
    }
    ```

    After processing the command, and the server should respond to all players with:

    ```json
    {
      "type": "player_discard",
      "user": {
        "id": "{uuid}"
      },
      "card": {
        "id": "{uuid}",
        "rank": "Ace",
        "suit": "Spades",
        "value": 1
        // no idx here because player_discard action is only emitted after a drawn card is immediately discarded
      }
    }
    ```

2. If replaces a card already in their hand with the drawn card, the client sends this payload:

    ```json
    {
      "type": "action_replace",
      "card": {
        "id": "{uuid}",
        "idx": 0 // original idx of the card from the discarding player's hand
      }
    }
    ```

#### Special Card Discard Actions

As established previously, certain cards have special actions, which may be invoked always on a fresh card draw, and sometimes on a replace action (if the house rule for this setting is enabled).

If the card discarded in the previous step has a special action which can be utilized, these things happen:

1. The turn timer is reset to the value by the house rule (i.e. reset the timer so the player has time to decide their action)
2. The payload is broadcasted to all players:

    ```json
    {
      "type": "player_special_choice",
      "user": {
        "id": "{uuid}"
      },
      "card": {
        "id": "{uuid}",
        "rank": "7"
      },
      "special": "peek_self"
    }
    ```

    where "special" is a field of the following enums: `peek_self` (7 or 8), `peek_other` (9 or 10), `swap_blind` (J or Q), `swap_peek` (K)
    The clients intercept this message, and the player with the matching id will be faced with the decision of invoking the special turn option, with timer. Optionally, they will also be able to skip. They respond with the payload:

    An ability triggered by a **replace** (`allowReplaceAbilities`) carries `"payload": {"mandatory": true}` and **cannot be skipped**. The engine folds the decline into the discard action (`DiscardNoAbility` vs `DiscardWithAbility`) and offers an already-armed ability nothing but targets, so a `skip` against it is refused with `private_special_action_fail` and the prompt stands. Clients must not render a skip affordance for it; the turn timer resolves it by playing the first legal target (a King looks and then declines the swap). The same flag is carried in `private_sync_state` under `specialAction.mandatory` so a reconnecting client restores the prompt without the affordance.

    A replace only triggers an ability when the drawn card came from the **stockpile** (RULES.md 3B) and the ability has a legal target; a replace fed by a discard-pile draw emits no `player_special_choice` at all.

    ```json
    {
      "type": "action_special",
      "special": "peek_self", // can be peek_self, peek_other, swap_blind, or swap_look, OR skip - if they choose to skip this special action
      "card1": {
        "id": "{uuid}",
        "idx": 0
      },
      "card2": {
        "id": "{uuid}",
        "idx": 0
      }
    }
    ```

    Now, note that there are two card fields. This is because J/Q/K allows you to do a swap, requiring two cards to be named. If the action is of a 7/8/9/10 (e.g. `peek_*`), the card2 obj should be `null` or `undefined`. It doesn't have to be supplied.

3. The server receives the action special response, processes the action, and then broadcasts the update to all players. If a `peek` (self, other, or swap peek) action is taken, then a private message is sent to that action-taking player revealing the card options.

    ```json: server -> all clients
    {
      "type": "player_special_action", 
      "special": "peek_self", // or peek_other, or swap_blind, or swap_peek
      "card": {
        "id": "{uuid}", // note there is no further information revealed to all clients; just knowledge that this specific card was viewed
        "idx": 0
      }
    }
    ```

    ```json server -> client taking the action
    {
      "type": "private_special_action_success",
      "special": "peek_self",
      "card": {
        "id": "{uuid}",
        "idx": 0,
        "rank": "King",
        "suit": "Hearts",
        "value": -1
      }
    }
    ```

    a sample payload for a swap (no peek actions) is:

    ```json: server -> all clients
    {
      "type": "player_special_action", 
      "special": "swap_blind",
      "card1": {
        "id": "{uuid}",
        "idx": 0
      },
      "card2": {
        "id": "{uuid}",
        "idx": 0
      }
    }
    ```

    There is an additional caveat with any sort of swap action. If a target player has already called cambia, their cards cannot be moved (though, they can be viewed only either by a peek or swap peek). If a player attempts to make this action, they receive a private payload from the server, and must issue a new `action_special`.

    ```json: server -> client taking bad action
    {
      "type": "private_special_action_fail",
      "special": "swap_blind",
      "card1": {
        "id": "{uuid}",
        "idx": 0
      },
      "card2": {
        "id": "{uuid}",
        "idx": 0
      }
    }
    ```

4. The King card (`swap_peek`) is special, as the player can peek at any two cards first (even two cards in their own hand), then decide if they want to swap. This requires a second back forth talk. After discarding the king, the player then can choose two cards, similarly to `player_special_action` `special: swap_blind`. Except this time, the server will FIRST privately message the client user the two cards' identities, before allowing the player a chance to decide if they want to swap.

    ```json: client -> server (draws card from stockpile)
    {
      "type": "action_draw_stockpile"
    }
    ```

    ```json: server -> client (card info, private)
    {
      "type": "private_draw_stockpile",
      "card": {
        "id": "{uuid}",
        "rank": "King",
        "suit": "Clubs",
        "value": 13
      }
    }
    ```

    ```json: server -> all clients (announcing that a card was drawn)
    {
      "type": "player_draw_stockpile",
      "user": {
        "id": "{uuid}"
      },
      "card": {
        "id": "{uuid}"
      }
    }
    ```

    ```json: client -> server (client decides to immediately discard the king, triggering special card flow)
    {
      "type": "action_discard",
      "card": {
        "id": "{uuid}"
      }
    }
    ```

    ```json: server -> all clients (announcing that the card drawn was discarded, updating the discard pile locally and in the server)
    {
      "type": "player_discard",
      "user": {
        "id": "{uuid}"
      },
      "card": {
        "id": "{uuid}",
        "rank": "King",
        "suit": "Clubs",
        "value": 13
      }
    }
    ```

    ```json: server -> all clients
    {
      "type": "player_special_choice",
      "user": {
        "id": "{uuid}"
      },
      "card": {
        "id": "{uuid}",
        "rank": "King"
      },
      "special": "swap_peek"
    }
    ```

    ```json: client (taking the special action) -> server
    {
      "type": "action_special",
      "special": "swap_peek", // swap peek action means peek first then decide to swap
      "card1": {
        "id": "{uuid}",
        "user": {
          "id": "{uuid}" // the ID of the player the card belongs to
        },
        "idx": 0
      },
      "card2": {
        "id": "{uuid}",
        "user": {
          "id": "{uuid}" // the ID of the player the card belongs to
        },
        "idx": 0
      }
    }
    ```

    ```json: server -> client taking the action (updating the infomation revealed by taking the special action)
    {
      "type": "private_special_action_success", 
      "special": "swap_peek_reveal",
      "card1": {
        "id": "{uuid}",
        "rank": "Ace",
        "suit": "Spades",
        "value": 1,
        "idx": 0
      },
      "card2": {
        "id": "{uuid}",
        "rank": "Ace",
        "suit": "Diamonds",
        "value": 1,
        "idx": 0
      }
    }
    ```

    ```json: server -> all clients (to inform them the two cards that were selected and essentially picked up)
    {
      "type": "player_special_action", 
      "special": "swap_peek_reveal",
      "card1": {
        "user": {
          "id": "{uuid}" // the ID of the player the card belongs to
        },
        "id": "{uuid}",
        "idx": 0
      },
      "card2": {
        "user": {
          "id": "{uuid}" // the ID of the player the card belongs to
        },
        "id": "{uuid}",
        "idx": 0
      }
    }
    ```

    Now, if either of the cards selected belongs to a player that has called cambia, we cannot proceed with swapping. The special action will immediately end here--do not proceed to swap. Do not give the player a chance to choose different card(s).

    Otherwise, the server should reset the timer once more. The client then submits a final action to decide if they want to make the swap or not.

    ```json: client -> server (decides to swap)
    {
      "type": "action_special",
      "special": "swap_peek_swap", // or, if they decide to cancel, just "skip" - in which case there is no further payload required
      "card1": {
        "id": "{uuid}",
        "user": {
          "id": "{uuid}" // the ID of the player the card belongs to
        },
        "idx": 0
      },
      "card2": {
        "id": "{uuid}",
        "user": {
          "id": "{uuid}" // the ID of the player the card belongs to
        },
        "idx": 0
      }
    }
    ```

    Once this is complete, the server announces one last time to all players the result of the transaction.

    ```json: server -> all clients
    {
      "type": "player_special_action",
      "special": "swap_peek_swap", // or, skip
      "card1": {
        "id": "{uuid}",
        "user": {
          "id": "{uuid}" // the ID of the player the card belongs to
        },
        "idx": 0
      },
      "card2": {
        "id": "{uuid}",
        "user": {
          "id": "{uuid}" // the ID of the player the card belongs to
        },
        "idx": 0
      }
    }
    ```

5. If the player times out at any point and doesn't submit an action_special command in time, we default to `special: skip`, and the player forfeits their special action chance. The turn moves to the next player.

## Calling Cambia Action

If a player decides to call Cambia, their turn ends. All players will get another turn, and the game ends when the turn reaches the original caller. Whoever calls Cambia "locks" their hand, so their cards are unmoveable. However, other players can peek at them to gain more information. This state and locking mechanism should be fully implemented.

The client action payload for this will look like:

```json: client -> server (calling cambia)
{
  "type": "action_cambia"
}
```

the server will tell all clients:

```json: server -> all clients
{
  "type": "player_cambia",
  "user": {
    "id": "{id}"
  }
}
```

## Turn Timer and Current Turn Broadcast

Every time someone's turn is over, the server should automatically increment the current player turn marker. When this happens, the server must emit a message to all players:

```json: server -> all clients
{
  "type": "game_player_turn",
  "user": {
    "id": "{id}"
  }
}
```

## Hand visibility

While a round is running, no card in any hand is ever face-up on the wire, the requesting player's
own hand included. `private_sync_state` sends every hand slot, own and opponent, as an id plus its
index with `known: false` and no `rank`/`suit`/`value`, in every phase of a live round: during the
initial reveal, in live play, and in the repair snapshot a reconnecting player is sent. The round
ending is what turns them up, once, and that is the "Round-end reveal" section below.

```json: server -> one client (own seat inside private_sync_state)
{
  "playerId": "{uuid}",
  "handSize": 4,
  "revealedHand": [
    { "id": "{uuid}", "known": false, "idx": 0 },
    { "id": "{uuid}", "known": false, "idx": 1 },
    { "id": "{uuid}", "known": false, "idx": 2 },
    { "id": "{uuid}", "known": false, "idx": 3 }
  ],
  "drawnCard": { "id": "{uuid}", "known": true, "rank": "K", "suit": "S", "value": 13 }
}
```

The ids and indices are there for targeting, not for rendering: `action_special` names an own card
by id (`peek_self`, `swap_blind`, `swap_peek`), and the slot count is what a client draws card
backs from.

This mirrors the physical game: you are shown two cards before the deal, they go face-down with
everything else, and you play the round on memory. A snapshot that repeated a face you had already
been shown would make every reveal permanent and remove the memory element entirely.

Each reveal a player is entitled to therefore travels in its own event, once, and the client shows
it for that window and then turns the card back down:

| Reveal                        | Carrier                            | Window                              |
|-------------------------------|------------------------------------|-------------------------------------|
| Pregame peek                  | `private_initial_cards`            | the pre-game phase                  |
| Card drawn from a pile        | `private_draw_stockpile`           | until it is discarded or replaced   |
| 7/8 peek own, 9/10 peek other | `private_special_action_success`   | a short client-side hold            |
| King look (own and target)    | `private_special_action_success`   | the confirm step, then a hold       |

`drawnCard` in `private_sync_state` is the single exception, and only while it is pending: a card
drawn and not yet placed is in the player's hand rather than their fan, and the client needs its
face to choose between discarding and replacing it. Once it is placed, its slot is a face-down id
like every other.

A player who reconnects while the pre-game phase is still running is re-sent
`private_initial_cards` after their `private_sync_state`. That event is the only carrier of those
faces, so without the re-fire a reload during the reveal would cost the returning player the peek
for the whole round.

The server still records what each seat has legitimately been shown
(`CardUUIDTracker.SeenByPlayer`, keyed by card id so knowledge travels with a card across swaps).
That record is server-side reasoning about knowledge, not a rendering gate: nothing in it reaches
a client.

## Round-end reveal

RULES.md 3C ends a round with all cards revealed, and the service reveals on every terminal path it
reaches, not just a called Cambia: the turn cap, an exhausted stockpile, and a forfeit that empties
the table all run through the same `endGame`. A seat that forfeited is left out, since it is not
scored either.

The reveal is built once, in `endGame`, and carried by every frame that reports the result:

| Frame               | Who gets it                                        |
|---------------------|----------------------------------------------------|
| `game_end`          | every client at the table when the round ends       |
| `game_results`      | the same clients, and any that reconnect afterwards |
| `round_end`         | a ranked round that is not the match's last         |
| `match_end`         | the ranked match's last round                       |

`game_results` needs its own copy: the game is dropped from the store the moment it is emitted, so
a client that reconnects into the results is answered with the hub's held copy of that frame and
never sees `game_end`. Every frame spells the field `finalHands`, so one client shape reads them
all, and every one carries the same list:

```json: server -> all clients (finalHands, on game_end / game_results / round_end / match_end)
"finalHands": [
  {
    "playerId": "{uuid}",
    "cards": [
      { "id": "{uuid}", "idx": 0, "rank": "K", "suit": "H", "value": -1 },
      { "id": "{uuid}", "idx": 1, "rank": "4", "suit": "S", "value": 4 },
      { "id": "{uuid}", "idx": 2, "rank": "O", "suit": "R", "value": 0 },
      { "id": "{uuid}", "idx": 3, "rank": "9", "suit": "D", "value": 9 }
    ]
  }
]
```

`private_sync_state` reveals the same hands the same way, off the `gameOver` flag it already
carries: once a round is over every scored seat's `revealedHand` slots arrive `known: true` with
their `rank`/`suit`/`value`, own hand and opponents alike. That is the one carve-out from the
face-down rule above, so a client that resyncs or reconnects into a finished round sees the same
table as one that watched it end.

## Disconnect grace

A dropped socket does not forfeit on the spot. The seat is held for the lobby's
`disconnectGraceSec` house rule (default 60 seconds; 0 restores the immediate forfeit), and only
when that window closes does `forfeitOnDisconnect` take it. Three public events report where a
player stands, alongside the `connected`, `forfeited` and `reconnectDeadline` fields every
`private_sync_state` carries, so a client that joins or resyncs mid-window renders the same state
as one that watched it happen.

The table keeps playing throughout: the turn timer stays armed for a player inside their window
and its timeout draws and discards without touching their hand, which is the defensive play
RULES.md T5 and MATCHMAKING.md 8 describe. Pausing it instead would let any player freeze a game
for the length of the grace by pulling their network out.

```json: server -> all clients (socket dropped, seat held)
{
  "type": "player_reconnecting",
  "user": { "id": "{id}" },
  "payload": {
    "graceSeconds": 60,
    "deadline": 1756400000000,
    "serverNow": 1756399940000
  }
}
```

`deadline` is the absolute server-clock epoch-ms time the window closes and `serverNow` this
event's send time, the same skew-correction pair `game_player_turn` uses for the turn clock.

```json: server -> all clients (returned inside the window)
{
  "type": "player_reconnected",
  "user": { "id": "{id}" }
}
```

The returning player is separately sent a `private_sync_state` with the table as they left it:
same hand, same stockpile, same turn.

```json: server -> all clients (window closed with nobody there)
{
  "type": "player_forfeited",
  "user": { "id": "{id}" }
}
```

A forfeited player drops out of the final scoring, and the game ends there if it leaves one
player or fewer connected. A player who was away but had not yet forfeited is scored normally if
the table finishes without them, and one who comes back to a game that is still running takes
their seat back: the forfeit only sticks once the game itself is over.

The window opens from the deal onwards, not from the first turn: a drop during the initial card
reveal holds the seat and forfeits it on expiry exactly as a mid-game drop does, whether the
window closes before or after the reveal ends. Leaving the reveal exempt would have scored a
player who abandoned there as if they had played the game out, since scoring reads the forfeit
set rather than the connection flag.

## Reconnecting to a finished game

A hub holds its finished game for the post-game results interval, so a client that reconnects in
that window (a reload after the game ended, or the forfeited player's own tab coming back) is
sent the finished table's `private_sync_state` and then the terminal results frame again:
`game_results`, or `match_end` for a ranked circuit. Nothing else re-sends those scores: the
lobby snapshot a joining connection gets carries the phase but no results, which is why such a
reload used to land on a results screen with no winner and no scores.

Both of those frames carry `finalHands`, and the `private_sync_state` sent ahead of them has the
hands face-up, so the returning client gets the round-end reveal as well as the scores.
