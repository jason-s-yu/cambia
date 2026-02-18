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
  }
}
```

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
      "special": "swap_peek_swap", // or, if they decide to cancel, just "skip" -- in which case there is no further payload required
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
