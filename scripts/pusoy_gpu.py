#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

RANKS = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
SUITS = ["D", "C", "H", "S"]
KIND_TO_INDEX = {
    "single": 0,
    "pair": 1,
    "triple": 2,
    "straight": 3,
    "flush": 4,
    "full-house": 5,
    "four-kind": 6,
    "straight-flush": 7,
}
FIVE_KIND_ORDER = {
    "straight": 0,
    "flush": 1,
    "full-house": 2,
    "four-kind": 3,
    "straight-flush": 4,
}
STRAIGHT_PATTERNS = [
    ["3", "4", "5", "6", "7"],
    ["4", "5", "6", "7", "8"],
    ["5", "6", "7", "8", "9"],
    ["6", "7", "8", "9", "10"],
    ["7", "8", "9", "10", "J"],
    ["8", "9", "10", "J", "Q"],
    ["9", "10", "J", "Q", "K"],
    ["10", "J", "Q", "K", "A"],
    ["A", "2", "3", "4", "5"],
    ["2", "3", "4", "5", "6"],
    ["J", "Q", "K", "A", "2"],
]
STRAIGHT_SIGNATURES = {
    tuple(sorted(RANKS.index(rank) for rank in pattern)): index for index, pattern in enumerate(STRAIGHT_PATTERNS)
}
STATE_DIM = 216
MOVE_DIM = 62


@dataclass(frozen=True)
class Combo:
    kind: str
    cards: tuple[int, ...]
    size: int
    tiebreak: tuple[float, ...]


@dataclass
class Game:
    hands: list[list[int]]
    current_player: int
    active_combo: Combo | None = None
    last_player: int | None = None
    passes_since_play: int = 0
    finished: list[int] | None = None
    played: list[int] | None = None
    played_by_player: list[list[int]] | None = None
    turns: int = 0

    def __post_init__(self) -> None:
        if self.finished is None:
            self.finished = []
        if self.played is None:
            self.played = []
        if self.played_by_player is None:
            self.played_by_player = [[], []]


class PusoyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_fc = nn.Linear(STATE_DIM, 128)
        self.move_fc = nn.Linear(MOVE_DIM, 64)
        self.policy_fc = nn.Linear(192, 64)
        self.policy_out = nn.Linear(64, 1)
        self.value_fc = nn.Linear(128, 64)
        self.value_out = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor, move: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state_h = F.relu(self.state_fc(state))
        move_h = F.relu(self.move_fc(move))
        policy_h = F.relu(self.policy_fc(torch.cat([state_h, move_h], dim=-1)))
        logits = self.policy_out(policy_h).squeeze(-1)
        value = torch.tanh(self.value_out(F.relu(self.value_fc(state_h)))).squeeze(-1)
        return logits, value


def card_rank(card: int) -> int:
    return card // 4


def card_suit(card: int) -> int:
    return card % 4


def card_label(card: int) -> str:
    return f"{RANKS[card_rank(card)]}{SUITS[card_suit(card)]}"


def classify(cards_in: Iterable[int]) -> Combo | None:
    cards = tuple(sorted(cards_in))
    size = len(cards)
    ranks: dict[int, list[int]] = {}
    for card in cards:
        ranks.setdefault(card_rank(card), []).append(card)

    if size == 1:
        return Combo("single", cards, 1, (float(cards[0]),))
    if size == 2 and len(ranks) == 1:
        return Combo("pair", cards, 2, (float(next(iter(ranks))), float(max(card_suit(card) for card in cards))))
    if size == 3 and len(ranks) == 1:
        return Combo("triple", cards, 3, (float(next(iter(ranks))),))
    if size != 5:
        return None

    rank_values = sorted(ranks)
    straight_order = STRAIGHT_SIGNATURES.get(tuple(rank_values))
    is_flush = all(card_suit(card) == card_suit(cards[0]) for card in cards)
    groups = sorted(ranks.items(), key=lambda item: len(item[1]), reverse=True)

    if straight_order is not None and is_flush:
        top_rank = max(rank_values)
        top_suit = max(card_suit(card) for card in cards if card_rank(card) == top_rank)
        return Combo("straight-flush", cards, 5, (4.0, float(straight_order), float(top_suit)))
    if len(groups[0][1]) == 4:
        return Combo("four-kind", cards, 5, (3.0, float(groups[0][0])))
    if len(groups[0][1]) == 3 and len(groups[1][1]) == 2:
        return Combo("full-house", cards, 5, (2.0, float(groups[0][0])))
    if is_flush:
        descending_ranks = sorted((card_rank(card) for card in cards), reverse=True)
        return Combo("flush", cards, 5, (1.0, float(card_suit(cards[0])), *map(float, descending_ranks)))
    if straight_order is not None:
        top_rank = max(rank_values)
        top_suit = max(card_suit(card) for card in cards if card_rank(card) == top_rank)
        return Combo("straight", cards, 5, (0.0, float(straight_order), float(top_suit)))
    return None


def compare_combos(a: Combo, b: Combo) -> int:
    if a.size != b.size:
        return -1
    if a.size == 5:
        diff = FIVE_KIND_ORDER[a.kind] - FIVE_KIND_ORDER[b.kind]
        if diff:
            return diff
    elif a.kind != b.kind:
        return -1
    for left, right in itertools.zip_longest(a.tiebreak, b.tiebreak, fillvalue=0):
        if left != right:
            return 1 if left > right else -1
    return 0


def enumerate_combos(hand: list[int]) -> list[Combo]:
    cards = sorted(hand)
    combos: list[Combo] = []
    for card in cards:
        combos.append(classify([card]))  # type: ignore[arg-type]
    rank_groups: dict[int, list[int]] = {}
    for card in cards:
        rank_groups.setdefault(card_rank(card), []).append(card)
    for group in rank_groups.values():
        for size in (2, 3):
            if len(group) >= size:
                for selected in itertools.combinations(group, size):
                    combo = classify(selected)
                    if combo:
                        combos.append(combo)
    for selected in itertools.combinations(cards, 5):
        combo = classify(selected)
        if combo:
            combos.append(combo)
    seen: dict[tuple[int, ...], Combo] = {combo.cards: combo for combo in combos if combo}
    return sorted(seen.values(), key=lambda combo: (combo.size, KIND_TO_INDEX[combo.kind], combo.tiebreak))


def create_game(rng: random.Random) -> Game:
    deck = list(range(52))
    rng.shuffle(deck)
    if 0 not in deck[:26]:
        swap_index = rng.randrange(26)
        lowest_index = deck.index(0)
        deck[swap_index], deck[lowest_index] = deck[lowest_index], deck[swap_index]
    hands = [sorted(deck[:13]), sorted(deck[13:26])]
    current_player = 0 if 0 in hands[0] else 1
    return Game(hands=hands, current_player=current_player)


def active_players(game: Game) -> list[int]:
    return [player for player, hand in enumerate(game.hands) if player not in game.finished and hand]


def is_terminal(game: Game) -> bool:
    return len(active_players(game)) <= 1


def placements(game: Game) -> list[int]:
    order = list(game.finished)
    for player in active_players(game):
        if player not in order:
            order.append(player)
    return order


def reward_for(game: Game, player: int) -> float:
    order = placements(game)
    return 1.0 if order and order[0] == player else -1.0


def next_player(game: Game, after: int) -> int:
    for offset in range(1, 3):
        player = (after + offset) % 2
        if player not in game.finished and game.hands[player]:
            return player
    return -1


def legal_moves(game: Game) -> list[Combo | None]:
    if is_terminal(game) or game.current_player < 0:
        return []
    combos = enumerate_combos(game.hands[game.current_player])
    if game.turns == 0:
        combos = [combo for combo in combos if 0 in combo.cards]
    if game.active_combo:
        combos = [combo for combo in combos if compare_combos(combo, game.active_combo) > 0]
    moves: list[Combo | None] = list(combos)
    if game.active_combo and game.turns > 0:
        moves.append(None)
    return moves


def apply_move(game: Game, move: Combo | None) -> None:
    player = game.current_player
    if move is None:
        game.passes_since_play += 1
        if game.active_combo and game.last_player is not None and game.passes_since_play >= 1:
            game.current_player = game.last_player
            game.active_combo = None
            game.last_player = None
            game.passes_since_play = 0
        else:
            game.current_player = next_player(game, player)
    else:
        remove = set(move.cards)
        game.hands[player] = [card for card in game.hands[player] if card not in remove]
        game.played.extend(move.cards)
        game.played_by_player[player].extend(move.cards)
        if not game.hands[player] and player not in game.finished:
            game.finished.append(player)
            game.active_combo = None
            game.last_player = None
            game.passes_since_play = 0
        else:
            game.active_combo = move
            game.last_player = player
            game.passes_since_play = 0
        game.current_player = -1 if is_terminal(game) else next_player(game, player)
    game.turns += 1


def state_features(game: Game, player: int) -> list[float]:
    hand_bits = [0.0] * 52
    for card in game.hands[player]:
        hand_bits[card] = 1.0
    self_played_bits = [0.0] * 52
    opponent_played_bits = [0.0] * 52
    for card in game.played_by_player[player]:
        self_played_bits[card] = 1.0
    for card in game.played_by_player[1 - player]:
        opponent_played_bits[card] = 1.0
    active_bits = [0.0] * 52
    if game.active_combo:
        for card in game.active_combo.cards:
            active_bits[card] = 1.0
    scalars = [
        len(game.hands[player]) / 13.0,
        len(game.hands[1 - player]) / 13.0,
        (game.active_combo.size if game.active_combo else 0) / 5.0,
        1.0 if game.active_combo is None else 0.0,
        game.passes_since_play / 2.0,
        min(game.turns, 100) / 100.0,
        1.0 if game.active_combo is not None and game.last_player == player else 0.0,
        1.0 if game.active_combo is not None and game.last_player == 1 - player else 0.0,
    ]
    return hand_bits + self_played_bits + opponent_played_bits + active_bits + scalars


def move_features(move: Combo | None) -> list[float]:
    bits = [0.0] * 52
    if move:
        for card in move.cards:
            bits[card] = 1.0
    kind = [0.0] * 8
    if move:
        kind[KIND_TO_INDEX[move.kind]] = 1.0
    return bits + [1.0 if move is None else 0.0, (move.size if move else 0) / 5.0] + kind


def layer_to_json(layer: nn.Linear) -> dict[str, list[float] | list[list[float]]]:
    return {
        "weight": layer.weight.detach().cpu().tolist(),
        "bias": layer.bias.detach().cpu().tolist(),
    }


def load_json_weights(net: PusoyNet, path: Path) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    weights = data.get("weights", {})
    mapping = {
        "state_fc": net.state_fc,
        "move_fc": net.move_fc,
        "policy_fc": net.policy_fc,
        "policy_out": net.policy_out,
        "value_fc": net.value_fc,
        "value_out": net.value_out,
    }
    with torch.no_grad():
        for name, layer in mapping.items():
            if name not in weights:
                return False
            layer.weight.copy_(torch.tensor(weights[name]["weight"], dtype=layer.weight.dtype))
            layer.bias.copy_(torch.tensor(weights[name]["bias"], dtype=layer.bias.dtype))
    return True
