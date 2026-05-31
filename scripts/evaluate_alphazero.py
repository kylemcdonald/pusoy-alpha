#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from pusoy_gpu import (
    Combo,
    Game,
    PusoyNet,
    apply_move,
    card_power,
    card_rank,
    classify,
    create_game,
    is_terminal,
    legal_moves,
    load_json_weights,
    move_features,
    placements,
    reward_for,
    state_features,
)


class PolicyValue(Protocol):
    name: str

    def evaluate(
        self,
        game: Game,
        perspective_player: int,
        moves: list[Combo | None],
        device: torch.device,
    ) -> tuple[list[float], float]:
        ...


@dataclass
class SearchNode:
    game: Game
    move: Combo | None = None
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: list["SearchNode"] | None = None


@dataclass
class SearchResult:
    move: Combo | None
    visits: dict[str, int]
    value: float


def clone_game(game: Game) -> Game:
    return Game(
        hands=[list(hand) for hand in game.hands],
        current_player=game.current_player,
        active_combo=game.active_combo,
        last_player=game.last_player,
        passes_since_play=game.passes_since_play,
        finished=list(game.finished or []),
        played=list(game.played or []),
        played_by_player=[list(cards) for cards in (game.played_by_player or [[], []])],
        turns=game.turns,
    )


def move_key(move: Combo | None) -> str:
    if move is None:
        return "pass"
    return "-".join(str(card) for card in move.cards)


def node_value(node: SearchNode) -> float:
    return node.value_sum / node.visits if node.visits else 0.0


def normalize(values: list[float]) -> list[float]:
    total = sum(max(0.0001, value) for value in values)
    if total <= 0:
        return [1 / len(values)] * len(values)
    return [max(0.0001, value) / total for value in values]


def combo_strength(combo: Combo) -> float:
    raw = sum(value / (index + 1) for index, value in enumerate(combo.tiebreak))
    return raw / 70


def handcrafted_move_score(game: Game, move: Combo | None) -> float:
    if move is None:
        return 0.08

    actor = game.current_player
    combo = classify(move.cards)
    if combo is None:
        return 0.01

    remaining_after_move = len(game.hands[actor]) - combo.size
    score = 0.25 + combo.size * 0.18

    if remaining_after_move == 0:
        score += 5.0
    elif remaining_after_move <= 2:
        score += 0.8

    if game.active_combo:
        score += max(0.0, 0.45 - combo_strength(combo))
    else:
        score += max(0.0, 0.7 - combo_strength(combo))

    has_rank_two = any(card_rank(card) == 12 for card in combo.cards)
    if has_rank_two and remaining_after_move > 0:
        score -= 0.2

    next_player = 1 - actor
    if len(game.hands[next_player]) == 1 and combo.size == 1:
        score += combo_strength(combo) * 0.6

    return max(0.01, score)


def handcrafted_value(game: Game, perspective_player: int) -> float:
    if is_terminal(game):
        return reward_for(game, perspective_player)

    my_count = len(game.hands[perspective_player])
    opponent_count = len(game.hands[1 - perspective_player])
    count_value = (opponent_count - my_count) / 13
    if my_count > 0:
        hand_strength = sum(card_power(card) for card in game.hands[perspective_player]) / (my_count * 51)
    else:
        hand_strength = 0.0
    control_value = 0.12 if game.current_player == perspective_player and game.active_combo is None else 0.0
    return max(-1.0, min(1.0, count_value * 1.25 + hand_strength * 0.22 + control_value))


class HandcraftedPolicyValue:
    name = "handcrafted"

    def evaluate(
        self,
        game: Game,
        perspective_player: int,
        moves: list[Combo | None],
        device: torch.device,
    ) -> tuple[list[float], float]:
        return normalize([handcrafted_move_score(game, move) for move in moves]), handcrafted_value(game, perspective_player)


class NeuralPolicyValue:
    name = "neural"

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        handcrafted_prior_weight: float | None = None,
        handcrafted_value_weight: float | None = None,
        neural_policy_temperature: float | None = None,
        opponent_aware_prior_weight: float | None = None,
        endgame_solver_cards: int | None = None,
    ) -> None:
        self.model_path = model_path
        self.net = PusoyNet().to(device)
        if not load_json_weights(self.net, model_path):
            raise SystemExit(f"Could not load neural weights from {model_path}")
        self.net.eval()
        data = json.loads(model_path.read_text())
        inference = data.get("inference", {})
        self.handcrafted_prior_weight = max(
            0.0,
            min(1.0, handcrafted_prior_weight if handcrafted_prior_weight is not None else inference.get("handcraftedPriorWeight", 0.0)),
        )
        self.handcrafted_value_weight = max(
            0.0,
            min(1.0, handcrafted_value_weight if handcrafted_value_weight is not None else inference.get("handcraftedValueWeight", 0.0)),
        )
        self.neural_policy_temperature = max(
            0.05,
            neural_policy_temperature if neural_policy_temperature is not None else inference.get("neuralPolicyTemperature", 1.0),
        )
        self.opponent_aware_prior_weight = max(
            0.0,
            min(1.0, opponent_aware_prior_weight if opponent_aware_prior_weight is not None else inference.get("opponentAwarePriorWeight", 0.0)),
        )
        self.endgame_solver_cards = max(
            0,
            int(endgame_solver_cards if endgame_solver_cards is not None else inference.get("endgameSolverCards", 0)),
        )
        self.endgame_cache: dict[tuple[int, tuple], float] = {}
        self.handcrafted = HandcraftedPolicyValue()

    @torch.no_grad()
    def evaluate(
        self,
        game: Game,
        perspective_player: int,
        moves: list[Combo | None],
        device: torch.device,
    ) -> tuple[list[float], float]:
        if is_terminal(game):
            return [], reward_for(game, perspective_player)
        if not moves:
            return [], 0.0

        states = torch.tensor(
            [state_features(game, perspective_player)] * len(moves),
            device=device,
            dtype=torch.float32,
        )
        move_t = torch.tensor([move_features(move) for move in moves], device=device, dtype=torch.float32)
        logits, values = self.net(states, move_t)
        logits = logits / self.neural_policy_temperature
        priors = torch.softmax(logits, dim=0).detach().cpu().tolist()
        value = float(values[0].item())

        if self.endgame_solver_cards > 0 and remaining_cards(game) <= self.endgame_solver_cards:
            value = self.solve_endgame(game, perspective_player)
            priors = exact_policy_priors(game, perspective_player, moves, self.solve_endgame)

        if self.handcrafted_prior_weight > 0 or self.handcrafted_value_weight > 0:
            handcrafted_priors, handcrafted_value_result = self.handcrafted.evaluate(game, perspective_player, moves, device)
            priors = [
                (1 - self.handcrafted_prior_weight) * neural_prior
                + self.handcrafted_prior_weight * handcrafted_prior
                for neural_prior, handcrafted_prior in zip(priors, handcrafted_priors, strict=True)
            ]
            value = (1 - self.handcrafted_value_weight) * value + self.handcrafted_value_weight * handcrafted_value_result

        if self.opponent_aware_prior_weight > 0:
            aware_priors = opponent_aware_priors(game, moves)
            priors = [
                (1 - self.opponent_aware_prior_weight) * prior + self.opponent_aware_prior_weight * aware_prior
                for prior, aware_prior in zip(priors, aware_priors, strict=True)
            ]

        return priors, max(-1.0, min(1.0, value))

    def solve_endgame(self, game: Game, perspective_player: int) -> float:
        key = (perspective_player, game_key(game))
        if key in self.endgame_cache:
            return self.endgame_cache[key]
        if is_terminal(game):
            value = reward_for(game, perspective_player)
        else:
            moves = legal_moves(game)
            if not moves:
                value = 0.0
            else:
                child_values = []
                for move in moves:
                    child = clone_game(game)
                    apply_move(child, move)
                    child_values.append(self.solve_endgame(child, perspective_player))
                value = max(child_values) if game.current_player == perspective_player else min(child_values)
        self.endgame_cache[key] = value
        return value


def remaining_cards(game: Game) -> int:
    return sum(len(hand) for player, hand in enumerate(game.hands) if player not in (game.finished or []))


def game_key(game: Game) -> tuple:
    active = tuple(game.active_combo.cards) if game.active_combo else ()
    return (
        tuple(game.hands[0]),
        tuple(game.hands[1]),
        game.current_player,
        active,
        game.last_player,
        game.passes_since_play,
        tuple(game.finished or []),
        0 if game.turns == 0 else 1,
    )


def exact_policy_priors(
    game: Game,
    perspective_player: int,
    moves: list[Combo | None],
    solver: callable,
) -> list[float]:
    if not moves:
        return []
    values = []
    for move in moves:
        child = clone_game(game)
        apply_move(child, move)
        values.append(solver(child, perspective_player))
    scale = 4.0 if game.current_player == perspective_player else -4.0
    logits = [value * scale for value in values]
    top = max(logits)
    exp = [math.exp(logit - top) for logit in logits]
    total = sum(exp)
    return [value / total for value in exp] if total > 0 else [1 / len(moves)] * len(moves)


def opponent_can_answer(game: Game, move: Combo) -> bool:
    next_game = clone_game(game)
    apply_move(next_game, move)
    if is_terminal(next_game) or next_game.current_player < 0 or next_game.active_combo is None:
        return False
    return any(reply is not None for reply in legal_moves(next_game))


def opponent_aware_priors(game: Game, moves: list[Combo | None]) -> list[float]:
    actor = game.current_player
    scores: list[float] = []
    for move in moves:
        if move is None:
            scores.append(0.05)
            continue

        remaining = len(game.hands[actor]) - move.size
        score = 0.15 + move.size * 0.25
        if remaining == 0:
            score += 8.0
        elif remaining <= 2:
            score += 1.2

        can_answer = opponent_can_answer(game, move)
        if can_answer:
            score -= 0.35
        else:
            score += 1.5 + move.size * 0.2

        if any(card_rank(card) == 12 for card in move.cards) and remaining > 0:
            score -= 0.25
        if len(game.hands[1 - actor]) == 1 and move.size == 1:
            score += 1.5 if not can_answer else -0.8

        scores.append(max(0.01, score))

    total = sum(scores)
    return [score / total for score in scores] if total > 0 else [1 / len(moves)] * len(moves)


def expand(
    node: SearchNode,
    model: PolicyValue,
    root_player: int,
    device: torch.device,
    policy_perspective: str,
) -> float:
    moves = legal_moves(node.game)
    if not moves:
        node.children = []
        return 0.0

    policy_player = node.game.current_player if policy_perspective == "actor" else root_player
    priors, policy_value = model.evaluate(node.game, policy_player, moves, device)
    if policy_player == root_player:
        value = policy_value
    else:
        value = model.evaluate(node.game, root_player, moves, device)[1]
    node.children = [
        SearchNode(game=apply_to_clone(node.game, move), move=move, prior=max(0.0001, prior))
        for move, prior in zip(moves, priors, strict=True)
    ]
    return value


def apply_to_clone(game: Game, move: Combo | None) -> Game:
    next_game = clone_game(game)
    apply_move(next_game, move)
    return next_game


def select_child(node: SearchNode, root_player: int, exploration: float) -> SearchNode:
    if not node.children:
        raise RuntimeError("Cannot select from an unexpanded node")

    parent_visits = max(1, node.visits)
    actor = node.game.current_player
    best_child = node.children[0]
    best_score = -math.inf

    for child in node.children:
        q = node_value(child)
        if actor != root_player:
            q = -q
        u = exploration * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
        score = q + u
        if score > best_score:
            best_score = score
            best_child = child

    return best_child


def visit(
    node: SearchNode,
    model: PolicyValue,
    root_player: int,
    device: torch.device,
    depth: int,
    max_depth: int,
    exploration: float,
    policy_perspective: str,
) -> float:
    node.visits += 1

    if is_terminal(node.game):
        value = reward_for(node.game, root_player)
        node.value_sum += value
        return value

    moves = legal_moves(node.game)
    if depth >= max_depth or not moves:
        value = model.evaluate(node.game, root_player, moves, device)[1]
        node.value_sum += value
        return value

    if node.children is None:
        value = expand(node, model, root_player, device, policy_perspective)
        node.value_sum += value
        return value

    child = select_child(node, root_player, exploration)
    value = visit(child, model, root_player, device, depth + 1, max_depth, exploration, policy_perspective)
    node.value_sum += value
    return value


def search_move(
    game: Game,
    model: PolicyValue,
    root_player: int,
    device: torch.device,
    simulations: int,
    max_depth: int,
    exploration: float,
    policy_perspective: str,
) -> SearchResult:
    moves = legal_moves(game)
    if not moves:
        raise RuntimeError("Cannot search from terminal state")
    if len(moves) == 1:
        return SearchResult(move=moves[0], visits={move_key(moves[0]): 1}, value=0.0)

    root = SearchNode(game=clone_game(game))
    for _ in range(max(1, simulations)):
        visit(root, model, root_player, device, 0, max_depth, exploration, policy_perspective)

    children = root.children or []
    best = max(children, key=lambda child: (child.visits, node_value(child)))
    return SearchResult(
        move=best.move,
        visits={move_key(child.move): child.visits for child in children},
        value=node_value(root),
    )


def determinize_for_player(game: Game, observer: int, rng: random.Random) -> Game:
    next_game = clone_game(game)
    known = set(next_game.hands[observer])
    known.update(next_game.played)
    unknown = [card for card in range(52) if card not in known]
    rng.shuffle(unknown)

    cursor = 0
    for player in range(2):
        if player == observer:
            continue
        count = len(next_game.hands[player])
        next_game.hands[player] = sorted(unknown[cursor : cursor + count])
        cursor += count

    return next_game


def observer_search(
    game: Game,
    observer: int,
    model: PolicyValue,
    device: torch.device,
    rng: random.Random,
    simulations: int,
    determinizations: int,
    max_depth: int,
    exploration: float,
    reveal_opponents: bool,
    policy_perspective: str,
) -> SearchResult:
    root_moves = legal_moves(game)
    if len(root_moves) <= 1 or reveal_opponents:
        return search_move(
            game,
            model,
            observer,
            device,
            simulations,
            max_depth,
            exploration,
            policy_perspective,
        )

    visits = {move_key(move): 0 for move in root_moves}
    value_sum = 0.0
    per_determination = max(1, simulations // max(1, determinizations))
    for _ in range(max(1, determinizations)):
        determinized = determinize_for_player(game, observer, rng)
        result = search_move(
            determinized,
            model,
            observer,
            device,
            per_determination,
            max_depth,
            exploration,
            policy_perspective,
        )
        value_sum += result.value
        for key, count in result.visits.items():
            if key in visits:
                visits[key] += count

    best_move = max(root_moves, key=lambda move: visits.get(move_key(move), 0))
    return SearchResult(move=best_move, visits=visits, value=value_sum / max(1, determinizations))


def play_evaluation_game(
    seed: int,
    candidate_seat: int,
    candidate: PolicyValue,
    opponent: PolicyValue,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[Game, int]:
    game_rng = random.Random(seed)
    search_rng = random.Random(seed + 17)
    game = create_game(game_rng)

    while not is_terminal(game) and game.turns < args.max_turns:
        player = game.current_player
        model = candidate if player == candidate_seat else opponent
        result = observer_search(
            game,
            player,
            model,
            device,
            search_rng,
            args.simulations,
            args.determinizations,
            args.max_depth,
            args.exploration,
            args.reveal_opponents,
            args.policy_perspective,
        )
        apply_move(game, result.move)

    return game, game.turns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="public/models/neural-policy.json")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--determinizations", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--max-turns", type=int, default=280)
    parser.add_argument("--exploration", type=float, default=1.35)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reveal-opponents", action="store_true")
    parser.add_argument("--policy-perspective", choices=["actor", "root"], default="actor")
    parser.add_argument("--handcrafted-prior-weight", type=float, default=None)
    parser.add_argument("--handcrafted-value-weight", type=float, default=None)
    parser.add_argument("--neural-policy-temperature", type=float, default=None)
    parser.add_argument("--opponent-aware-prior-weight", type=float, default=None)
    parser.add_argument("--endgame-solver-cards", type=int, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.set_float32_matmul_precision("high")

    candidate = NeuralPolicyValue(
        Path(args.model),
        device,
        args.handcrafted_prior_weight,
        args.handcrafted_value_weight,
        args.neural_policy_temperature,
        args.opponent_aware_prior_weight,
        args.endgame_solver_cards,
    )
    opponent = HandcraftedPolicyValue()
    first_places = 0
    total_reward = 0.0
    total_turns = 0

    for game_index in range(args.games):
        candidate_seat = game_index % 2
        game, turns = play_evaluation_game(
            args.seed + game_index * 1009,
            candidate_seat,
            candidate,
            opponent,
            device,
            args,
        )
        order = placements(game)
        if order and order[0] == candidate_seat:
            first_places += 1
        total_reward += reward_for(game, candidate_seat)
        total_turns += turns

    print(f"games: {args.games}")
    print(f"candidate: {candidate.name}")
    print(f"opponent: {opponent.name}")
    print(f"first_place_rate: {first_places / max(1, args.games):.4f}")
    print(f"average_reward: {total_reward / max(1, args.games):.4f}")
    print(f"average_turns: {total_turns / max(1, args.games):.2f}")
    print(f"simulations: {args.simulations}")
    print(f"determinizations: {args.determinizations}")
    print(f"policy_perspective: {args.policy_perspective}")
    print(f"reveal_opponents: {args.reveal_opponents}")
    print(f"handcrafted_prior_weight: {candidate.handcrafted_prior_weight:.4f}")
    print(f"handcrafted_value_weight: {candidate.handcrafted_value_weight:.4f}")
    print(f"neural_policy_temperature: {candidate.neural_policy_temperature:.4f}")
    print(f"opponent_aware_prior_weight: {candidate.opponent_aware_prior_weight:.4f}")
    print(f"endgame_solver_cards: {candidate.endgame_solver_cards}")


if __name__ == "__main__":
    main()
