#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from pusoy_gpu import (
    KIND_TO_INDEX,
    MOVE_DIM,
    STATE_DIM,
    Combo,
    Game,
    PusoyNet,
    apply_move,
    create_game,
    is_terminal,
    layer_to_json,
    legal_moves,
    load_json_weights,
    move_features,
    placements,
    reward_for,
    state_features,
)


@dataclass
class AlphaTransition:
    state: list[float]
    moves: list[list[float]]
    target: list[float]
    player: int
    reward: float = 0.0


@dataclass
class MctsEdge:
    move: Combo | None
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: "MctsNode | None" = None

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class MctsNode:
    game: Game
    visits: int = 0
    children: list[MctsEdge] | None = None


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


@torch.no_grad()
def evaluate_node(net: PusoyNet, game: Game, device: torch.device) -> tuple[list[Combo | None], list[float], float]:
    player = game.current_player
    moves = legal_moves(game)
    if not moves or player < 0:
        return [], [], 0.0

    states = torch.tensor([state_features(game, player)] * len(moves), device=device, dtype=torch.float32)
    move_t = torch.tensor([move_features(move) for move in moves], device=device, dtype=torch.float32)
    logits, values = net(states, move_t)
    priors = torch.softmax(logits, dim=0).detach().cpu().tolist()
    value = float(values[0].item())
    return moves, priors, value


def apply_root_noise(children: list[MctsEdge], rng: random.Random, alpha: float, fraction: float) -> None:
    if not children or fraction <= 0:
        return

    noise = [rng.gammavariate(alpha, 1.0) for _ in children]
    total = sum(noise)
    if total <= 0:
        return
    for edge, sample in zip(children, noise, strict=True):
        edge.prior = (1 - fraction) * edge.prior + fraction * (sample / total)


def expand(node: MctsNode, net: PusoyNet, device: torch.device) -> float:
    moves, priors, value = evaluate_node(net, node.game, device)
    node.children = [MctsEdge(move=move, prior=max(1e-5, prior)) for move, prior in zip(moves, priors, strict=True)]
    return value


def select_edge(node: MctsNode, exploration: float) -> MctsEdge:
    if not node.children:
        raise RuntimeError("Cannot select from an unexpanded node")

    parent_visits = max(1, node.visits)
    best_edge = node.children[0]
    best_score = -math.inf
    for edge in node.children:
        u = exploration * edge.prior * math.sqrt(parent_visits) / (1 + edge.visits)
        score = edge.value + u
        if score > best_score:
            best_score = score
            best_edge = edge
    return best_edge


def edge_child(edge: MctsEdge, parent: MctsNode) -> MctsNode:
    if edge.child is None:
        next_game = clone_game(parent.game)
        apply_move(next_game, edge.move)
        edge.child = MctsNode(next_game)
    return edge.child


def simulate(node: MctsNode, net: PusoyNet, device: torch.device, exploration: float) -> float:
    if is_terminal(node.game):
        return 0.0

    player = node.game.current_player
    node.visits += 1

    if node.children is None:
        return expand(node, net, device)

    if not node.children:
        return 0.0

    edge = select_edge(node, exploration)
    child = edge_child(edge, node)
    if is_terminal(child.game):
        value_for_player = reward_for(child.game, player)
    else:
        child_value = simulate(child, net, device, exploration)
        child_player = child.game.current_player
        value_for_player = child_value if child_player == player else -child_value

    edge.visits += 1
    edge.value_sum += value_for_player
    return value_for_player


def sample_index(probs: list[float], rng: random.Random) -> int:
    total = sum(probs)
    if total <= 0:
        return rng.randrange(len(probs))
    threshold = rng.random() * total
    running = 0.0
    for index, prob in enumerate(probs):
        running += prob
        if running >= threshold:
            return index
    return len(probs) - 1


def normalized_counts(children: list[MctsEdge]) -> list[float]:
    counts = [float(edge.visits) for edge in children]
    total = sum(counts)
    if total <= 0:
        priors = [edge.prior for edge in children]
        prior_total = sum(priors)
        return [prior / prior_total for prior in priors] if prior_total > 0 else [1 / len(children)] * len(children)
    return [count / total for count in counts]


def select_from_visits(
    children: list[MctsEdge],
    turn: int,
    temperature: float,
    temperature_turns: int,
    rng: random.Random,
) -> int:
    if len(children) == 1:
        return 0
    if turn >= temperature_turns or temperature <= 0:
        return max(range(len(children)), key=lambda index: (children[index].visits, children[index].value))

    counts = [max(0.0, float(edge.visits)) for edge in children]
    if sum(counts) <= 0:
        probs = [edge.prior for edge in children]
    else:
        exponent = 1.0 / max(temperature, 0.05)
        probs = [count**exponent for count in counts]
    return sample_index(probs, rng)


def mcts_search(
    net: PusoyNet,
    game: Game,
    device: torch.device,
    rng: random.Random,
    simulations: int,
    exploration: float,
    root_dirichlet_alpha: float,
    root_noise_fraction: float,
    temperature: float,
    temperature_turns: int,
) -> tuple[list[Combo | None], list[float], int, float]:
    root = MctsNode(clone_game(game))
    expand(root, net, device)
    if not root.children:
        return [], [], 0, 0.0
    apply_root_noise(root.children, rng, root_dirichlet_alpha, root_noise_fraction)

    for _ in range(max(1, simulations)):
        simulate(root, net, device, exploration)

    target = normalized_counts(root.children)
    action = select_from_visits(root.children, game.turns, temperature, temperature_turns, rng)
    root_value = sum(edge.value * target[index] for index, edge in enumerate(root.children))
    return [edge.move for edge in root.children], target, action, root_value


def play_batch(
    net: PusoyNet,
    batch_games: int,
    device: torch.device,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[list[AlphaTransition], dict[str, float]]:
    transitions: list[AlphaTransition] = []
    wins = 0
    turns = 0
    target_top_sum = 0.0
    value_sum = 0.0
    searches = 0

    net.eval()
    for _ in range(batch_games):
        game = create_game(rng)
        start = len(transitions)
        while not is_terminal(game) and game.turns < args.max_turns:
            player = game.current_player
            moves, target, action, root_value = mcts_search(
                net,
                game,
                device,
                rng,
                args.simulations,
                args.exploration,
                args.root_dirichlet_alpha,
                args.root_noise_fraction,
                args.temperature,
                args.temperature_turns,
            )
            if not moves:
                break
            transitions.append(
                AlphaTransition(
                    state=state_features(game, player),
                    moves=[move_features(move) for move in moves],
                    target=target,
                    player=player,
                )
            )
            target_top_sum += max(target)
            value_sum += root_value
            searches += 1
            apply_move(game, moves[action])

        turns += game.turns
        if placements(game)[0] == 0:
            wins += 1
        for transition in transitions[start:]:
            transition.reward = reward_for(game, transition.player)

    return transitions, {
        "win_rate_p0": wins / max(1, batch_games),
        "average_turns": turns / max(1, batch_games),
        "average_target_top": target_top_sum / max(1, searches),
        "average_root_value": value_sum / max(1, searches),
    }


def train_transitions(
    net: PusoyNet,
    optimizer: torch.optim.Optimizer,
    transitions: list[AlphaTransition],
    device: torch.device,
    minibatch_size: int,
    epochs: int,
    value_weight: float,
    entropy_weight: float,
) -> dict[str, float]:
    if not transitions:
        return {"loss": 0.0, "policyLoss": 0.0, "valueLoss": 0.0, "entropy": 0.0}

    net.train()
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    total_entropy = 0.0
    updates = 0

    for _ in range(epochs):
        random.shuffle(transitions)
        for start in range(0, len(transitions), minibatch_size):
            batch = transitions[start : start + minibatch_size]
            states: list[list[float]] = []
            moves: list[list[float]] = []
            offsets: list[int] = []
            rewards: list[float] = []
            cursor = 0
            for transition in batch:
                offsets.append(cursor)
                rewards.append(transition.reward)
                for move in transition.moves:
                    states.append(transition.state)
                    moves.append(move)
                cursor += len(transition.moves)

            state_t = torch.tensor(states, device=device, dtype=torch.float32)
            move_t = torch.tensor(moves, device=device, dtype=torch.float32)
            reward_t = torch.tensor(rewards, device=device, dtype=torch.float32)
            logits, values_all = net(state_t, move_t)

            policy_losses = []
            entropies = []
            for index, transition in enumerate(batch):
                segment = logits[offsets[index] : offsets[index] + len(transition.moves)]
                target = torch.tensor(transition.target, device=device, dtype=torch.float32)
                target = target / target.sum().clamp_min(1e-6)
                logp = torch.log_softmax(segment, dim=0)
                probs = torch.softmax(segment, dim=0)
                policy_losses.append(-(target * logp).sum())
                entropies.append(-(probs * logp).sum())

            policy_loss = torch.stack(policy_losses).mean()
            entropy = torch.stack(entropies).mean()
            value_t = values_all[torch.tensor(offsets, device=device, dtype=torch.long)]
            value_loss = F.mse_loss(value_t, reward_t)
            loss = policy_loss + value_weight * value_loss - entropy_weight * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            total_entropy += float(entropy.item())
            updates += 1

    return {
        "loss": total_loss / max(1, updates),
        "policyLoss": total_policy / max(1, updates),
        "valueLoss": total_value / max(1, updates),
        "entropy": total_entropy / max(1, updates),
    }


def save_model(
    net: PusoyNet,
    path: Path,
    args: argparse.Namespace,
    history: list[dict[str, float]],
    started_at: float,
    games: int,
    transitions: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "name": "gpu-alphazero-move-scorer",
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": {
            "stateDim": STATE_DIM,
            "moveDim": MOVE_DIM,
            "stateHidden": 128,
            "moveHidden": 64,
            "policyHidden": 64,
            "valueHidden": 64,
            "comboKinds": KIND_TO_INDEX,
        },
        "training": {
            "device": str(args.device),
            "style": "alphazero-mcts-visit-targets",
            "requestedMinutes": args.duration_minutes,
            "elapsedSeconds": time.time() - started_at,
            "games": games,
            "transitions": transitions,
            "batchGames": args.batch_games,
            "minibatchSize": args.minibatch_size,
            "epochs": args.epochs,
            "learningRate": args.lr,
            "mctsSimulations": args.simulations,
            "exploration": args.exploration,
            "temperature": args.temperature,
            "temperatureTurns": args.temperature_turns,
            "rootDirichletAlpha": args.root_dirichlet_alpha,
            "rootNoiseFraction": args.root_noise_fraction,
            "replaySize": args.replay_size,
            "trainSamples": args.train_samples,
            "history": history,
            "note": "AlphaZero-style self-play: MCTS visit counts train the policy head and terminal outcomes train the value head. Browser inference uses exported JSON weights.",
        },
        "inference": {
            "handcraftedPriorWeight": args.handcrafted_prior_weight,
            "handcraftedValueWeight": args.handcrafted_value_weight,
            "neuralPolicyTemperature": args.neural_policy_temperature,
        },
        "weights": {
            "state_fc": layer_to_json(net.state_fc),
            "move_fc": layer_to_json(net.move_fc),
            "policy_fc": layer_to_json(net.policy_fc),
            "policy_out": layer_to_json(net.policy_out),
            "value_fc": layer_to_json(net.value_fc),
            "value_out": layer_to_json(net.value_out),
        },
    }
    path.write_text(json.dumps(data, separators=(",", ":")))


def choose_training_samples(replay: list[AlphaTransition], train_samples: int, rng: random.Random) -> list[AlphaTransition]:
    if train_samples <= 0 or train_samples >= len(replay):
        return list(replay)
    return rng.sample(replay, train_samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="public/models/neural-policy.json")
    parser.add_argument("--init-model", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--duration-minutes", type=float, default=0)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--batch-games", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--exploration", type=float, default=1.35)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-turns", type=int, default=10)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-noise-fraction", type=float, default=0.18)
    parser.add_argument("--max-turns", type=int, default=160)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--handcrafted-prior-weight", type=float, default=0.0)
    parser.add_argument("--handcrafted-value-weight", type=float, default=0.0)
    parser.add_argument("--neural-policy-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-interval", type=int, default=1)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    device = torch.device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.set_float32_matmul_precision("high")

    net = PusoyNet().to(device)
    output = Path(args.output)
    init_model = Path(args.init_model) if args.init_model else output
    loaded = False
    if args.resume or args.init_model:
        loaded = load_json_weights(net, init_model)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    started_at = time.time()
    deadline = started_at + args.duration_minutes * 60 if args.duration_minutes > 0 else None
    history: list[dict[str, float]] = []
    replay: list[AlphaTransition] = []
    total_games = 0
    total_transitions = 0
    iteration = 0

    print(
        f"gpu alphazero training -> {output} | device={device} | batch={args.batch_games} games | "
        f"sims={args.simulations} | duration={args.duration_minutes or 'fixed'} | loaded={loaded}",
        flush=True,
    )

    while args.iterations <= 0 or iteration < args.iterations:
        if deadline and time.time() >= deadline and iteration > 0:
            break
        iteration += 1

        transitions, stats = play_batch(net, args.batch_games, device, rng, args)
        replay.extend(transitions)
        if len(replay) > args.replay_size:
            del replay[: len(replay) - args.replay_size]
        train_batch = choose_training_samples(replay, args.train_samples, rng)
        train_stats = train_transitions(
            net,
            optimizer,
            train_batch,
            device,
            args.minibatch_size,
            args.epochs,
            args.value_weight,
            args.entropy_weight,
        )

        total_games += args.batch_games
        total_transitions += len(transitions)
        row = {
            "iteration": iteration,
            "elapsedSeconds": time.time() - started_at,
            "games": total_games,
            "transitions": total_transitions,
            "replayTransitions": len(replay),
            "loss": train_stats["loss"],
            "policyLoss": train_stats["policyLoss"],
            "valueLoss": train_stats["valueLoss"],
            "entropy": train_stats["entropy"],
            "winRateP0": stats["win_rate_p0"],
            "averageTurns": stats["average_turns"],
            "averageTargetTop": stats["average_target_top"],
            "averageRootValue": stats["average_root_value"],
        }
        history.append(row)
        print(
            f"iter {iteration} | games={total_games} | decisions={total_transitions} | "
            f"loss={row['loss']:.4f} | policy={row['policyLoss']:.4f} | value={row['valueLoss']:.4f} | "
            f"p0_win={row['winRateP0']:.3f} | turns={row['averageTurns']:.1f} | target_top={row['averageTargetTop']:.3f}",
            flush=True,
        )
        if args.save_interval > 0 and iteration % args.save_interval == 0:
            save_model(net, output, args, history, started_at, total_games, total_transitions)

    save_model(net, output, args, history, started_at, total_games, total_transitions)
    print(f"finished gpu alphazero training in {time.time() - started_at:.1f}s, games={total_games}", flush=True)


if __name__ == "__main__":
    main()
