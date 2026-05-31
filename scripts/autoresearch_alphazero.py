#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class InferenceSetting:
    name: str
    handcrafted_prior_weight: float
    handcrafted_value_weight: float
    neural_policy_temperature: float
    rollout_value_weight: float
    opponent_aware_prior_weight: float = 0.0
    endgame_solver_cards: int = 0


@dataclass
class EvalResult:
    checkpoint: Path
    setting: InferenceSetting
    games: int
    first_place_rate: float
    average_reward: float
    average_turns: float
    log_path: Path


DEFAULT_SETTINGS = [
    InferenceSetting("rollout-p60-t05-r25", 0.60, 0.0, 0.50, 0.25),
    InferenceSetting("rollout-p45-t05-r25", 0.45, 0.0, 0.50, 0.25),
    InferenceSetting("rollout-p75-t05-r25", 0.75, 0.0, 0.50, 0.25),
    InferenceSetting("rollout-p60-t08-r25", 0.60, 0.0, 0.80, 0.25),
    InferenceSetting("rollout-p60-t05-r10", 0.60, 0.0, 0.50, 0.10),
    InferenceSetting("rollout-p60-t05-r40", 0.60, 0.0, 0.50, 0.40),
]


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code != 0:
        raise SystemExit(f"Command failed with exit code {code}: {' '.join(command)}")


def capture_command(command: list[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    output = process.stdout + process.stderr
    log_path.write_text(output)
    print(output, end="", flush=True)
    if process.returncode != 0:
        raise SystemExit(f"Command failed with exit code {process.returncode}: {' '.join(command)}")
    return output


def parse_metric(output: str, name: str) -> float:
    match = re.search(rf"^{re.escape(name)}:\s+([-0-9.]+)", output, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not parse {name} from evaluator output")
    return float(match.group(1))


def evaluate(
    checkpoint: Path,
    setting: InferenceSetting,
    games: int,
    simulations: int,
    seed: int,
    log_path: Path,
) -> EvalResult:
    command = [
        "npm",
        "run",
        "eval:alphazero",
        "--",
        "--model",
        str(checkpoint),
        "--games",
        str(games),
        "--simulations",
        str(simulations),
        "--seed",
        str(seed),
        "--handcrafted-prior-weight",
        str(setting.handcrafted_prior_weight),
        "--handcrafted-value-weight",
        str(setting.handcrafted_value_weight),
        "--neural-policy-temperature",
        str(setting.neural_policy_temperature),
        "--rollout-value-weight",
        str(setting.rollout_value_weight),
        "--opponent-aware-prior-weight",
        str(setting.opponent_aware_prior_weight),
        "--endgame-solver-cards",
        str(setting.endgame_solver_cards),
    ]
    output = capture_command(command, log_path)
    return EvalResult(
        checkpoint=checkpoint,
        setting=setting,
        games=games,
        first_place_rate=parse_metric(output, "first_place_rate"),
        average_reward=parse_metric(output, "average_reward"),
        average_turns=parse_metric(output, "average_turns"),
        log_path=log_path,
    )


def train_chunk(args: argparse.Namespace, init_model: Path, output: Path, chunk_index: int, phase: dict[str, object]) -> None:
    command = [
        "npm",
        "run",
        "train:alphazero",
        "--",
        "--output",
        str(output),
        "--init-model",
        str(init_model),
        "--duration-minutes",
        str(args.chunk_minutes),
        "--batch-games",
        str(phase["batch_games"]),
        "--simulations",
        str(phase["simulations"]),
        "--train-samples",
        str(args.train_samples),
        "--minibatch-size",
        str(args.minibatch_size),
        "--replay-size",
        str(args.replay_size),
        "--lr",
        str(phase["lr"]),
        "--opponent-mode",
        "handcrafted",
        "--handcrafted-search-simulations",
        str(phase["handcrafted_search_simulations"]),
        "--handcrafted-temperature",
        "0",
        "--save-interval",
        str(args.save_interval),
        "--seed",
        str(args.seed + chunk_index),
    ]
    run_command(command, args.run_dir / "logs" / f"train-{chunk_index:03d}-{phase['name']}.log")


def write_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def phase_for_index(index: int) -> dict[str, object]:
    phases = [
        {
            "name": "search12-lr5e6",
            "lr": 0.000005,
            "simulations": 24,
            "batch_games": 64,
            "handcrafted_search_simulations": 12,
        },
        {
            "name": "search24-lr3e6",
            "lr": 0.000003,
            "simulations": 24,
            "batch_games": 48,
            "handcrafted_search_simulations": 24,
        },
        {
            "name": "search12-lr2e6",
            "lr": 0.000002,
            "simulations": 32,
            "batch_games": 48,
            "handcrafted_search_simulations": 12,
        },
    ]
    return phases[index % len(phases)]


def promote_model(source: Path, target: Path, result: EvalResult, confirm: EvalResult | None) -> None:
    model = json.loads(source.read_text())
    model["name"] = "gpu-alphazero-rollout-hybrid"
    model["inference"] = {
        "handcraftedPriorWeight": result.setting.handcrafted_prior_weight,
        "handcraftedValueWeight": result.setting.handcrafted_value_weight,
        "neuralPolicyTemperature": result.setting.neural_policy_temperature,
        "opponentAwarePriorWeight": result.setting.opponent_aware_prior_weight,
        "endgameSolverCards": result.setting.endgame_solver_cards,
        "rolloutValueWeight": result.setting.rollout_value_weight,
    }
    model["training"] = {
        **model.get("training", {}),
        "selectedAt": datetime.now(timezone.utc).isoformat(),
        "selectedBy": "12-hour checkpointed autoresearch",
        "arena": {
            "opponent": "handcrafted",
            "games": confirm.games if confirm else result.games,
            "simulationsPerMove": 48,
            "seed": 20260531,
            "firstPlaceRate": confirm.first_place_rate if confirm else result.first_place_rate,
            "averageReward": confirm.average_reward if confirm else result.average_reward,
            "averageTurns": confirm.average_turns if confirm else result.average_turns,
        },
    }
    target.write_text(json.dumps(model, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("research-runs/12h-autoresearch") / stamp())
    parser.add_argument("--initial-model", type=Path, default=Path("public/models/neural-policy.json"))
    parser.add_argument("--promote-to", type=Path, default=Path("public/models/neural-policy.json"))
    parser.add_argument("--total-hours", type=float, default=12)
    parser.add_argument("--chunk-minutes", type=float, default=45)
    parser.add_argument("--screen-games", type=int, default=200)
    parser.add_argument("--confirm-games", type=int, default=1000)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--baseline-rate", type=float, default=0.601)
    parser.add_argument("--seed", type=int, default=2026060201)
    parser.add_argument("--eval-seed", type=int, default=20260531)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--confirm-margin", type=float, default=0.01)
    parser.add_argument("--reserve-minutes", type=float, default=20)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.run_dir / "events.jsonl",
        {
            "event": "start",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "args": {key: str(value) for key, value in vars(args).items()},
        },
    )

    deadline = time.time() + args.total_hours * 3600
    current_model = args.initial_model
    best_screen: EvalResult | None = None
    best_confirm: EvalResult | None = None
    best_rate = args.baseline_rate
    chunk_index = 0

    while time.time() < deadline:
        remaining_minutes = (deadline - time.time()) / 60
        if remaining_minutes < args.chunk_minutes + args.reserve_minutes:
            break

        chunk_index += 1
        phase = phase_for_index(chunk_index - 1)
        checkpoint = args.run_dir / "checkpoints" / f"checkpoint-{chunk_index:03d}-{phase['name']}.json"
        print(f"\n=== chunk {chunk_index} phase={phase['name']} init={current_model} output={checkpoint} ===", flush=True)
        train_chunk(args, current_model, checkpoint, chunk_index, phase)

        screen_results: list[EvalResult] = []
        for setting in DEFAULT_SETTINGS:
            log_path = args.run_dir / "logs" / f"eval-screen-{chunk_index:03d}-{phase['name']}-{setting.name}.log"
            result = evaluate(checkpoint, setting, args.screen_games, args.simulations, args.eval_seed, log_path)
            screen_results.append(result)
            write_jsonl(
                args.run_dir / "events.jsonl",
                {
                    "event": "screen",
                    "chunk": chunk_index,
                    "phase": phase,
                    "checkpoint": str(checkpoint),
                    "setting": result.setting.__dict__,
                    "games": result.games,
                    "firstPlaceRate": result.first_place_rate,
                    "averageReward": result.average_reward,
                    "averageTurns": result.average_turns,
                    "log": str(result.log_path),
                },
            )

        winner = max(screen_results, key=lambda item: (item.first_place_rate, item.average_reward))
        print(
            f"chunk {chunk_index} winner: {winner.setting.name} "
            f"{winner.first_place_rate:.4f} reward={winner.average_reward:.4f}",
            flush=True,
        )
        if best_screen is None or winner.first_place_rate > best_screen.first_place_rate:
            best_screen = winner

        if winner.first_place_rate >= best_rate + args.confirm_margin and (deadline - time.time()) / 60 > args.reserve_minutes:
            confirm_log = args.run_dir / "logs" / f"eval-confirm-{chunk_index:03d}-{phase['name']}-{winner.setting.name}.log"
            confirm = evaluate(checkpoint, winner.setting, args.confirm_games, args.simulations, args.eval_seed, confirm_log)
            write_jsonl(
                args.run_dir / "events.jsonl",
                {
                    "event": "confirm",
                    "chunk": chunk_index,
                    "phase": phase,
                    "checkpoint": str(checkpoint),
                    "setting": confirm.setting.__dict__,
                    "games": confirm.games,
                    "firstPlaceRate": confirm.first_place_rate,
                    "averageReward": confirm.average_reward,
                    "averageTurns": confirm.average_turns,
                    "log": str(confirm.log_path),
                },
            )
            if confirm.first_place_rate > best_rate:
                best_rate = confirm.first_place_rate
                best_confirm = confirm
                current_model = checkpoint
                promote_model(checkpoint, args.promote_to, winner, confirm)
                shutil.copyfile(checkpoint, args.run_dir / "best-confirmed.json")
                print(f"new confirmed best: {best_rate:.4f} from {checkpoint}", flush=True)
            else:
                current_model = checkpoint
        else:
            current_model = checkpoint

    if best_confirm is not None:
        promote_model(best_confirm.checkpoint, args.promote_to, best_confirm, best_confirm)
    summary = {
        "event": "finish",
        "finishedAt": datetime.now(timezone.utc).isoformat(),
        "bestScreen": None
        if best_screen is None
        else {
            "checkpoint": str(best_screen.checkpoint),
            "setting": best_screen.setting.__dict__,
            "firstPlaceRate": best_screen.first_place_rate,
            "averageReward": best_screen.average_reward,
            "games": best_screen.games,
        },
        "bestConfirm": None
        if best_confirm is None
        else {
            "checkpoint": str(best_confirm.checkpoint),
            "setting": best_confirm.setting.__dict__,
            "firstPlaceRate": best_confirm.first_place_rate,
            "averageReward": best_confirm.average_reward,
            "games": best_confirm.games,
        },
    }
    write_jsonl(args.run_dir / "events.jsonl", summary)
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
