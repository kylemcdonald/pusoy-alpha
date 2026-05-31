import { Move } from "../core/combinations";
import { GameState, PLAYER_COUNT, applyMove, createGame, isTerminal, placements, rewardForPlayer } from "../core/game";
import { RandomSource, createRng } from "../core/random";
import { HandcraftedModel, PolicyValueModel } from "./model";
import { searchMoveForObserver } from "./mcts";

export interface SeatModel {
  seat: number;
  model: PolicyValueModel;
}

export interface EvaluationOptions {
  games?: number;
  simulationsPerMove?: number;
  determinizations?: number;
  seed?: string | number;
  maxTurns?: number;
}

export interface EvaluationResult {
  games: number;
  averageReward: number;
  firstPlaceRate: number;
  averageTurns: number;
}

function selectModel(player: number, seatModels: SeatModel[], fallback: PolicyValueModel): PolicyValueModel {
  return seatModels.find((entry) => entry.seat === player)?.model ?? fallback;
}

function playEvaluationGame(
  seed: string,
  candidateSeat: number,
  candidate: PolicyValueModel,
  opponent: PolicyValueModel,
  options: Required<Pick<EvaluationOptions, "simulationsPerMove" | "determinizations" | "maxTurns">>,
  rng: RandomSource
): { state: GameState; moves: Move[] } {
  let state = createGame(seed);
  const moves: Move[] = [];

  while (!isTerminal(state) && moves.length < options.maxTurns) {
    const player = state.currentPlayer;
    const model = selectModel(player, [{ seat: candidateSeat, model: candidate }], opponent);
    const result = searchMoveForObserver(state, player, model, {
      simulations: options.simulationsPerMove,
      determinizations: options.determinizations,
      rng,
      rootPlayer: player
    });
    moves.push(result.move);
    state = applyMove(state, result.move);
  }

  return { state, moves };
}

export function evaluateCandidateAgainstOpponent(
  candidate: PolicyValueModel,
  opponent: PolicyValueModel = new HandcraftedModel(),
  options: EvaluationOptions = {}
): EvaluationResult {
  const games = options.games ?? 12;
  const simulationsPerMove = options.simulationsPerMove ?? 48;
  const determinizations = options.determinizations ?? 1;
  const maxTurns = options.maxTurns ?? 280;
  const seed = String(options.seed ?? Date.now());
  const rng = createRng(seed);
  let totalReward = 0;
  let firstPlaces = 0;
  let totalTurns = 0;

  for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
    const candidateSeat = gameIndex % PLAYER_COUNT;
    const result = playEvaluationGame(
      `${seed}:eval:${gameIndex}`,
      candidateSeat,
      candidate,
      opponent,
      { simulationsPerMove, determinizations, maxTurns },
      rng
    );
    const order = placements(result.state);
    totalReward += rewardForPlayer(result.state, candidateSeat);
    firstPlaces += order[0] === candidateSeat ? 1 : 0;
    totalTurns += result.moves.length;
  }

  return {
    games,
    averageReward: totalReward / games,
    firstPlaceRate: firstPlaces / games,
    averageTurns: totalTurns / games
  };
}
