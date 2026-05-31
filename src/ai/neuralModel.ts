import { Card, cardId, rankValue, suitIndex } from "../core/cards";
import { Move, classifyCombo, moveKey } from "../core/combinations";
import { GameState, applyLegalMove, isTerminal, legalMoves, rewardForPlayer } from "../core/game";
import { HandcraftedModel, ModelEvaluation, PolicyValueModel } from "./model";

type LayerWeights = {
  weight: number[][];
  bias: number[];
};

export interface NeuralModelFile {
  version: number;
  name: string;
  createdAt: string;
  architecture: {
    stateDim: number;
    moveDim: number;
    comboKinds: Record<string, number>;
  };
  training: {
    device: string;
    style?: string;
    requestedMinutes?: number;
    elapsedSeconds: number;
    games: number;
    transitions: number;
    mctsSimulations?: number;
    replaySize?: number;
    trainSamples?: number;
    history: Array<Record<string, number>>;
    note?: string;
  };
  inference?: {
    handcraftedPriorWeight?: number;
    handcraftedValueWeight?: number;
    neuralPolicyTemperature?: number;
    opponentAwarePriorWeight?: number;
    endgameSolverCards?: number;
  };
  weights: {
    state_fc: LayerWeights;
    move_fc: LayerWeights;
    policy_fc: LayerWeights;
    policy_out: LayerWeights;
    value_fc: LayerWeights;
    value_out: LayerWeights;
  };
}

const SUPPORTED_STATE_DIMS = new Set([216]);
const MOVE_DIM = 62;

function cardIndex(card: Card): number {
  return rankValue(card.rank) * 4 + suitIndex(card.suit);
}

function relu(values: number[]): number[] {
  return values.map((value) => Math.max(0, value));
}

function tanh(value: number): number {
  return Math.tanh(value);
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function linear(input: number[], layer: LayerWeights): number[] {
  return layer.weight.map((row, rowIndex) => {
    let sum = layer.bias[rowIndex] ?? 0;
    for (let index = 0; index < input.length; index += 1) {
      sum += row[index] * input[index];
    }
    return sum;
  });
}

function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - max));
  const total = exp.reduce((sum, value) => sum + value, 0);
  return exp.map((value) => value / total);
}

function normalize(values: number[]): number[] {
  const total = values.reduce((sum, value) => sum + Math.max(0.0001, value), 0);
  return values.map((value) => Math.max(0.0001, value) / total);
}

function stateFeatures(state: GameState, player: number, stateDim: number): number[] {
  const handBits = Array.from({ length: 52 }, () => 0);
  for (const card of state.hands[player]) {
    handBits[cardIndex(card)] = 1;
  }

  const selfPlayedBits = Array.from({ length: 52 }, () => 0);
  const opponentPlayedBits = Array.from({ length: 52 }, () => 0);
  for (const entry of state.history) {
    if (entry.move.type === "play") {
      for (const card of entry.move.cards) {
        const index = cardIndex(card);
        if (entry.player === player) {
          selfPlayedBits[index] = 1;
        } else {
          opponentPlayedBits[index] = 1;
        }
      }
    }
  }

  const activeBits = Array.from({ length: 52 }, () => 0);
  if (state.activeCombo) {
    for (const card of state.activeCombo.cards) {
      activeBits[cardIndex(card)] = 1;
    }
  }

  const opponent = 1 - player;
  const activeOwnerSelf = state.activeCombo && state.lastPlayer === player ? 1 : 0;
  const activeOwnerOpponent = state.activeCombo && state.lastPlayer === opponent ? 1 : 0;
  const scalars = [
    state.hands[player].length / 13,
    (state.hands[opponent]?.length ?? 0) / 13,
    (state.activeCombo?.size ?? 0) / 5,
    state.activeCombo ? 0 : 1,
    state.passesSincePlay / 2,
    Math.min(state.history.length, 100) / 100
  ];

  return [
    ...handBits,
    ...selfPlayedBits,
    ...opponentPlayedBits,
    ...activeBits,
    ...scalars,
    activeOwnerSelf,
    activeOwnerOpponent
  ];
}

function moveFeatures(move: Move, comboKinds: Record<string, number>): number[] {
  const bits = Array.from({ length: 52 }, () => 0);
  let pass = 0;
  let size = 0;
  const kind = Array.from({ length: 8 }, () => 0);

  if (move.type === "pass") {
    pass = 1;
  } else {
    const combo = classifyCombo(move.cards);
    for (const card of move.cards) {
      bits[cardIndex(card)] = 1;
    }
    if (combo) {
      size = combo.size;
      const kindIndex = comboKinds[combo.kind];
      if (kindIndex !== undefined) {
        kind[kindIndex] = 1;
      }
    }
  }

  return [...bits, pass, size / 5, ...kind];
}

export class NeuralPolicyValueModel implements PolicyValueModel {
  readonly name: string;
  private readonly handcrafted = new HandcraftedModel();
  private readonly handcraftedPriorWeight: number;
  private readonly handcraftedValueWeight: number;
  private readonly neuralPolicyTemperature: number;
  private readonly opponentAwarePriorWeight: number;
  private readonly endgameSolverCards: number;
  private readonly endgameCache = new Map<string, number>();

  constructor(private readonly file: NeuralModelFile) {
    this.name = file.name;
    if (!SUPPORTED_STATE_DIMS.has(file.architecture.stateDim) || file.architecture.moveDim !== MOVE_DIM) {
      throw new Error(`Unsupported neural model dimensions: ${file.architecture.stateDim}/${file.architecture.moveDim}`);
    }
    this.handcraftedPriorWeight = clamp01(file.inference?.handcraftedPriorWeight ?? 0);
    this.handcraftedValueWeight = clamp01(file.inference?.handcraftedValueWeight ?? 0);
    this.neuralPolicyTemperature = Math.max(0.05, file.inference?.neuralPolicyTemperature ?? 1);
    this.opponentAwarePriorWeight = clamp01(file.inference?.opponentAwarePriorWeight ?? 0);
    this.endgameSolverCards = Math.max(0, Math.floor(file.inference?.endgameSolverCards ?? 0));
  }

  evaluate(state: GameState, perspectivePlayer: number, moves = legalMoves(state)): ModelEvaluation {
    if (isTerminal(state)) {
      return { priors: new Map(), value: rewardForPlayer(state, perspectivePlayer) };
    }

    const weights = this.file.weights;
    const stateInput = stateFeatures(state, perspectivePlayer, this.file.architecture.stateDim);
    const stateHidden = relu(linear(stateInput, weights.state_fc));
    const valueHidden = relu(linear(stateHidden, weights.value_fc));
    const logits = moves.map((move) => {
      const moveHidden = relu(linear(moveFeatures(move, this.file.architecture.comboKinds), weights.move_fc));
      const policyInput = [...stateHidden, ...moveHidden];
      const policyHidden = relu(linear(policyInput, weights.policy_fc));
      return linear(policyHidden, weights.policy_out)[0] ?? 0;
    });
    const scaledLogits = logits.map((logit) => logit / this.neuralPolicyTemperature);
    const probabilities = scaledLogits.length > 0 ? softmax(scaledLogits) : [];
    let value = tanh(linear(valueHidden, weights.value_out)[0] ?? 0);
    let baseProbabilities = probabilities;
    if (this.endgameSolverCards > 0 && remainingCards(state) <= this.endgameSolverCards) {
      value = this.solveEndgame(state, perspectivePlayer);
      baseProbabilities = exactPolicyPriors(state, perspectivePlayer, moves, (next) =>
        this.solveEndgame(next, perspectivePlayer)
      );
    }
    const handcraftedEvaluation =
      this.handcraftedPriorWeight > 0 || this.handcraftedValueWeight > 0
        ? this.handcrafted.evaluate(state, perspectivePlayer, moves)
        : null;
    const priors = new Map<string, number>();
    moves.forEach((move, index) => {
      const key = moveKey(move);
      const neuralPrior = baseProbabilities[index] ?? 0;
      const handcraftedPrior = handcraftedEvaluation?.priors.get(key) ?? neuralPrior;
      priors.set(
        key,
        (1 - this.handcraftedPriorWeight) * neuralPrior + this.handcraftedPriorWeight * handcraftedPrior
      );
    });
    if (this.opponentAwarePriorWeight > 0) {
      const awarePriors = opponentAwarePriors(state, moves);
      moves.forEach((move, index) => {
        const key = moveKey(move);
        const prior = priors.get(key) ?? 0;
        priors.set(key, (1 - this.opponentAwarePriorWeight) * prior + this.opponentAwarePriorWeight * awarePriors[index]);
      });
    }
    const blendedValue = handcraftedEvaluation
      ? (1 - this.handcraftedValueWeight) * value + this.handcraftedValueWeight * handcraftedEvaluation.value
      : value;

    return { priors, value: Math.max(-1, Math.min(1, blendedValue)) };
  }

  private solveEndgame(state: GameState, perspectivePlayer: number): number {
    const key = `${perspectivePlayer}:${endgameKey(state)}`;
    const cached = this.endgameCache.get(key);
    if (cached !== undefined) {
      return cached;
    }

    let value: number;
    if (isTerminal(state)) {
      value = rewardForPlayer(state, perspectivePlayer);
    } else {
      const moves = legalMoves(state);
      if (moves.length === 0) {
        value = 0;
      } else {
        const childValues = moves.map((move) => this.solveEndgame(applyLegalMove(state, move), perspectivePlayer));
        value =
          state.currentPlayer === perspectivePlayer
            ? Math.max(...childValues)
            : Math.min(...childValues);
      }
    }

    this.endgameCache.set(key, value);
    return value;
  }
}

function opponentCanAnswer(state: GameState, move: Move): boolean {
  if (move.type === "pass") {
    return true;
  }
  const next = applyLegalMove(state, move);
  if (isTerminal(next) || next.currentPlayer < 0 || next.activeCombo === null) {
    return false;
  }
  return legalMoves(next).some((reply) => reply.type === "play");
}

function opponentAwarePriors(state: GameState, moves: Move[]): number[] {
  const actor = state.currentPlayer;
  const opponent = 1 - actor;
  const scores = moves.map((move) => {
    if (move.type === "pass") {
      return 0.05;
    }
    const combo = classifyCombo(move.cards);
    if (!combo) {
      return 0.01;
    }

    const remaining = state.hands[actor].length - combo.size;
    const canAnswer = opponentCanAnswer(state, move);
    let score = 0.15 + combo.size * 0.25;
    if (remaining === 0) {
      score += 8;
    } else if (remaining <= 2) {
      score += 1.2;
    }
    score += canAnswer ? -0.35 : 1.5 + combo.size * 0.2;
    if (combo.cards.some((card) => card.rank === "2") && remaining > 0) {
      score -= 0.25;
    }
    if (state.hands[opponent]?.length === 1 && combo.size === 1) {
      score += canAnswer ? -0.8 : 1.5;
    }
    return Math.max(0.01, score);
  });
  return normalize(scores);
}

function remainingCards(state: GameState): number {
  return state.hands.reduce((sum, hand, player) => {
    return state.finished.includes(player) ? sum : sum + hand.length;
  }, 0);
}

function endgameKey(state: GameState): string {
  const active = state.activeCombo?.cards.map(cardId).join("-") ?? "";
  return [
    state.hands[0].map(cardId).join("-"),
    state.hands[1].map(cardId).join("-"),
    state.currentPlayer,
    active,
    state.lastPlayer ?? "n",
    state.passesSincePlay,
    state.finished.join("-"),
    state.history.length === 0 ? 0 : 1
  ].join("|");
}

function exactPolicyPriors(state: GameState, perspectivePlayer: number, moves: Move[], solver: (state: GameState) => number): number[] {
  if (moves.length === 0) {
    return [];
  }
  const scale = state.currentPlayer === perspectivePlayer ? 4 : -4;
  const logits = moves.map((move) => solver(applyLegalMove(state, move)) * scale);
  return softmax(logits);
}
