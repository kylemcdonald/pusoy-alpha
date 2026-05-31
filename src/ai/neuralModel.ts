import { Card, rankValue, suitIndex } from "../core/cards";
import { Move, classifyCombo, moveKey } from "../core/combinations";
import { GameState, legalMoves, rewardForPlayer, isTerminal } from "../core/game";
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

  constructor(private readonly file: NeuralModelFile) {
    this.name = file.name;
    if (!SUPPORTED_STATE_DIMS.has(file.architecture.stateDim) || file.architecture.moveDim !== MOVE_DIM) {
      throw new Error(`Unsupported neural model dimensions: ${file.architecture.stateDim}/${file.architecture.moveDim}`);
    }
    this.handcraftedPriorWeight = clamp01(file.inference?.handcraftedPriorWeight ?? 0);
    this.handcraftedValueWeight = clamp01(file.inference?.handcraftedValueWeight ?? 0);
    this.neuralPolicyTemperature = Math.max(0.05, file.inference?.neuralPolicyTemperature ?? 1);
  }

  evaluate(state: GameState, perspectivePlayer: number, moves = legalMoves(state)): ModelEvaluation {
    if (isTerminal(state)) {
      return { priors: new Map(), value: rewardForPlayer(state, perspectivePlayer) };
    }

    const weights = this.file.weights;
    const stateInput = stateFeatures(state, perspectivePlayer, this.file.architecture.stateDim);
    const stateHidden = relu(linear(stateInput, weights.state_fc));
    const valueHidden = relu(linear(stateHidden, weights.value_fc));
    const value = tanh(linear(valueHidden, weights.value_out)[0] ?? 0);
    const logits = moves.map((move) => {
      const moveHidden = relu(linear(moveFeatures(move, this.file.architecture.comboKinds), weights.move_fc));
      const policyInput = [...stateHidden, ...moveHidden];
      const policyHidden = relu(linear(policyInput, weights.policy_fc));
      return linear(policyHidden, weights.policy_out)[0] ?? 0;
    });
    const scaledLogits = logits.map((logit) => logit / this.neuralPolicyTemperature);
    const probabilities = scaledLogits.length > 0 ? softmax(scaledLogits) : [];
    const handcraftedEvaluation =
      this.handcraftedPriorWeight > 0 || this.handcraftedValueWeight > 0
        ? this.handcrafted.evaluate(state, perspectivePlayer, moves)
        : null;
    const priors = new Map<string, number>();
    moves.forEach((move, index) => {
      const key = moveKey(move);
      const neuralPrior = probabilities[index] ?? 0;
      const handcraftedPrior = handcraftedEvaluation?.priors.get(key) ?? neuralPrior;
      priors.set(
        key,
        (1 - this.handcraftedPriorWeight) * neuralPrior + this.handcraftedPriorWeight * handcraftedPrior
      );
    });
    const blendedValue = handcraftedEvaluation
      ? (1 - this.handcraftedValueWeight) * value + this.handcraftedValueWeight * handcraftedEvaluation.value
      : value;

    return { priors, value: Math.max(-1, Math.min(1, blendedValue)) };
  }
}
