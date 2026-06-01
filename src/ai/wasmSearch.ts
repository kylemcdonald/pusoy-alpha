import { GameState, PASS_MOVE, RANKS, SUITS, legalMoves, moveKey, playMove, rankValue, suitIndex } from "../core";
import { hashSeed } from "../core/random";
import type { NeuralModelFile } from "./neuralModel";
import type { SearchResult } from "./mcts";
import initWasm, { init_model, observer_search } from "../wasm/pkg/pusoy_alpha_wasm.js";

export interface WasmSearchBudget {
  timeLimitMs: number;
  simulationsPerDetermination: number;
}

export class WasmSearchEngine {
  search(state: GameState, observer: number, seed: string, budget: WasmSearchBudget): SearchResult | null {
    const packed = packState(state);
    const raw = observer_search(
      packed,
      observer,
      hashSeed(seed),
      budget.timeLimitMs,
      budget.simulationsPerDetermination
    );
    if (raw.length < 9) {
      return null;
    }

    const moveLength = raw[0];
    const move =
      moveLength === 0
        ? PASS_MOVE
        : playMove(Array.from({ length: moveLength }, (_, index) => cardFromWasmId(raw[index + 1])));
    const legal = legalMoves(state).find((candidate) => moveKey(candidate) === moveKey(move));
    if (!legal) {
      return null;
    }

    return {
      move: legal,
      visits: { [moveKey(legal)]: raw[7] },
      value: 0,
      determinizations: Math.max(0, raw[6]),
      simulations: Math.max(0, raw[7]),
      legalMoveCount: Math.max(0, raw[8])
    };
  }
}

let wasmInit: Promise<void> | null = null;

export async function createWasmSearchEngine(model: NeuralModelFile): Promise<WasmSearchEngine> {
  if ((model.inference?.opponentAwarePriorWeight ?? 0) > 0 || (model.inference?.endgameSolverCards ?? 0) > 0) {
    throw new Error("WASM search does not support opponent-aware priors or exact endgame solving yet");
  }

  wasmInit ??= initWasm().then(() => undefined);
  await wasmInit;

  const initialized = init_model(flattenWeights(model), modelDims(model), modelSettings(model));
  if (!initialized) {
    throw new Error("WASM model initialization failed");
  }

  return new WasmSearchEngine();
}

function modelDims(model: NeuralModelFile): Uint32Array {
  return new Uint32Array([
    model.architecture.stateDim,
    model.architecture.moveDim,
    model.weights.state_fc.bias.length,
    model.weights.move_fc.bias.length,
    model.weights.policy_fc.bias.length,
    model.weights.value_fc.bias.length
  ]);
}

function modelSettings(model: NeuralModelFile): Float32Array {
  return new Float32Array([
    model.inference?.handcraftedPriorWeight ?? 0,
    model.inference?.handcraftedValueWeight ?? 0,
    model.inference?.neuralPolicyTemperature ?? 1,
    model.inference?.opponentAwarePriorWeight ?? 0,
    model.inference?.endgameSolverCards ?? 0,
    model.inference?.rolloutValueWeight ?? 0
  ]);
}

function flattenWeights(model: NeuralModelFile): Float32Array {
  const values: number[] = [];
  pushLayer(values, model.weights.state_fc);
  pushLayer(values, model.weights.move_fc);
  pushLayer(values, model.weights.policy_fc);
  pushLayer(values, model.weights.policy_out);
  pushLayer(values, model.weights.value_fc);
  pushLayer(values, model.weights.value_out);
  return new Float32Array(values);
}

function pushLayer(values: number[], layer: { weight: number[][]; bias: number[] }) {
  for (const row of layer.weight) {
    values.push(...row);
  }
  values.push(...layer.bias);
}

function packState(state: GameState): Int32Array {
  const values: number[] = [
    state.currentPlayer,
    state.activeCombo?.cards.length ?? 0,
    ...fixedCards(state.activeCombo?.cards ?? [], 5),
    state.lastPlayer ?? -1,
    state.passesSincePlay,
    state.finished.includes(0) ? 1 : 0,
    state.finished.includes(1) ? 1 : 0,
    state.history.length
  ];
  pushCards(values, cardsPlayedBy(state, 0));
  pushCards(values, cardsPlayedBy(state, 1));
  pushCards(values, state.hands[0]);
  pushCards(values, state.hands[1]);
  return new Int32Array(values);
}

function fixedCards(cards: { rank: (typeof RANKS)[number]; suit: (typeof SUITS)[number] }[], size: number): number[] {
  return Array.from({ length: size }, (_, index) => {
    const card = cards[index];
    return card ? wasmCardId(card) : -1;
  });
}

function pushCards(values: number[], cards: { rank: (typeof RANKS)[number]; suit: (typeof SUITS)[number] }[]) {
  values.push(cards.length);
  for (const card of cards) {
    values.push(wasmCardId(card));
  }
}

function cardsPlayedBy(state: GameState, player: number) {
  return state.history.flatMap((entry) => (entry.player === player && entry.move.type === "play" ? entry.move.cards : []));
}

function wasmCardId(card: { rank: (typeof RANKS)[number]; suit: (typeof SUITS)[number] }): number {
  return rankValue(card.rank) * 4 + suitIndex(card.suit);
}

function cardFromWasmId(id: number) {
  const rank = RANKS[Math.floor(id / 4)];
  const suit = SUITS[id % 4];
  if (!rank || !suit) {
    throw new Error(`Invalid WASM card id: ${id}`);
  }
  return { rank, suit };
}
