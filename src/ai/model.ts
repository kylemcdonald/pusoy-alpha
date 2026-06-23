import { cardPower } from "../core/cards";
import { Combo, Move, classifyCombo, moveKey } from "../core/combinations";
import { GameState, isTerminal, legalMoves, rewardForPlayer } from "../core/game";

export interface ModelEvaluation {
  priors: Map<string, number>;
  value: number;
}

export interface PolicyValueModel {
  readonly name: string;
  evaluate(state: GameState, perspectivePlayer: number, moves?: Move[]): ModelEvaluation;
}

function clamp(value: number, min = -1, max = 1): number {
  return Math.max(min, Math.min(max, value));
}

function comboStrength(combo: Combo): number {
  const raw = combo.tiebreak.reduce((sum, value, index) => sum + value / (index + 1), 0);
  return raw / 70;
}

function moveFeatureScore(state: GameState, move: Move): number {
  if (move.type === "pass") {
    return 0.08;
  }

  const actor = state.currentPlayer;
  const combo = classifyCombo(move.cards, state.rules);
  if (!combo) {
    return 0.01;
  }

  const remainingAfterMove = state.hands[actor].length - combo.cards.length;
  let score = 0.25 + combo.cards.length * 0.18;

  if (remainingAfterMove === 0) {
    score += 5;
  } else if (remainingAfterMove <= 2) {
    score += 0.8;
  }

  if (state.activeCombo) {
    score += Math.max(0, 0.45 - comboStrength(combo));
  } else {
    score += Math.max(0, 0.7 - comboStrength(combo));
  }

  const hasRankTwo = combo.cards.some((card) => card.rank === "2");
  if (hasRankTwo && remainingAfterMove > 0) {
    score -= 0.2;
  }

  const nextPlayer = (actor + 1) % state.hands.length;
  if (state.hands[nextPlayer].length === 1 && combo.size === 1) {
    score += comboStrength(combo) * 0.6;
  }

  return Math.max(0.01, score);
}

function handcraftedValue(state: GameState, perspectivePlayer: number): number {
  if (isTerminal(state)) {
    return rewardForPlayer(state, perspectivePlayer);
  }

  const myCount = state.hands[perspectivePlayer].length;
  const opponentCounts = state.hands
    .map((hand, player) => (player === perspectivePlayer ? null : hand.length))
    .filter((count): count is number => count !== null && count > 0);
  const averageOpponentCount =
    opponentCounts.reduce((sum, count) => sum + count, 0) / Math.max(1, opponentCounts.length);
  const countValue = (averageOpponentCount - myCount) / 13;
  const handStrength =
    state.hands[perspectivePlayer].reduce((sum, card) => sum + cardPower(card), 0) /
    Math.max(1, state.hands[perspectivePlayer].length * 51);
  const controlValue =
    state.currentPlayer === perspectivePlayer && state.activeCombo === null ? 0.12 : 0;

  return clamp(countValue * 1.25 + handStrength * 0.22 + controlValue);
}

function normalizePriors(scores: Map<string, number>): Map<string, number> {
  const total = [...scores.values()].reduce((sum, value) => sum + Math.max(0.0001, value), 0);
  const priors = new Map<string, number>();
  for (const [key, value] of scores.entries()) {
    priors.set(key, Math.max(0.0001, value) / total);
  }
  return priors;
}

export class HandcraftedModel implements PolicyValueModel {
  readonly name = "handcrafted";

  evaluate(state: GameState, perspectivePlayer: number, moves = legalMoves(state)): ModelEvaluation {
    const scores = new Map<string, number>();
    for (const move of moves) {
      scores.set(moveKey(move, state.rules), moveFeatureScore(state, move));
    }

    return {
      priors: normalizePriors(scores),
      value: handcraftedValue(state, perspectivePlayer)
    };
  }
}
