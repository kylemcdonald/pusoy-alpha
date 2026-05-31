import {
  Card,
  LOWEST_CARD,
  cardId,
  containsCard,
  newDeck,
  removeCards,
  sameCard,
  sortCards
} from "./cards";
import {
  Combo,
  Move,
  PASS_MOVE,
  beats,
  classifyCombo,
  containsLowestCard,
  enumerateCombos,
  formatMove,
  moveKey,
  playMove
} from "./combinations";
import { RandomSource, createRng, shuffle } from "./random";

export const PLAYER_COUNT = 2;

export interface HistoryEntry {
  turn: number;
  player: number;
  move: Move;
  combo: Combo | null;
  activeCombo: Combo | null;
  remaining: number[];
  finished: number[];
}

export interface GameState {
  id: string;
  hands: Card[][];
  currentPlayer: number;
  activeCombo: Combo | null;
  lastPlayer: number | null;
  passesSincePlay: number;
  finished: number[];
  history: HistoryEntry[];
}

export function createGame(seed: string | number = Date.now()): GameState {
  const rng = createRng(seed);
  const deck = shuffle(newDeck(), rng);
  const cardsInPlay = PLAYER_COUNT * 13;
  const lowestIndex = deck.findIndex((card) => sameCard(card, LOWEST_CARD));
  if (lowestIndex >= cardsInPlay) {
    const swapIndex = Math.floor(rng() * cardsInPlay);
    [deck[lowestIndex], deck[swapIndex]] = [deck[swapIndex], deck[lowestIndex]];
  }
  const hands = Array.from({ length: PLAYER_COUNT }, (_, player) => {
    return sortCards(deck.slice(player * 13, player * 13 + 13));
  });
  const currentPlayer = hands.findIndex((hand) => containsCard(hand, LOWEST_CARD));

  return {
    id: String(seed),
    hands,
    currentPlayer,
    activeCombo: null,
    lastPlayer: null,
    passesSincePlay: 0,
    finished: [],
    history: []
  };
}

export function cloneGameState(state: GameState): GameState {
  return {
    ...state,
    hands: state.hands.map((hand) => hand.map((card) => ({ ...card }))),
    activeCombo: state.activeCombo ? { ...state.activeCombo, cards: [...state.activeCombo.cards] } : null,
    finished: [...state.finished],
    history: state.history.map((entry) => ({
      ...entry,
      move: entry.move.type === "pass" ? PASS_MOVE : playMove(entry.move.cards),
      combo: entry.combo ? { ...entry.combo, cards: [...entry.combo.cards] } : null,
      activeCombo: entry.activeCombo ? { ...entry.activeCombo, cards: [...entry.activeCombo.cards] } : null,
      remaining: [...entry.remaining],
      finished: [...entry.finished]
    }))
  };
}

export function activePlayers(state: GameState): number[] {
  return Array.from({ length: state.hands.length }, (_, player) => player).filter((player) => {
    return !state.finished.includes(player) && state.hands[player].length > 0;
  });
}

export function isTerminal(state: GameState): boolean {
  return activePlayers(state).length <= 1;
}

export function placements(state: GameState): number[] {
  const placed = [...state.finished];
  for (const player of activePlayers(state)) {
    if (!placed.includes(player)) {
      placed.push(player);
    }
  }
  return placed;
}

export function rewardForPlayer(state: GameState, player: number): number {
  const order = placements(state);
  const place = order.indexOf(player);
  if (place < 0) return -1;
  if (order.length <= 1) return 1;
  if (place === 0) return 1;
  if (place === order.length - 1) return -1;
  return 1 - (2 * place) / (order.length - 1);
}

function nextActivePlayer(state: GameState, afterPlayer: number): number {
  for (let offset = 1; offset <= state.hands.length; offset += 1) {
    const player = (afterPlayer + offset) % state.hands.length;
    if (!state.finished.includes(player) && state.hands[player].length > 0) {
      return player;
    }
  }
  return -1;
}

export function legalMoves(state: GameState): Move[] {
  if (isTerminal(state) || state.currentPlayer < 0) {
    return [];
  }

  const player = state.currentPlayer;
  const isFirstMove = state.history.length === 0;
  let combos = enumerateCombos(state.hands[player]);

  if (isFirstMove) {
    combos = combos.filter(containsLowestCard);
  }

  if (state.activeCombo) {
    combos = combos.filter((combo) => beats(combo, state.activeCombo));
  }

  const moves = combos.map((combo) => playMove(combo.cards));
  if (state.activeCombo && !isFirstMove) {
    moves.push(PASS_MOVE);
  }

  return moves;
}

export function isLegalMove(state: GameState, move: Move): boolean {
  return legalMoves(state).some((legalMove) => moveKey(legalMove) === moveKey(move));
}

export function applyMove(state: GameState, move: Move): GameState {
  const legal = legalMoves(state);
  const normalized = legal.find((candidate) => moveKey(candidate) === moveKey(move));
  if (!normalized) {
    throw new Error(`Illegal move for player ${state.currentPlayer}: ${formatMove(move)}`);
  }

  return applyLegalMove(state, normalized);
}

export function applyLegalMove(state: GameState, normalized: Move): GameState {
  const next = cloneGameState(state);
  const player = next.currentPlayer;
  let combo: Combo | null = null;

  if (normalized.type === "pass") {
    next.passesSincePlay += 1;
    const remainingAfterPass = activePlayers(next).length;

    if (
      next.activeCombo &&
      next.lastPlayer !== null &&
      next.passesSincePlay >= Math.max(1, remainingAfterPass - 1) &&
      !next.finished.includes(next.lastPlayer)
    ) {
      next.currentPlayer = next.lastPlayer;
      next.activeCombo = null;
      next.lastPlayer = null;
      next.passesSincePlay = 0;
    } else {
      next.currentPlayer = nextActivePlayer(next, player);
    }
  } else {
    combo = classifyCombo(normalized.cards);
    if (!combo) {
      throw new Error(`Cannot apply invalid combo: ${normalized.cards.map(cardId).join(" ")}`);
    }

    next.hands[player] = removeCards(next.hands[player], normalized.cards);
    const wentOut = next.hands[player].length === 0;
    if (wentOut && !next.finished.includes(player)) {
      next.finished.push(player);
    }

    if (wentOut) {
      next.activeCombo = null;
      next.lastPlayer = null;
      next.passesSincePlay = 0;
    } else {
      next.activeCombo = combo;
      next.lastPlayer = player;
      next.passesSincePlay = 0;
    }

    next.currentPlayer = isTerminal(next) ? -1 : nextActivePlayer(next, player);
  }

  next.history.push({
    turn: next.history.length + 1,
    player,
    move: normalized,
    combo,
    activeCombo: next.activeCombo,
    remaining: next.hands.map((hand) => hand.length),
    finished: [...next.finished]
  });

  return next;
}

export function cardsPlayed(state: GameState): Card[] {
  return state.history.flatMap((entry) => (entry.move.type === "play" ? entry.move.cards : []));
}

export function findMoveByCards(state: GameState, cards: Card[]): Move | null {
  const key = moveKey(playMove(cards));
  return legalMoves(state).find((move) => moveKey(move) === key) ?? null;
}

export function hasLowestCardInFirstMove(state: GameState, cards: Card[]): boolean {
  return state.history.length > 0 || containsCard(cards, LOWEST_CARD);
}
