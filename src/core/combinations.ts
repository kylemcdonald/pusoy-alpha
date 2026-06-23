import {
  Card,
  LOWEST_CARD,
  RANKS,
  Rank,
  cardId,
  cardPower,
  containsCard,
  handKey,
  rankValue,
  sortCards,
  sortCardsDesc,
  suitValue
} from "./cards";

export type ComboKind =
  | "single"
  | "pair"
  | "triple"
  | "straight"
  | "flush"
  | "full-house"
  | "four-kind"
  | "straight-flush";

export interface Combo {
  kind: ComboKind;
  cards: Card[];
  size: 1 | 2 | 3 | 5;
  tiebreak: number[];
  label: string;
}

export type Move = { type: "pass" } | { type: "play"; cards: Card[] };

export interface ComboRules {
  allowWraparoundStraights: boolean;
}

export const DEFAULT_COMBO_RULES: ComboRules = {
  allowWraparoundStraights: false
};

const FIVE_CARD_KIND_ORDER: Record<ComboKind, number> = {
  single: -1,
  pair: -1,
  triple: -1,
  straight: 0,
  flush: 1,
  "full-house": 2,
  "four-kind": 3,
  "straight-flush": 4
};

const STRAIGHT_PATTERNS: Rank[][] = [
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
  ["J", "Q", "K", "A", "2"]
];
const NON_WRAP_STRAIGHT_PATTERN_COUNT = 8;

interface StraightInfo {
  order: number;
  topSuit: number;
}

export function normalizeComboRules(rules: Partial<ComboRules> | null | undefined): ComboRules {
  return {
    allowWraparoundStraights: Boolean(rules?.allowWraparoundStraights)
  };
}

function rankCounts(cards: Card[]): Map<Rank, Card[]> {
  const groups = new Map<Rank, Card[]>();
  for (const card of cards) {
    const group = groups.get(card.rank) ?? [];
    group.push(card);
    groups.set(card.rank, group);
  }
  return groups;
}

function detectStraight(cards: Card[], rules: ComboRules): StraightInfo | null {
  const uniqueRanks = new Set(cards.map((card) => card.rank));
  if (uniqueRanks.size !== 5) {
    return null;
  }

  const signature = [...uniqueRanks].sort((a, b) => rankValue(a) - rankValue(b)).join("-");
  const order = STRAIGHT_PATTERNS.findIndex((pattern, patternIndex) => {
    if (!rules.allowWraparoundStraights && patternIndex >= NON_WRAP_STRAIGHT_PATTERN_COUNT) {
      return false;
    }
    return [...pattern].sort((a, b) => rankValue(a) - rankValue(b)).join("-") === signature;
  });

  if (order < 0) {
    return null;
  }

  const strongestRank = cards.reduce((best, card) => {
    return rankValue(card.rank) > rankValue(best) ? card.rank : best;
  }, cards[0].rank);
  const topSuit = Math.max(
    ...cards.filter((card) => card.rank === strongestRank).map((card) => suitValue(card.suit))
  );

  return { order, topSuit };
}

function comboLabel(kind: ComboKind, cards: Card[]): string {
  const cardText = sortCards(cards).map(cardId).join(" ");
  return `${kind.replace("-", " ")}: ${cardText}`;
}

export function classifyCombo(inputCards: Card[], rawRules?: Partial<ComboRules> | null): Combo | null {
  const rules = normalizeComboRules(rawRules);
  const cards = sortCards(inputCards);
  const size = cards.length;
  const groups = rankCounts(cards);

  if (size === 1) {
    return {
      kind: "single",
      cards,
      size,
      tiebreak: [cardPower(cards[0])],
      label: comboLabel("single", cards)
    };
  }

  if (size === 2 && groups.size === 1) {
    return {
      kind: "pair",
      cards,
      size,
      tiebreak: [rankValue(cards[0].rank), Math.max(...cards.map((card) => suitValue(card.suit)))],
      label: comboLabel("pair", cards)
    };
  }

  if (size === 3 && groups.size === 1) {
    return {
      kind: "triple",
      cards,
      size,
      tiebreak: [rankValue(cards[0].rank)],
      label: comboLabel("triple", cards)
    };
  }

  if (size !== 5) {
    return null;
  }

  const straight = detectStraight(cards, rules);
  const isFlush = cards.every((card) => card.suit === cards[0].suit);
  const groupSizes = [...groups.values()]
    .map((group) => group.length)
    .sort((a, b) => b - a);

  if (straight && isFlush) {
    return {
      kind: "straight-flush",
      cards,
      size,
      tiebreak: [FIVE_CARD_KIND_ORDER["straight-flush"], straight.order, straight.topSuit],
      label: comboLabel("straight-flush", cards)
    };
  }

  if (groupSizes[0] === 4) {
    const quad = [...groups.entries()].find(([, group]) => group.length === 4);
    if (!quad) {
      return null;
    }
    return {
      kind: "four-kind",
      cards,
      size,
      tiebreak: [FIVE_CARD_KIND_ORDER["four-kind"], rankValue(quad[0])],
      label: comboLabel("four-kind", cards)
    };
  }

  if (groupSizes[0] === 3 && groupSizes[1] === 2) {
    const triple = [...groups.entries()].find(([, group]) => group.length === 3);
    if (!triple) {
      return null;
    }
    return {
      kind: "full-house",
      cards,
      size,
      tiebreak: [FIVE_CARD_KIND_ORDER["full-house"], rankValue(triple[0])],
      label: comboLabel("full-house", cards)
    };
  }

  if (isFlush) {
    const descending = sortCardsDesc(cards);
    return {
      kind: "flush",
      cards,
      size,
      tiebreak: [
        FIVE_CARD_KIND_ORDER.flush,
        suitValue(cards[0].suit),
        ...descending.map((card) => rankValue(card.rank))
      ],
      label: comboLabel("flush", cards)
    };
  }

  if (straight) {
    return {
      kind: "straight",
      cards,
      size,
      tiebreak: [FIVE_CARD_KIND_ORDER.straight, straight.order, straight.topSuit],
      label: comboLabel("straight", cards)
    };
  }

  return null;
}

export function compareCombos(a: Combo, b: Combo): number {
  if (a.size !== b.size) {
    throw new Error(`Cannot compare ${a.size}-card combo to ${b.size}-card combo`);
  }

  if (a.size === 5) {
    const kindDiff = FIVE_CARD_KIND_ORDER[a.kind] - FIVE_CARD_KIND_ORDER[b.kind];
    if (kindDiff !== 0) {
      return kindDiff;
    }
  } else if (a.kind !== b.kind) {
    throw new Error(`Cannot compare ${a.kind} to ${b.kind}`);
  }

  const length = Math.max(a.tiebreak.length, b.tiebreak.length);
  for (let index = 0; index < length; index += 1) {
    const diff = (a.tiebreak[index] ?? 0) - (b.tiebreak[index] ?? 0);
    if (diff !== 0) {
      return diff;
    }
  }

  return 0;
}

export function beats(candidate: Combo, target: Combo | null): boolean {
  if (!target) {
    return true;
  }
  if (candidate.size !== target.size) {
    return false;
  }
  return compareCombos(candidate, target) > 0;
}

export function containsLowestCard(combo: Combo): boolean {
  return containsCard(combo.cards, LOWEST_CARD);
}

function chooseN<T>(items: T[], size: number): T[][] {
  if (size === 0) {
    return [[]];
  }
  if (items.length < size) {
    return [];
  }

  const [head, ...tail] = items;
  return [
    ...chooseN(tail, size - 1).map((choice) => [head, ...choice]),
    ...chooseN(tail, size)
  ];
}

const COMBO_CACHE_LIMIT = 100_000;
const comboCache = new Map<string, Combo[]>();

function rememberCombos(key: string, combos: Combo[]): Combo[] {
  if (comboCache.size >= COMBO_CACHE_LIMIT) {
    const firstKey = comboCache.keys().next().value as string | undefined;
    if (firstKey) {
      comboCache.delete(firstKey);
    }
  }
  comboCache.set(key, combos);
  return combos;
}

function comboRulesKey(rules: ComboRules): string {
  return rules.allowWraparoundStraights ? "wrap" : "no-wrap";
}

export function enumerateCombos(hand: Card[], rawRules?: Partial<ComboRules> | null): Combo[] {
  const rules = normalizeComboRules(rawRules);
  const key = `${comboRulesKey(rules)}:${handKey(hand)}`;
  const cached = comboCache.get(key);
  if (cached) {
    return cached;
  }

  const cards = sortCards(hand);
  const combos: Combo[] = [];

  for (const card of cards) {
    const combo = classifyCombo([card], rules);
    if (combo) {
      combos.push(combo);
    }
  }

  const groups = rankCounts(cards);
  for (const group of groups.values()) {
    for (const pair of chooseN(group, 2)) {
      const combo = classifyCombo(pair, rules);
      if (combo) {
        combos.push(combo);
      }
    }
    for (const triple of chooseN(group, 3)) {
      const combo = classifyCombo(triple, rules);
      if (combo) {
        combos.push(combo);
      }
    }
  }

  for (const fiveCards of chooseN(cards, 5)) {
    const combo = classifyCombo(fiveCards, rules);
    if (combo) {
      combos.push(combo);
    }
  }

  const unique = new Map<string, Combo>();
  for (const combo of combos) {
    unique.set(comboKey(combo), combo);
  }

  const sorted = [...unique.values()].sort((a, b) => {
    const sizeDiff = a.size - b.size;
    if (sizeDiff !== 0) {
      return sizeDiff;
    }
    if (a.size === b.size) {
      return compareCombosSafe(a, b);
    }
    return 0;
  });

  return rememberCombos(key, sorted);
}

function compareCombosSafe(a: Combo, b: Combo): number {
  try {
    return compareCombos(a, b);
  } catch {
    return a.kind.localeCompare(b.kind);
  }
}

export function comboKey(combo: Combo): string {
  return sortCards(combo.cards).map(cardId).join("-");
}

export function moveKey(move: Move, rules?: Partial<ComboRules> | null): string {
  if (move.type === "pass") {
    return "PASS";
  }
  const combo = classifyCombo(move.cards, rules);
  return combo ? comboKey(combo) : sortCards(move.cards).map(cardId).join("-");
}

export function formatMove(move: Move, rules?: Partial<ComboRules> | null): string {
  if (move.type === "pass") {
    return "Pass";
  }
  const combo = classifyCombo(move.cards, rules);
  return combo ? combo.label : sortCards(move.cards).map(cardId).join(" ");
}

export function playMove(cards: Card[]): Move {
  return { type: "play", cards: sortCards(cards) };
}

export const PASS_MOVE: Move = { type: "pass" };
