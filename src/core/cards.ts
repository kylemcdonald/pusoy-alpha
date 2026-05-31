export const RANKS = [
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  "10",
  "J",
  "Q",
  "K",
  "A",
  "2"
] as const;

export const SUITS = ["C", "S", "H", "D"] as const;

export type Rank = (typeof RANKS)[number];
export type Suit = (typeof SUITS)[number];
export type CardId = `${Rank}${Suit}`;
export const SUIT_STRENGTH_ORDER: readonly Suit[] = ["D", "C", "H", "S"];

export interface Card {
  rank: Rank;
  suit: Suit;
}

const RANK_VALUES = new Map<Rank, number>(RANKS.map((rank, index) => [rank, index]));
const SUIT_VALUES = new Map<Suit, number>(SUIT_STRENGTH_ORDER.map((suit, index) => [suit, index]));
const SUIT_INDICES = new Map<Suit, number>(SUITS.map((suit, index) => [suit, index]));

export const LOWEST_CARD: Card = { rank: "3", suit: "D" };

export function rankValue(rank: Rank): number {
  const value = RANK_VALUES.get(rank);
  if (value === undefined) {
    throw new Error(`Unknown rank: ${rank}`);
  }
  return value;
}

export function suitValue(suit: Suit): number {
  const value = SUIT_VALUES.get(suit);
  if (value === undefined) {
    throw new Error(`Unknown suit: ${suit}`);
  }
  return value;
}

export function suitIndex(suit: Suit): number {
  const value = SUIT_INDICES.get(suit);
  if (value === undefined) {
    throw new Error(`Unknown suit: ${suit}`);
  }
  return value;
}

export function cardId(card: Card): CardId {
  return `${card.rank}${card.suit}` as CardId;
}

export function cardFromId(id: string): Card {
  const suit = id.slice(-1) as Suit;
  const rank = id.slice(0, -1) as Rank;

  if (!RANKS.includes(rank) || !SUITS.includes(suit)) {
    throw new Error(`Invalid card id: ${id}`);
  }

  return { rank, suit };
}

export function cardPower(card: Card): number {
  return rankValue(card.rank) * SUITS.length + suitValue(card.suit);
}

export function compareCards(a: Card, b: Card): number {
  return cardPower(a) - cardPower(b);
}

export function sameCard(a: Card, b: Card): boolean {
  return a.rank === b.rank && a.suit === b.suit;
}

export function sortCards(cards: Card[]): Card[] {
  return [...cards].sort(compareCards);
}

export function sortCardsDesc(cards: Card[]): Card[] {
  return [...cards].sort((a, b) => compareCards(b, a));
}

export function newDeck(): Card[] {
  const deck: Card[] = [];
  for (const rank of RANKS) {
    for (const suit of SUITS) {
      deck.push({ rank, suit });
    }
  }
  return deck;
}

export function containsCard(cards: Card[], target: Card): boolean {
  return cards.some((card) => sameCard(card, target));
}

export function removeCards(hand: Card[], cards: Card[]): Card[] {
  const counts = new Map<CardId, number>();
  for (const card of cards) {
    const id = cardId(card);
    counts.set(id, (counts.get(id) ?? 0) + 1);
  }

  const result: Card[] = [];
  for (const card of hand) {
    const id = cardId(card);
    const count = counts.get(id) ?? 0;
    if (count > 0) {
      counts.set(id, count - 1);
    } else {
      result.push(card);
    }
  }

  const missing = [...counts.entries()].filter(([, count]) => count > 0);
  if (missing.length > 0) {
    throw new Error(`Cannot remove cards not in hand: ${missing.map(([id]) => id).join(", ")}`);
  }

  return sortCards(result);
}

export function cardLabel(card: Card): string {
  return `${card.rank}${card.suit}`;
}

export function handKey(cards: Card[]): string {
  return sortCards(cards).map(cardId).join("-");
}
