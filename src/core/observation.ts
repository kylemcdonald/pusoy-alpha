import { Card, cardId, newDeck, sortCards } from "./cards";
import { GameState, cloneGameState, cardsPlayed } from "./game";
import { RandomSource, shuffle } from "./random";

function withoutCards(deck: Card[], excluded: Card[]): Card[] {
  const excludedIds = new Set(excluded.map(cardId));
  return deck.filter((card) => !excludedIds.has(cardId(card)));
}

export function determinizeForPlayer(state: GameState, observer: number, rng: RandomSource): GameState {
  const next = cloneGameState(state);
  const knownCards = [...next.hands[observer], ...cardsPlayed(next)];
  const unknownCards = shuffle(withoutCards(newDeck(), knownCards), rng);
  let cursor = 0;

  for (let player = 0; player < next.hands.length; player += 1) {
    if (player === observer) {
      continue;
    }

    const count = next.hands[player].length;
    next.hands[player] = sortCards(unknownCards.slice(cursor, cursor + count));
    cursor += count;
  }

  return next;
}
