import { describe, expect, it } from "vitest";
import { cardFromId, cardId } from "./cards";
import { beats, classifyCombo, enumerateCombos, moveKey, playMove } from "./combinations";
import { applyMove, createGame, isTerminal, legalMoves } from "./game";

function cards(ids: string[]) {
  return ids.map(cardFromId);
}

describe("Pusoy Dos combination rules", () => {
  it("orders singles by rank then suit", () => {
    const threeDiamonds = classifyCombo(cards(["3D"]))!;
    const threeClubs = classifyCombo(cards(["3C"]))!;
    const twoClubs = classifyCombo(cards(["2C"]))!;

    expect(beats(threeDiamonds, threeClubs)).toBe(true);
    expect(beats(twoClubs, threeDiamonds)).toBe(true);
  });

  it("classifies and ranks five-card hands", () => {
    const straight = classifyCombo(cards(["3C", "4S", "5H", "6D", "7C"]))!;
    const flush = classifyCombo(cards(["3D", "5D", "8D", "10D", "KD"]))!;
    const fullHouse = classifyCombo(cards(["9C", "9S", "9H", "4C", "4D"]))!;
    const fourKind = classifyCombo(cards(["JC", "JS", "JH", "JD", "5C"]))!;
    const straightFlush = classifyCombo(cards(["8H", "9H", "10H", "JH", "QH"]))!;

    expect(straight.kind).toBe("straight");
    expect(flush.kind).toBe("flush");
    expect(fullHouse.kind).toBe("full-house");
    expect(fourKind.kind).toBe("four-kind");
    expect(straightFlush.kind).toBe("straight-flush");
    expect(beats(flush, straight)).toBe(true);
    expect(beats(fullHouse, flush)).toBe(true);
    expect(beats(fourKind, fullHouse)).toBe(true);
    expect(beats(straightFlush, fourKind)).toBe(true);
  });

  it("enumerates all basic singles for a hand", () => {
    const hand = cards(["3C", "3S", "4C", "5D", "6H"]);
    const combos = enumerateCombos(hand);
    expect(combos.filter((combo) => combo.kind === "single")).toHaveLength(5);
    expect(combos.some((combo) => combo.kind === "pair")).toBe(true);
  });
});

describe("Pusoy Dos game flow", () => {
  it("requires the first move to include 3C", () => {
    const game = createGame("first-move-test");
    const firstPlayer = game.currentPlayer;
    expect(game.hands[firstPlayer].map(cardId)).toContain("3C");
    expect(legalMoves(game).every((move) => move.type === "play" && move.cards.some((card) => cardId(card) === "3C"))).toBe(
      true
    );
  });

  it("applies legal moves until a game reaches terminal state", () => {
    let game = createGame("terminal-test");
    let guard = 0;

    while (!isTerminal(game) && guard < 260) {
      const moves = legalMoves(game);
      const nonPass = moves.find((move) => move.type === "play");
      game = applyMove(game, nonPass ?? moves[0]);
      guard += 1;
    }

    expect(isTerminal(game)).toBe(true);
    expect(guard).toBeLessThan(260);
  });

  it("normalizes move keys independent of card order", () => {
    expect(moveKey(playMove(cards(["5D", "5C"])))).toBe(moveKey(playMove(cards(["5C", "5D"]))));
  });
});
