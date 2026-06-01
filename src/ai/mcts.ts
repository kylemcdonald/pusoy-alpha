import { Move, moveKey } from "../core/combinations";
import { GameState, applyLegalMove, isTerminal, legalMoves, rewardForPlayer } from "../core/game";
import { determinizeForPlayer } from "../core/observation";
import { RandomSource, createRng } from "../core/random";
import { HandcraftedModel, PolicyValueModel } from "./model";

export interface SearchOptions {
  simulations?: number;
  timeLimitMs?: number;
  maxDepth?: number;
  exploration?: number;
  rng?: RandomSource;
  rootPlayer?: number;
}

export interface ObserverSearchOptions extends SearchOptions {
  determinizations?: number;
  revealOpponents?: boolean;
  simulationsPerDetermination?: number;
}

export interface SearchResult {
  move: Move;
  visits: Record<string, number>;
  value: number;
  determinizations: number;
  simulations: number;
  legalMoveCount: number;
}

interface SearchNode {
  state: GameState;
  prior: number;
  visits: number;
  valueSum: number;
  move: Move | null;
  children: SearchNode[] | null;
}

function nodeValue(node: SearchNode): number {
  return node.visits === 0 ? 0 : node.valueSum / node.visits;
}

function nowMs(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function makeNode(state: GameState, move: Move | null, prior = 1): SearchNode {
  return {
    state,
    prior,
    visits: 0,
    valueSum: 0,
    move,
    children: null
  };
}

function selectChild(node: SearchNode, rootPlayer: number, exploration: number): SearchNode {
  if (!node.children || node.children.length === 0) {
    throw new Error("Cannot select from an unexpanded node");
  }

  const parentVisits = Math.max(1, node.visits);
  const actor = node.state.currentPlayer;
  let best = node.children[0];
  let bestScore = Number.NEGATIVE_INFINITY;

  for (const child of node.children) {
    const valueForRoot = nodeValue(child);
    const q = actor === rootPlayer ? valueForRoot : -valueForRoot;
    const u = exploration * child.prior * Math.sqrt(parentVisits) / (1 + child.visits);
    const score = q + u;
    if (score > bestScore) {
      bestScore = score;
      best = child;
    }
  }

  return best;
}

function visit(
  node: SearchNode,
  model: PolicyValueModel,
  rootPlayer: number,
  depth: number,
  maxDepth: number,
  exploration: number
): number {
  node.visits += 1;

  if (isTerminal(node.state)) {
    const value = rewardForPlayer(node.state, rootPlayer);
    node.valueSum += value;
    return value;
  }

  const moves = legalMoves(node.state);
  if (depth >= maxDepth || moves.length === 0) {
    const value = model.evaluate(node.state, rootPlayer, moves).value;
    node.valueSum += value;
    return value;
  }

  if (!node.children) {
    const actor = node.state.currentPlayer;
    const policyEvaluation = model.evaluate(node.state, actor, moves);
    const value =
      actor === rootPlayer ? policyEvaluation.value : model.evaluate(node.state, rootPlayer, moves).value;
    node.children = moves.map((move) => {
      return makeNode(applyLegalMove(node.state, move), move, policyEvaluation.priors.get(moveKey(move)) ?? 0.001);
    });
    node.valueSum += value;
    return value;
  }

  const child = selectChild(node, rootPlayer, exploration);
  const value = visit(child, model, rootPlayer, depth + 1, maxDepth, exploration);
  node.valueSum += value;
  return value;
}

export function searchMove(
  state: GameState,
  model: PolicyValueModel = new HandcraftedModel(),
  options: SearchOptions = {}
): SearchResult {
  const moves = legalMoves(state);
  if (moves.length === 0) {
    throw new Error("Cannot search from terminal state");
  }
  if (moves.length === 1) {
    return {
      move: moves[0],
      visits: { [moveKey(moves[0])]: 1 },
      value: model.evaluate(state, options.rootPlayer ?? state.currentPlayer, moves).value,
      determinizations: 0,
      simulations: 0,
      legalMoveCount: 1
    };
  }

  const simulations = options.simulations ?? 96;
  const deadline = options.timeLimitMs === undefined ? null : nowMs() + Math.max(1, options.timeLimitMs);
  const maxDepth = options.maxDepth ?? 80;
  const exploration = options.exploration ?? 1.35;
  const rootPlayer = options.rootPlayer ?? state.currentPlayer;
  const root = makeNode(state, null);
  let simulationsRun = 0;

  while (deadline === null ? simulationsRun < simulations : simulationsRun === 0 || nowMs() < deadline) {
    visit(root, model, rootPlayer, 0, maxDepth, exploration);
    simulationsRun += 1;
  }

  const children = root.children ?? [];
  const best = children.reduce((winner, child) => {
    if (!winner) return child;
    if (child.visits !== winner.visits) return child.visits > winner.visits ? child : winner;
    return nodeValue(child) > nodeValue(winner) ? child : winner;
  }, null as SearchNode | null);

  if (!best || !best.move) {
    return {
      move: moves[0],
      visits: { [moveKey(moves[0])]: 1 },
      value: 0,
      determinizations: 0,
      simulations: simulationsRun,
      legalMoveCount: moves.length
    };
  }

  return {
    move: best.move,
    visits: Object.fromEntries(children.map((child) => [moveKey(child.move!), child.visits])),
    value: nodeValue(root),
    determinizations: 0,
    simulations: simulationsRun,
    legalMoveCount: moves.length
  };
}

export function searchMoveForObserver(
  state: GameState,
  observer: number,
  model: PolicyValueModel = new HandcraftedModel(),
  options: ObserverSearchOptions = {}
): SearchResult {
  if (options.revealOpponents) {
    return searchMove(state, model, { ...options, rootPlayer: observer });
  }

  const rootMoves = legalMoves(state);
  if (rootMoves.length <= 1) {
    return searchMove(state, model, { ...options, rootPlayer: observer });
  }

  const rng = options.rng ?? createRng(`${state.id}:${state.history.length}:${observer}`);
  if (options.timeLimitMs !== undefined) {
    const deadline = nowMs() + Math.max(1, options.timeLimitMs);
    const simulationsPerDetermination = Math.max(2, Math.floor(options.simulationsPerDetermination ?? 12));
    const visits = new Map<string, number>();
    let valueSum = 0;
    let determinizationsRun = 0;
    let simulationsRun = 0;

    while (determinizationsRun === 0 || nowMs() < deadline) {
      const determinized = determinizeForPlayer(state, observer, rng);
      const result = searchMove(determinized, model, {
        ...options,
        simulations: simulationsPerDetermination,
        timeLimitMs: undefined,
        rootPlayer: observer,
        rng
      });

      determinizationsRun += 1;
      simulationsRun += result.simulations;
      valueSum += result.value;
      for (const [key, count] of Object.entries(result.visits)) {
        visits.set(key, (visits.get(key) ?? 0) + count);
      }
    }

    const bestMove = rootMoves.reduce((winner, move) => {
      const winnerVisits = visits.get(moveKey(winner)) ?? 0;
      const moveVisits = visits.get(moveKey(move)) ?? 0;
      return moveVisits > winnerVisits ? move : winner;
    }, rootMoves[0]);

    return {
      move: bestMove,
      visits: Object.fromEntries(rootMoves.map((move) => [moveKey(move), visits.get(moveKey(move)) ?? 0])),
      value: valueSum / determinizationsRun,
      determinizations: determinizationsRun,
      simulations: simulationsRun,
      legalMoveCount: rootMoves.length
    };
  }

  const determinizations = Math.max(1, options.determinizations ?? 3);
  const simulationsPerDetermination = Math.max(1, Math.floor((options.simulations ?? 96) / determinizations));
  const visits = new Map<string, number>();
  let valueSum = 0;

  for (let index = 0; index < determinizations; index += 1) {
    const determinized = determinizeForPlayer(state, observer, rng);
    const result = searchMove(determinized, model, {
      ...options,
      simulations: simulationsPerDetermination,
      rootPlayer: observer,
      rng
    });

    valueSum += result.value;
    for (const [key, count] of Object.entries(result.visits)) {
      visits.set(key, (visits.get(key) ?? 0) + count);
    }
  }

  const bestMove = rootMoves.reduce((winner, move) => {
    const winnerVisits = visits.get(moveKey(winner)) ?? 0;
    const moveVisits = visits.get(moveKey(move)) ?? 0;
    return moveVisits > winnerVisits ? move : winner;
  }, rootMoves[0]);

  return {
    move: bestMove,
    visits: Object.fromEntries(rootMoves.map((move) => [moveKey(move), visits.get(moveKey(move)) ?? 0])),
    value: valueSum / determinizations,
    determinizations,
    simulations: simulationsPerDetermination * determinizations,
    legalMoveCount: rootMoves.length
  };
}
