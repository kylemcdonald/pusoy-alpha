import {
  ArrowDown,
  ArrowUp,
  Bot,
  CornerUpLeft,
  RefreshCw,
  Settings,
  Sparkles,
  Undo2,
  User,
  X
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import {
  Card,
  GameState,
  Move,
  PASS_MOVE,
  SUIT_STRENGTH_ORDER,
  Suit,
  applyMove,
  cardId,
  cloneGameState,
  createGame,
  findMoveByCards,
  isTerminal,
  legalMoves,
  placements,
  rankValue
} from "./core";
import {
  HandcraftedModel,
  NeuralPolicyValueModel,
  PolicyValueModel,
  WasmSearchEngine,
  createWasmSearchEngine,
  searchMoveForObserver,
  type NeuralModelFile,
  type SearchResult
} from "./ai";
import { createRng } from "./core/random";

const HUMAN_PLAYER = 0;
const PLAYER_LABELS = ["You", "AI"];
const DEFAULT_SUIT_ORDER: Suit[] = [...SUIT_STRENGTH_ORDER];
const LEGACY_PUSOY_SUIT_ORDER: Suit[] = ["C", "S", "H", "D"];
const LEGACY_FRONTEND_SUIT_ORDER: Suit[] = ["S", "H", "C", "D"];
const SUIT_SYMBOLS: Record<Suit, string> = {
  C: "♣️",
  S: "♠️",
  H: "♥️",
  D: "♦️"
};
const SETTINGS_KEY = "pusoy-alpha-settings";
type ThinkingLevel = "short" | "medium" | "long";
const DEFAULT_THINKING_LEVEL: ThinkingLevel = "medium";
const THINKING_TIME_BUDGETS_MS: Record<ThinkingLevel, number> = {
  short: 100,
  medium: 500,
  long: 1000
};
const DECISION_SIMULATIONS_PER_DETERMINATION = 12;
const OUTCOME_ESTIMATE_TIME_BUDGET_MS = 1000;
const OUTCOME_TURN_TIME_BUDGET_MS = 30;
const OUTCOME_SIMULATIONS_PER_DETERMINATION = 2;

interface AppSettings {
  alwaysShowHints: boolean;
  allowWraparoundStraights: boolean;
  debug: boolean;
  showOutcome: boolean;
  thinkingLevel: ThinkingLevel;
  suitOrder: Suit[];
}

interface SearchBudget {
  timeLimitMs: number;
  simulationsPerDetermination: number;
}

interface OutcomePrediction {
  winner: number | null;
  simulatedTurns: number;
  determinizations: number;
  simulations: number;
}

interface DebugStats {
  label: string;
  determinizations: number;
  simulations: number;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function isSuit(value: string): value is Suit {
  return DEFAULT_SUIT_ORDER.includes(value as Suit);
}

function isThinkingLevel(value: unknown): value is ThinkingLevel {
  return value === "short" || value === "medium" || value === "long";
}

function migrateThinkingLevel(value: { thinkingLevel?: unknown; searchSimulations?: unknown } | null | undefined): ThinkingLevel {
  if (isThinkingLevel(value?.thinkingLevel)) {
    return value.thinkingLevel;
  }

  const simulations =
    typeof value?.searchSimulations === "number"
      ? value.searchSimulations
      : typeof value?.searchSimulations === "string"
        ? Number(value.searchSimulations)
        : Number.NaN;
  if (Number.isFinite(simulations)) {
    if (simulations <= 100) {
      return "short";
    }
    if (simulations >= 400) {
      return "long";
    }
  }

  return DEFAULT_THINKING_LEVEL;
}

function sameSuitOrder(left: Suit[], right: Suit[]): boolean {
  return left.length === right.length && left.every((suit, index) => suit === right[index]);
}

function normalizeSettings(
  value:
    | (Partial<AppSettings> & {
        aiSuggestion?: boolean;
        searchDeterminizations?: unknown;
        searchSimulations?: unknown;
        thinkingLevel?: unknown;
      })
    | null
    | undefined
): AppSettings {
  const suitOrder = Array.isArray(value?.suitOrder)
    ? value.suitOrder.filter((suit): suit is Suit => typeof suit === "string" && isSuit(suit))
    : [];
  const uniqueSuitOrder = [...new Set(suitOrder)];

  return {
    alwaysShowHints: Boolean(value?.alwaysShowHints ?? value?.aiSuggestion),
    allowWraparoundStraights: Boolean(value?.allowWraparoundStraights),
    debug: Boolean(value?.debug),
    showOutcome: Boolean(value?.showOutcome),
    thinkingLevel: migrateThinkingLevel(value),
    suitOrder:
      uniqueSuitOrder.length === DEFAULT_SUIT_ORDER.length
        ? uniqueSuitOrder
        : [...DEFAULT_SUIT_ORDER]
  };
}

function loadSettings(): AppSettings {
  if (typeof window === "undefined") {
    return normalizeSettings(null);
  }

  try {
    const settings = normalizeSettings(JSON.parse(window.localStorage.getItem(SETTINGS_KEY) ?? "null"));
    return sameSuitOrder(settings.suitOrder, LEGACY_PUSOY_SUIT_ORDER) ||
      sameSuitOrder(settings.suitOrder, LEGACY_FRONTEND_SUIT_ORDER)
      ? { ...settings, suitOrder: [...DEFAULT_SUIT_ORDER] }
      : settings;
  } catch {
    return normalizeSettings(null);
  }
}

function suitOrderValue(suitOrder: Suit[], suit: Suit): number {
  const index = suitOrder.indexOf(suit);
  return index >= 0 ? index : DEFAULT_SUIT_ORDER.indexOf(suit);
}

function compareCardsForDisplay(suitOrder: Suit[], a: Card, b: Card): number {
  const rankDiff = rankValue(a.rank) - rankValue(b.rank);
  if (rankDiff !== 0) {
    return rankDiff;
  }
  return suitOrderValue(suitOrder, a.suit) - suitOrderValue(suitOrder, b.suit);
}

function sortCardsForDisplay(cards: Card[], suitOrder: Suit[]): Card[] {
  return [...cards].sort((a, b) => compareCardsForDisplay(suitOrder, a, b));
}

function cardDisplay(card: Card): string {
  return `${card.rank}${SUIT_SYMBOLS[card.suit]}`;
}

function formatCardsDisplay(cards: Card[], suitOrder: Suit[]): string {
  return sortCardsForDisplay(cards, suitOrder).map(cardDisplay).join(" ");
}

function formatMoveDisplay(move: Move, suitOrder: Suit[]): string {
  if (move.type === "pass") {
    return "Pass";
  }
  return formatCardsDisplay(move.cards, suitOrder);
}

function suitGlyph(card: Card): string {
  return SUIT_SYMBOLS[card.suit];
}

function isRedSuit(card: Card): boolean {
  return card.suit === "H" || card.suit === "D";
}

function CardButton({
  card,
  selected = false,
  suggested = false,
  disabled = false,
  onClick,
  compact = false
}: {
  card: Card;
  selected?: boolean;
  suggested?: boolean;
  disabled?: boolean;
  onClick?: () => void;
  compact?: boolean;
}) {
  return (
    <button
      className={`playing-card ${selected ? "selected" : ""} ${suggested ? "suggested" : ""} ${isRedSuit(card) ? "red" : "black"} ${
        compact ? "compact" : ""
      }`}
      disabled={disabled}
      onClick={onClick}
      title={cardDisplay(card)}
      type="button"
    >
      <span className="rank">{card.rank}</span>
      <span className="suit">{suitGlyph(card)}</span>
    </button>
  );
}

function HandView({
  hand,
  selectedIds,
  suggestedIds,
  onToggle,
  onDoubleTap,
  reveal = true,
  compact = false
}: {
  hand: Card[];
  selectedIds?: Set<string>;
  suggestedIds?: Set<string>;
  onToggle?: (card: Card) => void;
  onDoubleTap?: () => void;
  reveal?: boolean;
  compact?: boolean;
}) {
  const lastTapRef = useRef(0);
  const handClassName = `hand ${onDoubleTap ? "revealable" : ""}`;
  const handEvents = onDoubleTap
    ? {
        onDoubleClick: () => onDoubleTap(),
        onPointerUp: (event: PointerEvent<HTMLDivElement>) => {
          if (event.pointerType === "mouse") {
            return;
          }

          const now = window.performance.now();
          if (now - lastTapRef.current <= 360) {
            lastTapRef.current = 0;
            event.preventDefault();
            onDoubleTap();
            return;
          }

          lastTapRef.current = now;
        }
      }
    : {};
  const displayedHand = sortCardsForDisplay(hand, DEFAULT_SUIT_ORDER);
  if (!reveal) {
    return (
      <div className={handClassName} {...handEvents}>
        {displayedHand.map((card) => (
          <div className={`card-back ${compact ? "compact" : ""}`} key={cardId(card)} />
        ))}
      </div>
    );
  }

  return (
    <div className={handClassName} {...handEvents}>
      {hand.map((card) => (
        <CardButton
          card={card}
          compact={compact}
          key={cardId(card)}
          onClick={onToggle ? () => onToggle(card) : undefined}
          selected={selectedIds?.has(cardId(card))}
          suggested={suggestedIds?.has(cardId(card))}
        />
      ))}
    </div>
  );
}

function PlayerStrip({
  player,
  state,
  reveal,
  selectedIds,
  suggestedIds,
  suitOrder,
  onToggle,
  onHandDoubleTap
}: {
  player: number;
  state: GameState;
  reveal: boolean;
  selectedIds?: Set<string>;
  suggestedIds?: Set<string>;
  suitOrder: Suit[];
  onToggle?: (card: Card) => void;
  onHandDoubleTap?: () => void;
}) {
  const finished = state.finished.indexOf(player);
  const displayedHand = sortCardsForDisplay(state.hands[player], suitOrder);
  return (
    <section className={`player-strip ${state.currentPlayer === player ? "active" : ""}`}>
      <div className="player-meta">
        <div className="seat-icon">{player === HUMAN_PLAYER ? <User size={16} /> : <Bot size={16} />}</div>
        <div>
          <div className="player-name">{PLAYER_LABELS[player]}</div>
          <div className="player-count">
            {state.hands[player].length} cards
            {finished >= 0 ? ` · #${finished + 1}` : ""}
          </div>
        </div>
      </div>
      <HandView
        compact={player !== HUMAN_PLAYER}
        hand={displayedHand}
        onDoubleTap={onHandDoubleTap}
        onToggle={onToggle}
        reveal={reveal}
        selectedIds={selectedIds}
        suggestedIds={suggestedIds}
      />
    </section>
  );
}

function MoveTray({
  canPlay,
  canPass,
  busy,
  passSuggested,
  hintButton,
  canShowHint,
  canAutoplay,
  canGoBack,
  canShowOutcomeOnce,
  gameOver,
  outcomeOnceActive,
  onShowHint,
  onShowOutcomeOnce,
  onPlay,
  onPass,
  onAutoplay,
  onBack,
  onNewGame
}: {
  canPlay: boolean;
  canPass: boolean;
  busy: boolean;
  passSuggested: boolean;
  hintButton: "show-hint" | "autoplay";
  canShowHint: boolean;
  canAutoplay: boolean;
  canGoBack: boolean;
  canShowOutcomeOnce: boolean;
  gameOver: boolean;
  outcomeOnceActive: boolean;
  onShowHint: () => void;
  onShowOutcomeOnce: () => void;
  onPlay: () => void;
  onPass: () => void;
  onAutoplay: () => void;
  onBack: () => void;
  onNewGame: () => void;
}) {
  if (gameOver) {
    return (
      <div className="move-tray">
        <div className="button-row">
          <button className="icon-button" disabled={busy || !canGoBack} onClick={onBack} title="Back one turn" type="button">
            <Undo2 size={18} />
          </button>
          <button className="command-button primary" onClick={onNewGame} type="button">
            New Game
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="move-tray">
      <div className="button-row">
        <button className="icon-button" disabled={busy || !canGoBack} onClick={onBack} title="Back one turn" type="button">
          <Undo2 size={18} />
        </button>
        {canShowOutcomeOnce ? (
          <button
            aria-label="Show predicted outcome this turn"
            className="icon-button monochrome-icon"
            disabled={busy || outcomeOnceActive}
            onClick={onShowOutcomeOnce}
            title="Show predicted outcome this turn"
            type="button"
          >
            <Sparkles size={18} strokeWidth={2.5} />
          </button>
        ) : null}
        <button
          className={`command-button ${passSuggested ? "suggested-pass" : ""}`}
          disabled={busy || !canPass}
          onClick={onPass}
          type="button"
        >
          Pass
        </button>
        <button className="command-button primary" disabled={busy || !canPlay} onClick={onPlay} type="button">
          Play
        </button>
        {hintButton === "show-hint" ? (
          <button
            className="command-button hint-action"
            disabled={busy || !canShowHint}
            onClick={onShowHint}
            type="button"
          >
            Hint
          </button>
        ) : null}
        {hintButton === "autoplay" ? (
          <button
            className="command-button autoplay hint-action"
            disabled={busy || !canAutoplay}
            onClick={onAutoplay}
            type="button"
          >
            Autoplay
          </button>
        ) : null}
      </div>
    </div>
  );
}

function Timeline({
  busy,
  currentIndex,
  onJump,
  suitOrder,
  timeline
}: {
  busy: boolean;
  currentIndex: number;
  onJump: (index: number) => void;
  suitOrder: Suit[];
  timeline: GameState[];
}) {
  const entries = timeline.slice(1).map((state, index) => ({
    entry: state.history[state.history.length - 1],
    stateIndex: index + 1
  }));

  return (
    <ol className="timeline">
      {entries.map(({ entry, stateIndex }) => (
        <li
          className={`${stateIndex === currentIndex ? "current" : ""} ${stateIndex > currentIndex ? "future" : ""}`}
          key={`${stateIndex}-${entry.turn}-${entry.player}`}
        >
          <span>{PLAYER_LABELS[entry.player]}</span>
          <strong>{formatMoveDisplay(entry.move, suitOrder)}</strong>
          {entry.player !== HUMAN_PLAYER ? (
            <button
              className="icon-button timeline-jump"
              disabled={busy || stateIndex === currentIndex}
              onClick={() => onJump(stateIndex)}
              title={stateIndex === currentIndex ? "Current state" : `Reset to turn ${entry.turn}`}
              type="button"
            >
              <CornerUpLeft size={15} />
            </button>
          ) : (
            <span aria-hidden="true" className="timeline-jump-spacer" />
          )}
        </li>
      ))}
    </ol>
  );
}

function turnText(state: GameState, busy: boolean): string {
  if (isTerminal(state)) {
    return placements(state)[0] === HUMAN_PLAYER ? "You win" : "You lose";
  }
  if (busy) {
    return "Thinking";
  }
  return state.currentPlayer === HUMAN_PLAYER ? "Your turn" : "AI's turn";
}

function createHumanGame(allowWraparoundStraights: boolean): GameState {
  return createGame(`human-${Date.now()}`, { allowWraparoundStraights });
}

function outcomeTurnKey(state: GameState): string {
  return `${state.id}:${state.history.length}:${state.currentPlayer}`;
}

function searchUiMove(
  state: GameState,
  player: number,
  model: PolicyValueModel,
  searchBudget: SearchBudget,
  wasmSearch: WasmSearchEngine | null
): SearchResult {
  const seed =
    player === HUMAN_PLAYER
      ? `${state.id}:hint:${state.history.length}`
      : `${state.id}:${state.history.length}:${player}`;
  const wasmResult = wasmSearch?.search(state, player, seed, searchBudget);
  if (wasmResult) {
    return wasmResult;
  }

  return searchMoveForObserver(state, player, model, {
    simulationsPerDetermination: searchBudget.simulationsPerDetermination,
    timeLimitMs: searchBudget.timeLimitMs,
    rng: createRng(seed)
  });
}

async function simulateLikelyOutcome(
  startState: GameState,
  model: PolicyValueModel,
  wasmSearch: WasmSearchEngine | null,
  shouldCancel: () => boolean
): Promise<OutcomePrediction | null> {
  let projected = cloneGameState(startState);
  let simulatedTurns = 0;
  let determinizations = 0;
  let simulations = 0;
  const deadline = window.performance.now() + OUTCOME_ESTIMATE_TIME_BUDGET_MS;

  while (!isTerminal(projected) && simulatedTurns < 80) {
    await wait(0);
    if (shouldCancel()) {
      return null;
    }

    const remainingMs = deadline - window.performance.now();
    if (remainingMs <= 0) {
      break;
    }

    const player = projected.currentPlayer;
    const result = searchUiMove(
      projected,
      player,
      model,
      {
        simulationsPerDetermination: OUTCOME_SIMULATIONS_PER_DETERMINATION,
        timeLimitMs: Math.min(OUTCOME_TURN_TIME_BUDGET_MS, remainingMs)
      },
      wasmSearch
    );
    determinizations += result.determinizations;
    simulations += result.simulations;
    projected = applyMove(projected, result.move);
    simulatedTurns += 1;
  }

  const winner = placements(projected)[0] ?? null;
  return { winner, simulatedTurns, determinizations, simulations };
}

function previousHumanTurnIndex(timeline: GameState[], currentIndex: number): number {
  for (let index = currentIndex - 1; index >= 0; index -= 1) {
    const state = timeline[index];
    if (!isTerminal(state) && state.currentPlayer === HUMAN_PLAYER) {
      return index;
    }
  }
  return Math.max(0, currentIndex - 1);
}

function OutcomeBadge({ enabled, prediction }: { enabled: boolean; prediction: OutcomePrediction | null }) {
  const hasWinner = prediction !== null && prediction.winner !== null;
  const winner = hasWinner ? PLAYER_LABELS[prediction.winner!] : "Calculating";

  return (
    <div aria-hidden={!enabled} className={`outcome-badge ${enabled ? "" : "empty"}`}>
      Likely winner: <strong>{winner}</strong>
    </div>
  );
}

function DebugStatus({ enabled, stats }: { enabled: boolean; stats: DebugStats | null }) {
  if (!enabled) {
    return <div aria-hidden="true" className="debug-status empty" />;
  }

  return (
    <div className="debug-status">
      {stats
        ? `${stats.label}: ${stats.determinizations} determinizations, ${stats.simulations} simulations`
        : "No thinking stats yet"}
    </div>
  );
}

function PlayAgainstAi({
  alwaysShowHints,
  allowWraparoundStraights,
  debug,
  model,
  onOpenSettings,
  showOutcome,
  thinkingLevel,
  suitOrder,
  wasmSearch
}: {
  alwaysShowHints: boolean;
  allowWraparoundStraights: boolean;
  debug: boolean;
  model: PolicyValueModel;
  onOpenSettings: () => void;
  showOutcome: boolean;
  thinkingLevel: ThinkingLevel;
  suitOrder: Suit[];
  wasmSearch: WasmSearchEngine | null;
}) {
  const [timeline, setTimeline] = useState<GameState[]>(() => [createHumanGame(allowWraparoundStraights)]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [singleHint, setSingleHint] = useState(false);
  const [outcomeOnceKey, setOutcomeOnceKey] = useState<string | null>(null);
  const [suggestion, setSuggestion] = useState<SearchResult | null>(null);
  const [outcomePrediction, setOutcomePrediction] = useState<OutcomePrediction | null>(null);
  const [debugStats, setDebugStats] = useState<DebugStats | null>(null);
  const [opponentCardsRevealed, setOpponentCardsRevealed] = useState(false);
  const [busy, setBusy] = useState(false);
  const timelineRef = useRef(timeline);
  const currentIndexRef = useRef(currentIndex);
  const busyRef = useRef(false);
  const game = (timeline[currentIndex] ?? timeline[timeline.length - 1])!;
  const hintsActive = alwaysShowHints || singleHint;
  const currentOutcomeTurnKey = outcomeTurnKey(game);
  const outcomeOnceActive = outcomeOnceKey === currentOutcomeTurnKey;
  const outcomeVisible = showOutcome || outcomeOnceActive;
  const canGoBack = currentIndex > 0;
  const opponentCardsVisible = opponentCardsRevealed || isTerminal(game);
  const decisionSearchBudget = useMemo(
    () => ({
      simulationsPerDetermination: DECISION_SIMULATIONS_PER_DETERMINATION,
      timeLimitMs: THINKING_TIME_BUDGETS_MS[thinkingLevel]
    }),
    [thinkingLevel]
  );

  useEffect(() => {
    timelineRef.current = timeline;
  }, [timeline]);

  useEffect(() => {
    currentIndexRef.current = currentIndex;
  }, [currentIndex]);

  const setTimelinePosition = (nextTimeline: GameState[], nextIndex: number) => {
    timelineRef.current = nextTimeline;
    currentIndexRef.current = nextIndex;
    setTimeline(nextTimeline);
    setCurrentIndex(nextIndex);
  };

  const appendGameState = (next: GameState) => {
    const baseTimeline = timelineRef.current.slice(0, currentIndexRef.current + 1);
    const nextTimeline = [...baseTimeline, next];
    setTimelinePosition(nextTimeline, nextTimeline.length - 1);
  };

  const jumpToState = (index: number) => {
    if (busyRef.current || index < 0 || index >= timelineRef.current.length) {
      return;
    }
    setSelected(new Set());
    setSingleHint(false);
    setOutcomeOnceKey(null);
    setSuggestion(null);
    currentIndexRef.current = index;
    setCurrentIndex(index);
  };

  const backOneTurn = () => {
    jumpToState(previousHumanTurnIndex(timelineRef.current, currentIndexRef.current));
  };

  const selectedCards = useMemo(() => {
    return sortCardsForDisplay(game.hands[HUMAN_PLAYER].filter((card) => selected.has(cardId(card))), suitOrder);
  }, [game.hands, selected, suitOrder]);

  const moves = useMemo(() => legalMoves(game), [game]);
  const selectedMove = selectedCards.length > 0 ? findMoveByCards(game, selectedCards) : null;
  const canPass = moves.some((move) => move.type === "pass") && game.currentPlayer === HUMAN_PLAYER;
  const canPlay = Boolean(selectedMove) && game.currentPlayer === HUMAN_PLAYER;
  const suggestedIds = useMemo(() => {
    if (!hintsActive || !suggestion || suggestion.move.type !== "play" || game.currentPlayer !== HUMAN_PLAYER) {
      return undefined;
    }
    return new Set(suggestion.move.cards.map(cardId));
  }, [game.currentPlayer, hintsActive, suggestion]);
  const passSuggested =
    hintsActive &&
    Boolean(suggestion) &&
    suggestion?.move.type === "pass" &&
    game.currentPlayer === HUMAN_PLAYER &&
    !isTerminal(game);
  const canShowHint = !alwaysShowHints && game.currentPlayer === HUMAN_PLAYER && !busy && !isTerminal(game);
  const canAutoplay =
    hintsActive && Boolean(suggestion) && game.currentPlayer === HUMAN_PLAYER && !busy && !isTerminal(game);
  const hintButton = alwaysShowHints || singleHint ? "autoplay" : "show-hint";
  const canShowOutcomeOnce = !showOutcome && !isTerminal(game);

  const toggleSelected = (card: Card) => {
    if (game.currentPlayer !== HUMAN_PLAYER || busy) {
      return;
    }
    setSelected((current) => {
      const next = new Set(current);
      const id = cardId(card);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const runAiTurns = async (startState: GameState) => {
    if (busyRef.current) {
      return;
    }
    busyRef.current = true;
    setBusy(true);

    let next = startState;
    let guard = 0;
    while (!isTerminal(next) && next.currentPlayer !== HUMAN_PLAYER && guard < 80) {
      await wait(180);
      const player = next.currentPlayer;
      const result = searchUiMove(next, player, model, decisionSearchBudget, wasmSearch);
      setDebugStats({
        label: "AI",
        determinizations: result.determinizations,
        simulations: result.simulations
      });
      next = applyMove(next, result.move);
      appendGameState(next);
      guard += 1;
    }

    busyRef.current = false;
    setBusy(false);
  };

  useEffect(() => {
    if (!isTerminal(game) && game.currentPlayer !== HUMAN_PLAYER) {
      void runAiTurns(game);
    }
  }, [game, model, decisionSearchBudget, wasmSearch]);

  useEffect(() => {
    if (!hintsActive || busy || isTerminal(game) || game.currentPlayer !== HUMAN_PLAYER) {
      setSuggestion(null);
      return;
    }

    let cancelled = false;
    setSuggestion(null);
    const timeout = window.setTimeout(() => {
      const result = searchUiMove(game, HUMAN_PLAYER, model, decisionSearchBudget, wasmSearch);
      if (!cancelled) {
        setSuggestion(result);
        setDebugStats({
          label: "Suggestion",
          determinizations: result.determinizations,
          simulations: result.simulations
        });
      }
    }, 20);

    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [busy, game, hintsActive, model, decisionSearchBudget, wasmSearch]);

  useEffect(() => {
    if (outcomeOnceKey !== null && outcomeOnceKey !== currentOutcomeTurnKey) {
      setOutcomeOnceKey(null);
    }
  }, [currentOutcomeTurnKey, outcomeOnceKey]);

  useEffect(() => {
    if (!outcomeVisible) {
      setOutcomePrediction(null);
      return;
    }

    if (busy) {
      return;
    }

    let cancelled = false;

    if (isTerminal(game)) {
      setOutcomePrediction({
        winner: placements(game)[0] ?? null,
        simulatedTurns: 0,
        determinizations: 0,
        simulations: 0
      });
      return () => {
        cancelled = true;
      };
    }

    const timeout = window.setTimeout(() => {
      void simulateLikelyOutcome(game, model, wasmSearch, () => cancelled).then((prediction) => {
        if (!cancelled && prediction) {
          setOutcomePrediction(prediction);
          setDebugStats({
            label: "Outcome",
            determinizations: prediction.determinizations,
            simulations: prediction.simulations
          });
        }
      });
    }, 120);

    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [busy, game, model, outcomeVisible, wasmSearch]);

  const applyHumanMove = (move: Move) => {
    if (game.currentPlayer !== HUMAN_PLAYER || busy) {
      return;
    }
    const next = applyMove(game, move);
    setSelected(new Set());
    setSingleHint(false);
    setOutcomeOnceKey(null);
    appendGameState(next);
  };

  const reset = () => {
    const next = createHumanGame(allowWraparoundStraights);
    setSelected(new Set());
    setSingleHint(false);
    setOutcomeOnceKey(null);
    setSuggestion(null);
    setDebugStats(null);
    setOpponentCardsRevealed(false);
    setTimelinePosition([next], 0);
  };

  useEffect(() => {
    if (game.rules?.allowWraparoundStraights !== allowWraparoundStraights) {
      reset();
    }
  }, [allowWraparoundStraights, game.rules?.allowWraparoundStraights]);

  const showOutcomeOnce = () => {
    if (!canShowOutcomeOnce || busy) {
      return;
    }
    setOutcomePrediction(null);
    setOutcomeOnceKey(currentOutcomeTurnKey);
  };

  const requestHint = () => {
    if (!canShowHint) {
      return;
    }
    setSuggestion(null);
    setSingleHint(true);
  };

  const autoplaySuggestion = () => {
    if (!suggestion || game.currentPlayer !== HUMAN_PLAYER || busy || isTerminal(game)) {
      return;
    }
    applyHumanMove(suggestion.move);
  };

  return (
    <main className="game-layout">
      <div className="table-surface">
        <div className="opponent-grid">
          {[1].map((player) => (
            <PlayerStrip
              key={player}
              onHandDoubleTap={() => setOpponentCardsRevealed(true)}
              player={player}
              reveal={opponentCardsVisible}
              state={game}
              suitOrder={suitOrder}
            />
          ))}
        </div>

        <div className="center-table">
          <div className="turn-badge">
            {turnText(game, busy)}
          </div>
          <OutcomeBadge enabled={outcomeVisible && !isTerminal(game)} prediction={outcomePrediction} />
          <DebugStatus enabled={debug} stats={debugStats} />
          <div className="active-combo">
            {game.activeCombo ? (
              <HandView hand={sortCardsForDisplay(game.activeCombo.cards, suitOrder)} reveal />
            ) : (
              <div className="lead-marker">Lead</div>
            )}
          </div>
        </div>

        <PlayerStrip
          onToggle={toggleSelected}
          player={HUMAN_PLAYER}
          reveal
          selectedIds={selected}
          suggestedIds={suggestedIds}
          state={game}
          suitOrder={suitOrder}
        />

        <MoveTray
          busy={busy}
          canAutoplay={canAutoplay}
          canGoBack={canGoBack}
          canPass={canPass}
          canPlay={canPlay}
          canShowHint={canShowHint}
          canShowOutcomeOnce={canShowOutcomeOnce}
          gameOver={isTerminal(game)}
          hintButton={hintButton}
          outcomeOnceActive={outcomeOnceActive}
          onAutoplay={autoplaySuggestion}
          onBack={backOneTurn}
          onNewGame={reset}
          onPass={() => applyHumanMove(PASS_MOVE)}
          onPlay={() => selectedMove && selectedMove.type === "play" && applyHumanMove(selectedMove)}
          onShowHint={requestHint}
          onShowOutcomeOnce={showOutcomeOnce}
          passSuggested={passSuggested}
        />
      </div>

      <aside className="side-rail">
        <div className="rail-actions">
          <button className="icon-button" disabled={busy} onClick={reset} title="New deal" type="button">
            <RefreshCw size={18} />
          </button>
          <button className="icon-button" onClick={onOpenSettings} title="Settings" type="button">
            <Settings size={18} />
          </button>
        </div>
        <Timeline
          busy={busy}
          currentIndex={currentIndex}
          onJump={jumpToState}
          suitOrder={suitOrder}
          timeline={timeline}
        />
      </aside>
    </main>
  );
}

function SettingsModal({
  settings,
  onChange,
  onClose
}: {
  settings: AppSettings;
  onChange: (settings: AppSettings) => void;
  onClose: () => void;
}) {
  const displayedSuitOrder = [...settings.suitOrder].reverse();

  const moveSuit = (from: number, to: number) => {
    if (to < 0 || to >= displayedSuitOrder.length) {
      return;
    }
    const nextDisplayedOrder = [...displayedSuitOrder];
    const [suit] = nextDisplayedOrder.splice(from, 1);
    nextDisplayedOrder.splice(to, 0, suit);
    onChange({ ...settings, suitOrder: nextDisplayedOrder.reverse() });
  };

  return (
    <div className="modal-backdrop" role="presentation">
      <section aria-label="Settings" aria-modal="true" className="settings-modal" role="dialog">
        <header className="modal-header">
          <h2>Settings</h2>
          <button className="icon-button" onClick={onClose} title="Close settings" type="button">
            <X size={18} />
          </button>
        </header>

        <div className="settings-section">
          <label className="toggle-row">
            <input
              checked={settings.alwaysShowHints}
              onChange={(event) => onChange({ ...settings, alwaysShowHints: event.currentTarget.checked })}
              type="checkbox"
            />
            <span>Always show hints</span>
          </label>
          <label className="toggle-row">
            <input
              checked={settings.showOutcome}
              onChange={(event) => onChange({ ...settings, showOutcome: event.currentTarget.checked })}
              type="checkbox"
            />
            <span>Show outcome</span>
          </label>
          <label className="toggle-row">
            <input
              checked={settings.allowWraparoundStraights}
              onChange={(event) => onChange({ ...settings, allowWraparoundStraights: event.currentTarget.checked })}
              type="checkbox"
            />
            <span>Allow wraparound straights</span>
          </label>
          <label className="toggle-row">
            <input
              checked={settings.debug}
              onChange={(event) => onChange({ ...settings, debug: event.currentTarget.checked })}
              type="checkbox"
            />
            <span>Debug</span>
          </label>
          <label className="setting-row">
            <span>Thinking</span>
            <select
              onChange={(event) => {
                const nextLevel = event.currentTarget.value;
                onChange({
                  ...settings,
                  thinkingLevel: isThinkingLevel(nextLevel) ? nextLevel : DEFAULT_THINKING_LEVEL
                });
              }}
              value={settings.thinkingLevel}
            >
              <option value="short">Short - 0.1s</option>
              <option value="medium">Medium - 0.5s</option>
              <option value="long">Long - 1s</option>
            </select>
          </label>
        </div>

        <div className="settings-section">
          <h3>Suit Order (High to Low)</h3>
          <div className="suit-order-list">
            {displayedSuitOrder.map((suit, index) => (
              <div className="suit-order-row" key={suit}>
                <span>{SUIT_SYMBOLS[suit]}</span>
                <strong>{suit}</strong>
                <button
                  className="icon-button"
                  disabled={index === 0}
                  onClick={() => moveSuit(index, index - 1)}
                  title={`Move ${suit} higher`}
                  type="button"
                >
                  <ArrowUp size={17} />
                </button>
                <button
                  className="icon-button"
                  disabled={index === displayedSuitOrder.length - 1}
                  onClick={() => moveSuit(index, index + 1)}
                  title={`Move ${suit} lower`}
                  type="button"
                >
                  <ArrowDown size={17} />
                </button>
              </div>
            ))}
          </div>
        </div>

        <footer className="modal-footer">
          <button className="command-button" onClick={() => onChange(normalizeSettings(null))} type="button">
            Reset
          </button>
          <button className="command-button primary" onClick={onClose} type="button">
            Done
          </button>
        </footer>
      </section>
    </div>
  );
}

export function App() {
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [model, setModel] = useState<PolicyValueModel>(() => new HandcraftedModel());
  const [wasmSearch, setWasmSearch] = useState<WasmSearchEngine | null>(null);

  useEffect(() => {
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }, [settings]);

  useEffect(() => {
    let cancelled = false;
    fetch(`${import.meta.env.BASE_URL}models/neural-policy.json`)
      .then((response) => (response.ok ? (response.json() as Promise<NeuralModelFile>) : null))
      .then(async (neural) => {
        if (!cancelled) {
          setModel(neural ? new NeuralPolicyValueModel(neural) : new HandcraftedModel());
          setWasmSearch(null);
        }
        if (!neural) {
          return;
        }

        try {
          const engine = await createWasmSearchEngine(neural);
          if (!cancelled) {
            setWasmSearch(engine);
          }
        } catch {
          if (!cancelled) {
            setWasmSearch(null);
          }
        }
      })
      .catch(() => {
        if (!cancelled) {
          setModel(new HandcraftedModel());
          setWasmSearch(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="app-shell">
      <PlayAgainstAi
        alwaysShowHints={settings.alwaysShowHints}
        allowWraparoundStraights={settings.allowWraparoundStraights}
        debug={settings.debug}
        model={model}
        onOpenSettings={() => setSettingsOpen(true)}
        showOutcome={settings.showOutcome}
        thinkingLevel={settings.thinkingLevel}
        suitOrder={settings.suitOrder}
        wasmSearch={wasmSearch}
      />
      {settingsOpen ? (
        <SettingsModal
          onChange={(nextSettings) => setSettings(normalizeSettings(nextSettings))}
          onClose={() => setSettingsOpen(false)}
          settings={settings}
        />
      ) : null}
    </div>
  );
}
