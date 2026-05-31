import {
  ArrowDown,
  ArrowUp,
  Bot,
  RefreshCw,
  RotateCcw,
  Settings,
  Trophy,
  User,
  X
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Card,
  GameState,
  Move,
  PASS_MOVE,
  Suit,
  applyMove,
  cardId,
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
  searchMoveForObserver,
  type NeuralModelFile,
  type SearchResult
} from "./ai";
import { createRng } from "./core/random";

const HUMAN_PLAYER = 0;
const PLAYER_LABELS = ["You", "AI"];
const DEFAULT_SUIT_ORDER: Suit[] = ["S", "H", "C", "D"];
const LEGACY_DEFAULT_SUIT_ORDER: Suit[] = ["C", "S", "H", "D"];
const SUIT_SYMBOLS: Record<Suit, string> = {
  C: "♣️",
  S: "♠️",
  H: "♥️",
  D: "♦️"
};
const SETTINGS_KEY = "pusoy-alpha-settings";

interface AppSettings {
  alwaysShowHints: boolean;
  suitOrder: Suit[];
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function isSuit(value: string): value is Suit {
  return DEFAULT_SUIT_ORDER.includes(value as Suit);
}

function sameSuitOrder(left: Suit[], right: Suit[]): boolean {
  return left.length === right.length && left.every((suit, index) => suit === right[index]);
}

function normalizeSettings(value: (Partial<AppSettings> & { aiSuggestion?: boolean }) | null | undefined): AppSettings {
  const suitOrder = Array.isArray(value?.suitOrder)
    ? value.suitOrder.filter((suit): suit is Suit => typeof suit === "string" && isSuit(suit))
    : [];
  const uniqueSuitOrder = [...new Set(suitOrder)];

  return {
    alwaysShowHints: Boolean(value?.alwaysShowHints ?? value?.aiSuggestion),
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
    return sameSuitOrder(settings.suitOrder, LEGACY_DEFAULT_SUIT_ORDER)
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
  reveal = true,
  compact = false
}: {
  hand: Card[];
  selectedIds?: Set<string>;
  suggestedIds?: Set<string>;
  onToggle?: (card: Card) => void;
  reveal?: boolean;
  compact?: boolean;
}) {
  const displayedHand = sortCardsForDisplay(hand, DEFAULT_SUIT_ORDER);
  if (!reveal) {
    return (
      <div className="hand">
        {displayedHand.map((card) => (
          <div className={`card-back ${compact ? "compact" : ""}`} key={cardId(card)} />
        ))}
      </div>
    );
  }

  return (
    <div className="hand">
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
  onToggle
}: {
  player: number;
  state: GameState;
  reveal: boolean;
  selectedIds?: Set<string>;
  suggestedIds?: Set<string>;
  suitOrder: Suit[];
  onToggle?: (card: Card) => void;
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
        onToggle={onToggle}
        reveal={reveal}
        selectedIds={selectedIds}
        suggestedIds={suggestedIds}
      />
    </section>
  );
}

function MoveTray({
  selectedCards,
  canPlay,
  canPass,
  busy,
  passSuggested,
  hintButton,
  canShowHint,
  canAutoplay,
  onShowHint,
  onPlay,
  onPass,
  onAutoplay,
  onClear
}: {
  selectedCards: Card[];
  canPlay: boolean;
  canPass: boolean;
  busy: boolean;
  passSuggested: boolean;
  hintButton: "show-hint" | "autoplay";
  canShowHint: boolean;
  canAutoplay: boolean;
  onShowHint: () => void;
  onPlay: () => void;
  onPass: () => void;
  onAutoplay: () => void;
  onClear: () => void;
}) {
  return (
    <div className="move-tray">
      <div className="button-row">
        <button className="icon-button" disabled={busy || selectedCards.length === 0} onClick={onClear} title="Clear" type="button">
          <RotateCcw size={18} />
        </button>
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
          <button className="command-button" disabled={busy || !canShowHint} onClick={onShowHint} type="button">
            Show hint
          </button>
        ) : null}
        {hintButton === "autoplay" ? (
          <button className="command-button autoplay" disabled={busy || !canAutoplay} onClick={onAutoplay} type="button">
            Autoplay
          </button>
        ) : null}
      </div>
    </div>
  );
}

function Timeline({ state, suitOrder }: { state: GameState; suitOrder: Suit[] }) {
  return (
    <ol className="timeline">
      {state.history.slice(-12).map((entry) => (
        <li key={`${entry.turn}-${entry.player}`}>
          <span>{PLAYER_LABELS[entry.player]}</span>
          <strong>{formatMoveDisplay(entry.move, suitOrder)}</strong>
        </li>
      ))}
    </ol>
  );
}

function Scoreboard({ state }: { state: GameState }) {
  const order = placements(state);
  return (
    <div className="scoreboard">
      {order.map((player, index) => (
        <div className="score-row" key={player}>
          <Trophy size={15} />
          <span>#{index + 1}</span>
          <strong>{PLAYER_LABELS[player]}</strong>
        </div>
      ))}
    </div>
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

function createHumanGame(): GameState {
  return createGame(`human-${Date.now()}`);
}

function PlayAgainstAi({
  alwaysShowHints,
  model,
  onOpenSettings,
  suitOrder
}: {
  alwaysShowHints: boolean;
  model: PolicyValueModel;
  onOpenSettings: () => void;
  suitOrder: Suit[];
}) {
  const [game, setGame] = useState<GameState>(() => createHumanGame());
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [singleHint, setSingleHint] = useState(false);
  const [suggestion, setSuggestion] = useState<SearchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const hintsActive = alwaysShowHints || singleHint;

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
      const result = searchMoveForObserver(next, player, model, {
        simulations: 90,
        determinizations: 3,
        rng: createRng(`${next.id}:${next.history.length}:${player}`)
      });
      next = applyMove(next, result.move);
      setGame(next);
      guard += 1;
    }

    busyRef.current = false;
    setBusy(false);
  };

  useEffect(() => {
    if (!isTerminal(game) && game.currentPlayer !== HUMAN_PLAYER) {
      void runAiTurns(game);
    }
  }, [game, model]);

  useEffect(() => {
    if (!hintsActive || busy || isTerminal(game) || game.currentPlayer !== HUMAN_PLAYER) {
      setSuggestion(null);
      return;
    }

    let cancelled = false;
    setSuggestion(null);
    const timeout = window.setTimeout(() => {
      const result = searchMoveForObserver(game, HUMAN_PLAYER, model, {
        simulations: 72,
        determinizations: 2,
        rng: createRng(`${game.id}:hint:${game.history.length}`)
      });
      if (!cancelled) {
        setSuggestion(result);
      }
    }, 20);

    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [busy, game, hintsActive, model]);

  const applyHumanMove = (move: Move) => {
    if (game.currentPlayer !== HUMAN_PLAYER || busy) {
      return;
    }
    const next = applyMove(game, move);
    setSelected(new Set());
    setSingleHint(false);
    setGame(next);
  };

  const reset = () => {
    setSelected(new Set());
    setSingleHint(false);
    setGame(createHumanGame());
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
            <PlayerStrip key={player} player={player} reveal={false} state={game} suitOrder={suitOrder} />
          ))}
        </div>

        <div className="center-table">
          <div className="turn-badge">
            {turnText(game, busy)}
          </div>
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
          busy={busy || isTerminal(game)}
          canAutoplay={canAutoplay}
          canPass={canPass}
          canPlay={canPlay}
          canShowHint={canShowHint}
          hintButton={hintButton}
          onAutoplay={autoplaySuggestion}
          onClear={() => setSelected(new Set())}
          onPass={() => applyHumanMove(PASS_MOVE)}
          onPlay={() => selectedMove && selectedMove.type === "play" && applyHumanMove(selectedMove)}
          onShowHint={requestHint}
          passSuggested={passSuggested}
          selectedCards={selectedCards}
        />
      </div>

      <aside className="side-rail">
        <div className="rail-actions">
          <button className="icon-button" onClick={reset} title="New deal" type="button">
            <RefreshCw size={18} />
          </button>
          <button className="icon-button" onClick={onOpenSettings} title="Settings" type="button">
            <Settings size={18} />
          </button>
        </div>
        <Scoreboard state={game} />
        <Timeline state={game} suitOrder={suitOrder} />
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
  const moveSuit = (from: number, to: number) => {
    if (to < 0 || to >= settings.suitOrder.length) {
      return;
    }
    const nextOrder = [...settings.suitOrder];
    const [suit] = nextOrder.splice(from, 1);
    nextOrder.splice(to, 0, suit);
    onChange({ ...settings, suitOrder: nextOrder });
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
        </div>

        <div className="settings-section">
          <h3>Suit Order</h3>
          <div className="suit-order-list">
            {settings.suitOrder.map((suit, index) => (
              <div className="suit-order-row" key={suit}>
                <span>{SUIT_SYMBOLS[suit]}</span>
                <strong>{suit}</strong>
                <button
                  className="icon-button"
                  disabled={index === 0}
                  onClick={() => moveSuit(index, index - 1)}
                  title={`Move ${suit} up`}
                  type="button"
                >
                  <ArrowUp size={17} />
                </button>
                <button
                  className="icon-button"
                  disabled={index === settings.suitOrder.length - 1}
                  onClick={() => moveSuit(index, index + 1)}
                  title={`Move ${suit} down`}
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

  useEffect(() => {
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }, [settings]);

  useEffect(() => {
    let cancelled = false;
    fetch("/models/neural-policy.json")
      .then((response) => (response.ok ? (response.json() as Promise<NeuralModelFile>) : null))
      .then((neural) => {
        if (!cancelled) {
          setModel(neural ? new NeuralPolicyValueModel(neural) : new HandcraftedModel());
        }
      })
      .catch(() => {
        if (!cancelled) {
          setModel(new HandcraftedModel());
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
        model={model}
        onOpenSettings={() => setSettingsOpen(true)}
        suitOrder={settings.suitOrder}
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
