use std::cell::RefCell;
use wasm_bindgen::prelude::*;

const LOWEST_CARD: u8 = 3;
const PASS_SENTINEL: i32 = -1;
const KIND_SINGLE: u8 = 0;
const KIND_PAIR: u8 = 1;
const KIND_TRIPLE: u8 = 2;
const KIND_STRAIGHT: u8 = 3;
const KIND_FLUSH: u8 = 4;
const KIND_FULL_HOUSE: u8 = 5;
const KIND_FOUR_KIND: u8 = 6;
const KIND_STRAIGHT_FLUSH: u8 = 7;
const STRAIGHT_PATTERNS: [[u8; 5]; 11] = [
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8],
    [5, 6, 7, 8, 9],
    [6, 7, 8, 9, 10],
    [7, 8, 9, 10, 11],
    [11, 12, 0, 1, 2],
    [12, 0, 1, 2, 3],
    [8, 9, 10, 11, 12],
];

thread_local! {
    static MODEL: RefCell<Option<Model>> = const { RefCell::new(None) };
}

#[derive(Clone)]
struct Layer {
    rows: usize,
    cols: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Clone)]
struct Model {
    state_dim: usize,
    move_dim: usize,
    state_fc: Layer,
    move_fc: Layer,
    policy_fc: Layer,
    policy_out: Layer,
    value_fc: Layer,
    value_out: Layer,
    handcrafted_prior_weight: f32,
    handcrafted_value_weight: f32,
    neural_policy_temperature: f32,
    rollout_value_weight: f32,
}

#[derive(Clone)]
struct Combo {
    kind: u8,
    cards: Vec<u8>,
    size: u8,
    tiebreak: Vec<f32>,
}

#[derive(Clone)]
struct MoveR {
    pass: bool,
    combo: Option<Combo>,
}

#[derive(Clone)]
struct Game {
    hands: [Vec<u8>; 2],
    current_player: i8,
    active_combo: Option<Combo>,
    last_player: i8,
    passes_since_play: u8,
    finished: [bool; 2],
    history_len: u16,
    played_by_player: [Vec<u8>; 2],
}

#[derive(Clone)]
struct Node {
    game: Game,
    prior: f32,
    visits: u32,
    value_sum: f32,
    mv: Option<MoveR>,
    children: Option<Vec<Node>>,
}

struct SearchOutcome {
    best_move: MoveR,
    visits: Vec<(MoveR, u32)>,
    value: f32,
    determinizations: u32,
    simulations: u32,
    legal_move_count: usize,
}

struct Rng {
    state: u32,
}

fn now_ms() -> f64 {
    js_sys::Date::now()
}

fn suit_strength(card: u8) -> u8 {
    match card % 4 {
        0 => 1,
        1 => 3,
        2 => 2,
        _ => 0,
    }
}

fn rank(card: u8) -> u8 {
    card / 4
}

fn card_power(card: u8) -> u8 {
    rank(card) * 4 + suit_strength(card)
}

fn sort_cards(cards: &mut [u8]) {
    cards.sort_by_key(|card| card_power(*card));
}

fn combo_kind_order(kind: u8) -> f32 {
    match kind {
        KIND_STRAIGHT => 0.0,
        KIND_FLUSH => 1.0,
        KIND_FULL_HOUSE => 2.0,
        KIND_FOUR_KIND => 3.0,
        KIND_STRAIGHT_FLUSH => 4.0,
        _ => -1.0,
    }
}

fn detect_straight(cards: &[u8]) -> Option<(usize, u8)> {
    let mut ranks: Vec<u8> = cards.iter().map(|card| rank(*card)).collect();
    ranks.sort_unstable();
    ranks.dedup();
    if ranks.len() != 5 {
        return None;
    }

    for (order, pattern) in STRAIGHT_PATTERNS.iter().enumerate() {
        let mut pattern_ranks = pattern.to_vec();
        pattern_ranks.sort_unstable();
        if ranks == pattern_ranks {
            let strongest_rank = cards.iter().map(|card| rank(*card)).max().unwrap_or(0);
            let top_suit = cards
                .iter()
                .filter(|card| rank(**card) == strongest_rank)
                .map(|card| suit_strength(*card))
                .max()
                .unwrap_or(0);
            return Some((order, top_suit));
        }
    }

    None
}

fn classify_combo(input_cards: &[u8]) -> Option<Combo> {
    let mut cards = input_cards.to_vec();
    sort_cards(&mut cards);
    let size = cards.len();
    let mut counts = [0u8; 13];
    for card in &cards {
        counts[rank(*card) as usize] += 1;
    }
    let groups = counts.iter().filter(|count| **count > 0).count();

    if size == 1 {
        return Some(Combo {
            kind: KIND_SINGLE,
            size: 1,
            tiebreak: vec![card_power(cards[0]) as f32],
            cards,
        });
    }

    if size == 2 && groups == 1 {
        return Some(Combo {
            kind: KIND_PAIR,
            size: 2,
            tiebreak: vec![
                rank(cards[0]) as f32,
                cards.iter().map(|card| suit_strength(*card)).max().unwrap_or(0) as f32,
            ],
            cards,
        });
    }

    if size == 3 && groups == 1 {
        return Some(Combo {
            kind: KIND_TRIPLE,
            size: 3,
            tiebreak: vec![rank(cards[0]) as f32],
            cards,
        });
    }

    if size != 5 {
        return None;
    }

    let straight = detect_straight(&cards);
    let is_flush = cards.iter().all(|card| card % 4 == cards[0] % 4);
    let mut group_sizes: Vec<u8> = counts.iter().copied().filter(|count| *count > 0).collect();
    group_sizes.sort_by(|left, right| right.cmp(left));

    if let Some((order, top_suit)) = straight {
        if is_flush {
            return Some(Combo {
                kind: KIND_STRAIGHT_FLUSH,
                size: 5,
                tiebreak: vec![combo_kind_order(KIND_STRAIGHT_FLUSH), order as f32, top_suit as f32],
                cards,
            });
        }
    }

    if group_sizes.first() == Some(&4) {
        let quad_rank = counts.iter().position(|count| *count == 4).unwrap_or(0);
        return Some(Combo {
            kind: KIND_FOUR_KIND,
            size: 5,
            tiebreak: vec![combo_kind_order(KIND_FOUR_KIND), quad_rank as f32],
            cards,
        });
    }

    if group_sizes.first() == Some(&3) && group_sizes.get(1) == Some(&2) {
        let triple_rank = counts.iter().position(|count| *count == 3).unwrap_or(0);
        return Some(Combo {
            kind: KIND_FULL_HOUSE,
            size: 5,
            tiebreak: vec![combo_kind_order(KIND_FULL_HOUSE), triple_rank as f32],
            cards,
        });
    }

    if is_flush {
        let mut descending = cards.clone();
        descending.sort_by_key(|card| std::cmp::Reverse(card_power(*card)));
        let mut tiebreak = vec![combo_kind_order(KIND_FLUSH), suit_strength(cards[0]) as f32];
        tiebreak.extend(descending.iter().map(|card| rank(*card) as f32));
        return Some(Combo {
            kind: KIND_FLUSH,
            size: 5,
            tiebreak,
            cards,
        });
    }

    if let Some((order, top_suit)) = straight {
        return Some(Combo {
            kind: KIND_STRAIGHT,
            size: 5,
            tiebreak: vec![combo_kind_order(KIND_STRAIGHT), order as f32, top_suit as f32],
            cards,
        });
    }

    None
}

fn compare_combos(a: &Combo, b: &Combo) -> i8 {
    if a.size != b.size {
        return 0;
    }
    if a.size == 5 {
        let kind_diff = combo_kind_order(a.kind) - combo_kind_order(b.kind);
        if kind_diff > 0.0 {
            return 1;
        }
        if kind_diff < 0.0 {
            return -1;
        }
    } else if a.kind != b.kind {
        return 0;
    }

    let len = a.tiebreak.len().max(b.tiebreak.len());
    for index in 0..len {
        let diff = a.tiebreak.get(index).copied().unwrap_or(0.0)
            - b.tiebreak.get(index).copied().unwrap_or(0.0);
        if diff > 0.0 {
            return 1;
        }
        if diff < 0.0 {
            return -1;
        }
    }
    0
}

fn combo_beats(candidate: &Combo, target: &Option<Combo>) -> bool {
    match target {
        None => true,
        Some(target_combo) => candidate.size == target_combo.size && compare_combos(candidate, target_combo) > 0,
    }
}

fn enumerate_combos(hand: &[u8]) -> Vec<Combo> {
    let mut cards = hand.to_vec();
    sort_cards(&mut cards);
    let mut combos = Vec::new();

    for card in &cards {
        if let Some(combo) = classify_combo(&[*card]) {
            combos.push(combo);
        }
    }

    let mut by_rank: [Vec<u8>; 13] = Default::default();
    for card in &cards {
        by_rank[rank(*card) as usize].push(*card);
    }

    for group in &by_rank {
        for left in 0..group.len() {
            for right in left + 1..group.len() {
                if let Some(combo) = classify_combo(&[group[left], group[right]]) {
                    combos.push(combo);
                }
            }
        }
        for a in 0..group.len() {
            for b in a + 1..group.len() {
                for c in b + 1..group.len() {
                    if let Some(combo) = classify_combo(&[group[a], group[b], group[c]]) {
                        combos.push(combo);
                    }
                }
            }
        }
    }

    for a in 0..cards.len() {
        for b in a + 1..cards.len() {
            for c in b + 1..cards.len() {
                for d in c + 1..cards.len() {
                    for e in d + 1..cards.len() {
                        if let Some(combo) = classify_combo(&[cards[a], cards[b], cards[c], cards[d], cards[e]]) {
                            combos.push(combo);
                        }
                    }
                }
            }
        }
    }

    combos.sort_by(|left, right| {
        let size_diff = left.size.cmp(&right.size);
        if size_diff != std::cmp::Ordering::Equal {
            return size_diff;
        }
        match compare_combos(left, right) {
            n if n < 0 => std::cmp::Ordering::Less,
            n if n > 0 => std::cmp::Ordering::Greater,
            _ => left.kind.cmp(&right.kind),
        }
    });
    combos.dedup_by(|left, right| cards_equal(&left.cards, &right.cards));
    combos
}

fn active_players(game: &Game) -> Vec<usize> {
    (0..2)
        .filter(|player| !game.finished[*player] && !game.hands[*player].is_empty())
        .collect()
}

fn is_terminal(game: &Game) -> bool {
    active_players(game).len() <= 1
}

fn placements(game: &Game) -> Vec<usize> {
    let mut result = Vec::new();
    for player in 0..2 {
        if game.finished[player] {
            result.push(player);
        }
    }
    for player in active_players(game) {
        if !result.contains(&player) {
            result.push(player);
        }
    }
    result
}

fn reward_for_player(game: &Game, player: usize) -> f32 {
    let order = placements(game);
    let Some(place) = order.iter().position(|entry| *entry == player) else {
        return -1.0;
    };
    if order.len() <= 1 || place == 0 {
        return 1.0;
    }
    if place == order.len() - 1 {
        return -1.0;
    }
    1.0 - (2.0 * place as f32) / (order.len() as f32 - 1.0)
}

fn next_active_player(game: &Game, after_player: usize) -> i8 {
    for offset in 1..=2 {
        let player = (after_player + offset) % 2;
        if !game.finished[player] && !game.hands[player].is_empty() {
            return player as i8;
        }
    }
    -1
}

fn legal_moves(game: &Game) -> Vec<MoveR> {
    if is_terminal(game) || game.current_player < 0 {
        return Vec::new();
    }

    let player = game.current_player as usize;
    let is_first_move = game.history_len == 0;
    let mut combos = enumerate_combos(&game.hands[player]);
    if is_first_move {
        combos.retain(|combo| combo.cards.contains(&LOWEST_CARD));
    }
    if game.active_combo.is_some() {
        combos.retain(|combo| combo_beats(combo, &game.active_combo));
    }

    let mut moves: Vec<MoveR> = combos
        .into_iter()
        .map(|combo| MoveR {
            pass: false,
            combo: Some(combo),
        })
        .collect();
    if game.active_combo.is_some() && !is_first_move {
        moves.push(MoveR { pass: true, combo: None });
    }
    moves
}

fn apply_move(game: &Game, mv: &MoveR) -> Game {
    let mut next = game.clone();
    let player = next.current_player as usize;

    if mv.pass {
        next.passes_since_play = next.passes_since_play.saturating_add(1);
        let remaining_after_pass = active_players(&next).len();
        if next.active_combo.is_some()
            && next.last_player >= 0
            && next.passes_since_play as usize >= std::cmp::max(1, remaining_after_pass.saturating_sub(1))
            && !next.finished[next.last_player as usize]
        {
            next.current_player = next.last_player;
            next.active_combo = None;
            next.last_player = -1;
            next.passes_since_play = 0;
        } else {
            next.current_player = next_active_player(&next, player);
        }
    } else if let Some(combo) = &mv.combo {
        for card in &combo.cards {
            if let Some(index) = next.hands[player].iter().position(|entry| entry == card) {
                next.hands[player].remove(index);
            }
        }
        sort_cards(&mut next.hands[player]);
        next.played_by_player[player].extend(combo.cards.iter());
        let went_out = next.hands[player].is_empty();
        if went_out && !next.finished[player] {
            next.finished[player] = true;
        }

        if went_out {
            next.active_combo = None;
            next.last_player = -1;
            next.passes_since_play = 0;
        } else {
            next.active_combo = Some(combo.clone());
            next.last_player = player as i8;
            next.passes_since_play = 0;
        }
        next.current_player = if is_terminal(&next) {
            -1
        } else {
            next_active_player(&next, player)
        };
    }

    next.history_len = next.history_len.saturating_add(1);
    next
}

fn combo_strength(combo: &Combo) -> f32 {
    combo.tiebreak
        .iter()
        .enumerate()
        .map(|(index, value)| value / (index as f32 + 1.0))
        .sum::<f32>()
        / 70.0
}

fn handcrafted_move_score(game: &Game, mv: &MoveR) -> f32 {
    if mv.pass {
        return 0.08;
    }
    let actor = game.current_player as usize;
    let Some(combo) = &mv.combo else {
        return 0.01;
    };
    let remaining_after_move = game.hands[actor].len().saturating_sub(combo.cards.len());
    let mut score = 0.25 + combo.cards.len() as f32 * 0.18;
    if remaining_after_move == 0 {
        score += 5.0;
    } else if remaining_after_move <= 2 {
        score += 0.8;
    }
    if game.active_combo.is_some() {
        score += (0.45 - combo_strength(combo)).max(0.0);
    } else {
        score += (0.7 - combo_strength(combo)).max(0.0);
    }
    if combo.cards.iter().any(|card| rank(*card) == 12) && remaining_after_move > 0 {
        score -= 0.2;
    }
    let next_player = (actor + 1) % 2;
    if game.hands[next_player].len() == 1 && combo.size == 1 {
        score += combo_strength(combo) * 0.6;
    }
    score.max(0.01)
}

fn handcrafted_value(game: &Game, perspective_player: usize) -> f32 {
    if is_terminal(game) {
        return reward_for_player(game, perspective_player);
    }
    let my_count = game.hands[perspective_player].len() as f32;
    let opponent = 1 - perspective_player;
    let opponent_count = game.hands[opponent].len() as f32;
    let count_value = (opponent_count - my_count) / 13.0;
    let hand_strength = game.hands[perspective_player]
        .iter()
        .map(|card| card_power(*card) as f32)
        .sum::<f32>()
        / (game.hands[perspective_player].len().max(1) as f32 * 51.0);
    let control_value = if game.current_player == perspective_player as i8 && game.active_combo.is_none() {
        0.12
    } else {
        0.0
    };
    (count_value * 1.25 + hand_strength * 0.22 + control_value).clamp(-1.0, 1.0)
}

fn handcrafted_priors(game: &Game, moves: &[MoveR]) -> Vec<f32> {
    let scores: Vec<f32> = moves.iter().map(|mv| handcrafted_move_score(game, mv).max(0.0001)).collect();
    let total: f32 = scores.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / moves.len().max(1) as f32; moves.len()];
    }
    scores.iter().map(|score| score / total).collect()
}

fn linear(input: &[f32], layer: &Layer) -> Vec<f32> {
    let mut output = vec![0.0; layer.rows];
    for row in 0..layer.rows {
        let mut sum = layer.bias[row];
        let offset = row * layer.cols;
        for col in 0..layer.cols {
            sum += layer.weight[offset + col] * input[col];
        }
        output[row] = sum;
    }
    output
}

fn relu(values: &mut [f32]) {
    for value in values {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

fn softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = values.iter().map(|value| (*value - max).exp()).collect();
    let total: f32 = exp.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / values.len() as f32; values.len()];
    }
    exp.iter().map(|value| value / total).collect()
}

fn state_features(game: &Game, player: usize, state_dim: usize) -> Vec<f32> {
    let mut features = vec![0.0; state_dim.max(216)];
    for card in &game.hands[player] {
        features[*card as usize] = 1.0;
    }
    for card in &game.played_by_player[player] {
        features[52 + *card as usize] = 1.0;
    }
    let opponent = 1 - player;
    for card in &game.played_by_player[opponent] {
        features[104 + *card as usize] = 1.0;
    }
    if let Some(active) = &game.active_combo {
        for card in &active.cards {
            features[156 + *card as usize] = 1.0;
        }
    }
    features[208] = game.hands[player].len() as f32 / 13.0;
    features[209] = game.hands[opponent].len() as f32 / 13.0;
    features[210] = game.active_combo.as_ref().map(|combo| combo.size as f32).unwrap_or(0.0) / 5.0;
    features[211] = if game.active_combo.is_some() { 0.0 } else { 1.0 };
    features[212] = game.passes_since_play as f32 / 2.0;
    features[213] = (game.history_len as f32).min(100.0) / 100.0;
    features[214] = if game.active_combo.is_some() && game.last_player == player as i8 {
        1.0
    } else {
        0.0
    };
    features[215] = if game.active_combo.is_some() && game.last_player == opponent as i8 {
        1.0
    } else {
        0.0
    };
    features.truncate(state_dim);
    features
}

fn move_features(mv: &MoveR, move_dim: usize) -> Vec<f32> {
    let mut features = vec![0.0; move_dim.max(62)];
    if mv.pass {
        features[52] = 1.0;
    } else if let Some(combo) = &mv.combo {
        for card in &combo.cards {
            features[*card as usize] = 1.0;
        }
        features[53] = combo.size as f32 / 5.0;
        let kind_index = match combo.kind {
            KIND_SINGLE => 0,
            KIND_PAIR => 1,
            KIND_TRIPLE => 2,
            KIND_STRAIGHT => 3,
            KIND_FLUSH => 4,
            KIND_FULL_HOUSE => 5,
            KIND_FOUR_KIND => 6,
            KIND_STRAIGHT_FLUSH => 7,
            _ => 0,
        };
        features[54 + kind_index] = 1.0;
    }
    features.truncate(move_dim);
    features
}

fn greedy_rollout_value(model: &Model, game: &Game, perspective_player: usize) -> f32 {
    let mut rollout = game.clone();
    let mut guard = 0;
    while !is_terminal(&rollout) && guard < 280 {
        let moves = legal_moves(&rollout);
        if moves.is_empty() {
            break;
        }
        let priors = handcrafted_priors(&rollout, &moves);
        let mut best_index = 0;
        for index in 1..moves.len() {
            if priors[index] > priors[best_index] {
                best_index = index;
            }
        }
        rollout = apply_move(&rollout, &moves[best_index]);
        guard += 1;
    }
    if is_terminal(&rollout) {
        reward_for_player(&rollout, perspective_player)
    } else {
        let _ = model;
        handcrafted_value(&rollout, perspective_player)
    }
}

fn evaluate_model(model: &Model, game: &Game, perspective_player: usize, moves: &[MoveR]) -> (Vec<f32>, f32) {
    if is_terminal(game) {
        return (Vec::new(), reward_for_player(game, perspective_player));
    }
    if moves.is_empty() {
        return (Vec::new(), handcrafted_value(game, perspective_player));
    }

    let mut state_hidden = linear(&state_features(game, perspective_player, model.state_dim), &model.state_fc);
    relu(&mut state_hidden);
    let mut value_hidden = linear(&state_hidden, &model.value_fc);
    relu(&mut value_hidden);
    let mut logits = Vec::with_capacity(moves.len());
    for mv in moves {
        let mut move_hidden = linear(&move_features(mv, model.move_dim), &model.move_fc);
        relu(&mut move_hidden);
        let mut policy_input = Vec::with_capacity(state_hidden.len() + move_hidden.len());
        policy_input.extend_from_slice(&state_hidden);
        policy_input.extend_from_slice(&move_hidden);
        let mut policy_hidden = linear(&policy_input, &model.policy_fc);
        relu(&mut policy_hidden);
        logits.push(linear(&policy_hidden, &model.policy_out).first().copied().unwrap_or(0.0));
    }
    let temperature = model.neural_policy_temperature.max(0.05);
    for logit in &mut logits {
        *logit /= temperature;
    }
    let neural_priors = softmax(&logits);
    let mut value = linear(&value_hidden, &model.value_out)
        .first()
        .copied()
        .unwrap_or(0.0)
        .tanh();
    if model.rollout_value_weight > 0.0 {
        let rollout = greedy_rollout_value(model, game, perspective_player);
        value = (1.0 - model.rollout_value_weight) * value + model.rollout_value_weight * rollout;
    }

    let mut priors = neural_priors;
    if model.handcrafted_prior_weight > 0.0 || model.handcrafted_value_weight > 0.0 {
        let handcrafted = handcrafted_priors(game, moves);
        for index in 0..priors.len() {
            priors[index] = (1.0 - model.handcrafted_prior_weight) * priors[index]
                + model.handcrafted_prior_weight * handcrafted[index];
        }
        if model.handcrafted_value_weight > 0.0 {
            value = (1.0 - model.handcrafted_value_weight) * value
                + model.handcrafted_value_weight * handcrafted_value(game, perspective_player);
        }
    }

    (priors, value.clamp(-1.0, 1.0))
}

fn node_value(node: &Node) -> f32 {
    if node.visits == 0 {
        0.0
    } else {
        node.value_sum / node.visits as f32
    }
}

fn make_node(game: Game, mv: Option<MoveR>, prior: f32) -> Node {
    Node {
        game,
        prior,
        visits: 0,
        value_sum: 0.0,
        mv,
        children: None,
    }
}

fn select_child_index(node: &Node, root_player: usize, exploration: f32) -> usize {
    let children = node.children.as_ref().unwrap();
    let parent_visits = node.visits.max(1) as f32;
    let actor = node.game.current_player;
    let mut best_index = 0;
    let mut best_score = f32::NEG_INFINITY;
    for (index, child) in children.iter().enumerate() {
        let value_for_root = node_value(child);
        let q = if actor == root_player as i8 {
            value_for_root
        } else {
            -value_for_root
        };
        let u = exploration * child.prior * parent_visits.sqrt() / (1.0 + child.visits as f32);
        let score = q + u;
        if score > best_score {
            best_score = score;
            best_index = index;
        }
    }
    best_index
}

fn visit(node: &mut Node, model: &Model, root_player: usize, depth: u16, max_depth: u16, exploration: f32) -> f32 {
    node.visits = node.visits.saturating_add(1);
    if is_terminal(&node.game) {
        let value = reward_for_player(&node.game, root_player);
        node.value_sum += value;
        return value;
    }

    let moves = legal_moves(&node.game);
    if depth >= max_depth || moves.is_empty() {
        let (_, value) = evaluate_model(model, &node.game, root_player, &moves);
        node.value_sum += value;
        return value;
    }

    if node.children.is_none() {
        let actor = node.game.current_player as usize;
        let (policy_priors, policy_value) = evaluate_model(model, &node.game, actor, &moves);
        let value = if actor == root_player {
            policy_value
        } else {
            evaluate_model(model, &node.game, root_player, &moves).1
        };
        let children = moves
            .into_iter()
            .enumerate()
            .map(|(index, mv)| {
                let prior = policy_priors.get(index).copied().unwrap_or(0.001).max(0.0001);
                make_node(apply_move(&node.game, &mv), Some(mv), prior)
            })
            .collect();
        node.children = Some(children);
        node.value_sum += value;
        return value;
    }

    let child_index = select_child_index(node, root_player, exploration);
    let value = {
        let children = node.children.as_mut().unwrap();
        visit(&mut children[child_index], model, root_player, depth + 1, max_depth, exploration)
    };
    node.value_sum += value;
    value
}

fn search_move(game: &Game, model: &Model, root_player: usize, simulations: u32, max_depth: u16, exploration: f32) -> SearchOutcome {
    let moves = legal_moves(game);
    if moves.len() == 1 {
        return SearchOutcome {
            best_move: moves[0].clone(),
            visits: vec![(moves[0].clone(), 1)],
            value: evaluate_model(model, game, root_player, &moves).1,
            determinizations: 0,
            simulations: 0,
            legal_move_count: 1,
        };
    }

    let mut root = make_node(game.clone(), None, 1.0);
    let mut simulations_run = 0;
    for _ in 0..simulations.max(1) {
        visit(&mut root, model, root_player, 0, max_depth, exploration);
        simulations_run += 1;
    }

    let root_value = node_value(&root);
    let children = root.children.unwrap_or_default();
    let mut best_index = 0;
    for index in 1..children.len() {
        let best = &children[best_index];
        let child = &children[index];
        if child.visits > best.visits || (child.visits == best.visits && node_value(child) > node_value(best)) {
            best_index = index;
        }
    }
    let best_move = children
        .get(best_index)
        .and_then(|child| child.mv.clone())
        .unwrap_or_else(|| moves.first().cloned().unwrap_or(MoveR { pass: true, combo: None }));
    SearchOutcome {
        best_move,
        visits: children
            .iter()
            .filter_map(|child| child.mv.clone().map(|mv| (mv, child.visits)))
            .collect(),
        value: root_value,
        determinizations: 0,
        simulations: simulations_run,
        legal_move_count: moves.len(),
    }
}

fn move_cards(mv: &MoveR) -> Vec<u8> {
    if mv.pass {
        Vec::new()
    } else {
        mv.combo.as_ref().map(|combo| combo.cards.clone()).unwrap_or_default()
    }
}

fn cards_equal(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut a = left.to_vec();
    let mut b = right.to_vec();
    sort_cards(&mut a);
    sort_cards(&mut b);
    a == b
}

fn moves_equal(left: &MoveR, right: &MoveR) -> bool {
    if left.pass || right.pass {
        return left.pass && right.pass;
    }
    cards_equal(&move_cards(left), &move_cards(right))
}

fn determinize_for_player(game: &Game, observer: usize, rng: &mut Rng) -> Game {
    let mut next = game.clone();
    let mut known = [false; 52];
    for card in &next.hands[observer] {
        known[*card as usize] = true;
    }
    for card in &next.played_by_player[0] {
        known[*card as usize] = true;
    }
    for card in &next.played_by_player[1] {
        known[*card as usize] = true;
    }
    let mut unknown: Vec<u8> = (0..52).filter(|card| !known[*card as usize]).collect();
    rng.shuffle(&mut unknown);
    let opponent = 1 - observer;
    let count = next.hands[opponent].len();
    next.hands[opponent] = unknown.into_iter().take(count).collect();
    sort_cards(&mut next.hands[opponent]);
    next
}

fn observer_search_impl(game: &Game, observer: usize, model: &Model, seed: u32, time_limit_ms: f64, simulations_per_determination: u32) -> SearchOutcome {
    let root_moves = legal_moves(game);
    if root_moves.len() <= 1 {
        return search_move(game, model, observer, simulations_per_determination.max(1), 80, 1.35);
    }

    let deadline = now_ms() + time_limit_ms.max(1.0);
    let per_determination = simulations_per_determination.max(1);
    let mut rng = Rng { state: seed };
    let mut visits = vec![0u32; root_moves.len()];
    let mut value_sum = 0.0;
    let mut determinizations_run = 0u32;
    let mut simulations_run = 0u32;

    while determinizations_run == 0 || now_ms() < deadline {
        let determinized = determinize_for_player(game, observer, &mut rng);
        let result = search_move(&determinized, model, observer, per_determination, 80, 1.35);
        determinizations_run += 1;
        simulations_run += result.simulations;
        value_sum += result.value;
        for (mv, count) in result.visits {
            if let Some(index) = root_moves.iter().position(|root_move| moves_equal(root_move, &mv)) {
                visits[index] = visits[index].saturating_add(count);
            }
        }
    }

    let mut best_index = 0;
    for index in 1..root_moves.len() {
        if visits[index] > visits[best_index] {
            best_index = index;
        }
    }
    SearchOutcome {
        best_move: root_moves[best_index].clone(),
        visits: root_moves
            .iter()
            .cloned()
            .zip(visits.iter().copied())
            .collect(),
        value: value_sum / determinizations_run.max(1) as f32,
        determinizations: determinizations_run,
        simulations: simulations_run,
        legal_move_count: root_moves.len(),
    }
}

impl Rng {
    fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6d2b79f5);
        let mut value = self.state;
        value = (value ^ (value >> 15)).wrapping_mul(value | 1);
        value ^= value.wrapping_add((value ^ (value >> 7)).wrapping_mul(value | 61));
        ((value ^ (value >> 14)) as f64) / 4294967296.0
    }

    fn shuffle(&mut self, values: &mut [u8]) {
        for index in (1..values.len()).rev() {
            let swap_index = (self.next() * (index as f64 + 1.0)).floor() as usize;
            values.swap(index, swap_index);
        }
    }
}

fn read_cards(input: &[i32], cursor: &mut usize) -> Vec<u8> {
    let len = input.get(*cursor).copied().unwrap_or(0).max(0) as usize;
    *cursor += 1;
    let mut cards = Vec::with_capacity(len);
    for _ in 0..len {
        if let Some(card) = input.get(*cursor) {
            cards.push((*card).clamp(0, 51) as u8);
        }
        *cursor += 1;
    }
    sort_cards(&mut cards);
    cards
}

fn parse_game(input: &[i32]) -> Option<Game> {
    if input.len() < 11 {
        return None;
    }
    let mut cursor = 0;
    let current_player = input[cursor] as i8;
    cursor += 1;
    let active_len = input[cursor].max(0) as usize;
    cursor += 1;
    let mut active_cards = Vec::new();
    for index in 0..5 {
        if index < active_len {
            active_cards.push(input[cursor].clamp(0, 51) as u8);
        }
        cursor += 1;
    }
    let active_combo = if active_cards.is_empty() {
        None
    } else {
        classify_combo(&active_cards)
    };
    let last_player = input[cursor] as i8;
    cursor += 1;
    let passes_since_play = input[cursor].max(0) as u8;
    cursor += 1;
    let finished = [input[cursor] != 0, input[cursor + 1] != 0];
    cursor += 2;
    let history_len = input[cursor].max(0) as u16;
    cursor += 1;
    let played0 = read_cards(input, &mut cursor);
    let played1 = read_cards(input, &mut cursor);
    let hand0 = read_cards(input, &mut cursor);
    let hand1 = read_cards(input, &mut cursor);

    Some(Game {
        hands: [hand0, hand1],
        current_player,
        active_combo,
        last_player,
        passes_since_play,
        finished,
        history_len,
        played_by_player: [played0, played1],
    })
}

fn read_layer(weights: &[f32], cursor: &mut usize, rows: usize, cols: usize) -> Layer {
    let weight_len = rows * cols;
    let weight = weights
        .get(*cursor..*cursor + weight_len)
        .unwrap_or(&[])
        .to_vec();
    *cursor += weight_len;
    let bias = weights.get(*cursor..*cursor + rows).unwrap_or(&[]).to_vec();
    *cursor += rows;
    Layer {
        rows,
        cols,
        weight,
        bias,
    }
}

#[wasm_bindgen]
pub fn init_model(weights: &[f32], dims: &[u32], settings: &[f32]) -> bool {
    if dims.len() < 6 {
        return false;
    }
    let state_dim = dims[0] as usize;
    let move_dim = dims[1] as usize;
    let state_hidden = dims[2] as usize;
    let move_hidden = dims[3] as usize;
    let policy_hidden = dims[4] as usize;
    let value_hidden = dims[5] as usize;
    let expected = state_hidden * state_dim
        + state_hidden
        + move_hidden * move_dim
        + move_hidden
        + policy_hidden * (state_hidden + move_hidden)
        + policy_hidden
        + policy_hidden
        + 1
        + value_hidden * state_hidden
        + value_hidden
        + value_hidden
        + 1;
    if weights.len() < expected {
        return false;
    }
    let mut cursor = 0;
    let state_fc = read_layer(weights, &mut cursor, state_hidden, state_dim);
    let move_fc = read_layer(weights, &mut cursor, move_hidden, move_dim);
    let policy_fc = read_layer(weights, &mut cursor, policy_hidden, state_hidden + move_hidden);
    let policy_out = read_layer(weights, &mut cursor, 1, policy_hidden);
    let value_fc = read_layer(weights, &mut cursor, value_hidden, state_hidden);
    let value_out = read_layer(weights, &mut cursor, 1, value_hidden);
    let model = Model {
        state_dim,
        move_dim,
        state_fc,
        move_fc,
        policy_fc,
        policy_out,
        value_fc,
        value_out,
        handcrafted_prior_weight: settings.get(0).copied().unwrap_or(0.0).clamp(0.0, 1.0),
        handcrafted_value_weight: settings.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0),
        neural_policy_temperature: settings.get(2).copied().unwrap_or(1.0).max(0.05),
        rollout_value_weight: settings.get(5).copied().unwrap_or(0.0).clamp(0.0, 1.0),
    };
    MODEL.with(|slot| {
        *slot.borrow_mut() = Some(model);
    });
    true
}

#[wasm_bindgen]
pub fn observer_search(state: &[i32], observer: u32, seed: u32, time_limit_ms: f64, simulations_per_determination: u32) -> Vec<i32> {
    let Some(game) = parse_game(state) else {
        return vec![0, PASS_SENTINEL, -1, -1, -1, -1, 0, 0, 0];
    };
    MODEL.with(|slot| {
        let Some(model) = slot.borrow().as_ref().cloned() else {
            return vec![0, PASS_SENTINEL, -1, -1, -1, -1, 0, 0, 0];
        };
        let result = observer_search_impl(
            &game,
            observer.min(1) as usize,
            &model,
            seed,
            time_limit_ms,
            simulations_per_determination,
        );
        let cards = move_cards(&result.best_move);
        let mut output = vec![cards.len() as i32, -1, -1, -1, -1, -1, 0, result.simulations as i32, result.legal_move_count as i32];
        for (index, card) in cards.iter().take(5).enumerate() {
            output[index + 1] = *card as i32;
        }
        output[6] = result.determinizations as i32;
        output
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_initial_state_with_legal_lowest_card_move() {
        let hand0 = [3, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44];
        let hand1 = [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25];
        let mut state = vec![0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0];
        state.push(hand0.len() as i32);
        state.extend(hand0.iter().map(|card| *card as i32));
        state.push(hand1.len() as i32);
        state.extend(hand1.iter().map(|card| *card as i32));

        let game = parse_game(&state).expect("state parses");
        let moves = legal_moves(&game);
        assert!(moves.iter().any(|mv| move_cards(mv) == vec![LOWEST_CARD]));
    }
}
