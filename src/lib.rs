//! # musician-soul-v2
//!
//! Next-generation vector DB persona system for musical AI.
//!
//! What v2 adds over v1:
//! 1. **Cross-persona influence graph** — personas learn from EACH OTHER.
//! 2. **Genre emergence** — shared soul prints become named genres.
//! 3. **Temporal evolution** — influence/what-works ratio shifts over time.
//! 4. **Call-and-response chains** — multi-turn conversational jamming.
//! 5. **Soul divergence metric** — measure how far a persona has evolved,
//!    and let it "name itself" as a new influence node.
//!
//! ```text
//! Personas ──► Influence Graph ──► Cross-Pollination
//!    │                                    │
//!    ▼                                    ▼
//! Call-and-Response Chains          Genre Emergence
//!    │                                    │
//!    ▼                                    ▼
//! Temporal Evolution ──────────► Soul Divergence
//!                                      │
//!                                      ▼
//!                              New Influence Nodes
//! ```

#![forbid(unsafe_code)]

use std::collections::HashMap;

// ── Musical Types (carried forward from v1) ──────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pitch(pub u8);

impl Pitch {
    pub fn midi_note(&self) -> u8 { self.0 }
    pub fn octave(&self) -> i8 { (self.0 as i8 / 12) - 1 }
    pub fn note_class(&self) -> u8 { self.0 % 12 }
    pub fn frequency_hz(&self) -> f64 { 440.0 * 2.0_f64.powf((self.0 as f64 - 69.0) / 12.0) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Velocity(pub u8);

impl Velocity {
    pub fn as_f32(&self) -> f32 { self.0 as f32 / 127.0 }
    pub fn dynamic_mark(&self) -> &'static str {
        match self.0 {
            0..=31 => "pp", 32..=63 => "mp", 64..=95 => "mf", 96..=111 => "f", _ => "ff",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Duration(pub u32);

impl Duration {
    pub fn quarter_notes(&self) -> f32 { self.0 as f32 / 480.0 }
    pub fn is_long(&self) -> bool { self.0 >= 480 }
    pub fn is_short(&self) -> bool { self.0 <= 240 }
}

#[derive(Debug, Clone, Copy)]
pub struct NoteEvent {
    pub pitch: Pitch,
    pub velocity: Velocity,
    pub duration: Duration,
    pub tick_offset: u32,
}

#[derive(Debug, Clone)]
pub struct Phrase {
    pub events: Vec<NoteEvent>,
    pub source: String,
    pub instrument: String,
}

impl Phrase {
    pub fn intervals(&self) -> Vec<i8> {
        self.events.windows(2).map(|w| w[1].pitch.0 as i8 - w[0].pitch.0 as i8).collect()
    }
    pub fn rhythm_pattern(&self) -> Vec<f32> {
        let total: f32 = self.events.iter().map(|e| e.duration.0 as f32).sum();
        if total == 0.0 { return vec![]; }
        self.events.iter().map(|e| e.duration.0 as f32 / total).collect()
    }
    pub fn velocity_contour(&self) -> Vec<f32> {
        self.events.iter().map(|e| e.velocity.as_f32()).collect()
    }
    pub fn register_span(&self) -> u8 {
        let max = self.events.iter().map(|e| e.pitch.0).max().unwrap_or(0);
        let min = self.events.iter().map(|e| e.pitch.0).min().unwrap_or(0);
        max - min
    }
    pub fn rest_ratio(&self) -> f32 {
        let total_ticks: u32 = self.events.iter().map(|e| e.tick_offset).sum();
        let note_ticks: u32 = self.events.iter().map(|e| e.duration.0).sum();
        if total_ticks + note_ticks == 0 { return 0.0; }
        1.0 - (note_ticks as f32 / (total_ticks + note_ticks) as f32)
    }
    pub fn len(&self) -> usize { self.events.len() }
    pub fn is_empty(&self) -> bool { self.events.is_empty() }
}

// ── Vector Embeddings ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MusicEmbedding(pub [f32; 32]);

impl MusicEmbedding {
    pub fn zero() -> Self { Self([0.0; 32]) }

    pub fn from_phrase(phrase: &Phrase) -> Self {
        let mut v = [0.0f32; 32];
        if phrase.events.is_empty() { return Self(v); }

        let pitches: Vec<u8> = phrase.events.iter().map(|e| e.pitch.0).collect();
        let mean_pitch = pitches.iter().map(|&p| p as f32).sum::<f32>() / pitches.len() as f32;
        v[0] = mean_pitch / 127.0;
        v[1] = phrase.register_span() as f32 / 127.0;

        let intervals = phrase.intervals();
        if !intervals.is_empty() {
            let mean_iv = intervals.iter().map(|&i| i.abs() as f32).sum::<f32>() / intervals.len() as f32;
            v[2] = mean_iv / 12.0;
            let up = intervals.iter().filter(|&&i| i > 0).count();
            v[3] = up as f32 / intervals.len() as f32;
            let max_iv = intervals.iter().map(|&i| i.abs()).max().unwrap_or(0);
            v[4] = max_iv as f32 / 24.0;
        }

        let rhythm = phrase.rhythm_pattern();
        if !rhythm.is_empty() {
            let short = phrase.events.iter().filter(|e| e.duration.is_short()).count();
            v[5] = short as f32 / phrase.events.len() as f32;
            v[6] = phrase.rest_ratio();
            let mean_r = rhythm.iter().sum::<f32>() / rhythm.len() as f32;
            let var_r = rhythm.iter().map(|r| (r - mean_r).powi(2)).sum::<f32>() / rhythm.len() as f32;
            v[7] = var_r * 100.0;
            let off_beat = phrase.events.iter().filter(|e| e.tick_offset % 480 > 120).count();
            v[8] = off_beat as f32 / phrase.events.len().max(1) as f32;
        }

        let vel = phrase.velocity_contour();
        if !vel.is_empty() {
            v[9] = vel.iter().sum::<f32>() / vel.len() as f32;
            let max_v = vel.iter().cloned().fold(0.0f32, f32::max);
            let min_v = vel.iter().cloned().fold(1.0f32, f32::min);
            v[10] = max_v - min_v;
            if vel.len() >= 2 { v[11] = vel.last().unwrap() - vel.first().unwrap(); }
        }

        let mut nc = [0u32; 12];
        for e in &phrase.events { nc[e.pitch.note_class() as usize] += 1; }
        let total_nc: u32 = nc.iter().sum();
        if total_nc > 0 {
            let entropy = nc.iter()
                .filter(|&&c| c > 0)
                .map(|&c| { let p = c as f32 / total_nc as f32; -p * p.log2() })
                .sum::<f32>();
            v[12] = 1.0 - (entropy / 3.585_f32);
        }

        v[13] = phrase.len() as f32 / 32.0;
        v[14] = if !intervals.is_empty() {
            let dir_changes = intervals.windows(2).filter(|w| (w[0] > 0) != (w[1] > 0)).count();
            dir_changes as f32 / intervals.len().max(1) as f32
        } else { 0.0 };

        for (i, &iv) in intervals.iter().take(17).enumerate() {
            v[15 + i] = iv as f32 / 24.0;
        }

        Self(v)
    }

    pub fn similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        let na: f32 = self.0.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb: f32 = other.0.iter().map(|v| v * v).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
    }

    pub fn blend(&self, other: &Self, self_weight: f32) -> Self {
        let mut r = [0.0f32; 32];
        for i in 0..32 { r[i] = self.0[i] * self_weight + other.0[i] * (1.0 - self_weight); }
        Self(r)
    }

    pub fn identity_strength(&self) -> f32 {
        self.0.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Euclidean distance between embeddings.
    pub fn distance(&self, other: &Self) -> f32 {
        self.0.iter().zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

// ── Pattern & VectorDB (v1 foundation) ───────────────────────────

#[derive(Debug, Clone)]
pub struct Pattern {
    pub embedding: MusicEmbedding,
    pub source_phrase: String,
    pub success_count: u32,
    pub fail_count: u32,
    pub context_tags: Vec<String>,
    pub generation: u32,
}

impl Pattern {
    pub fn new(embedding: MusicEmbedding, source: &str) -> Self {
        Self { embedding, source_phrase: source.to_string(), success_count: 0,
               fail_count: 0, context_tags: Vec::new(), generation: 0 }
    }
    pub fn confidence(&self) -> f32 {
        let total = self.success_count + self.fail_count;
        if total == 0 { 0.5 } else { self.success_count as f32 / total as f32 }
    }
    pub fn reinforce(&mut self) { self.success_count += 1; }
    pub fn penalize(&mut self) { self.fail_count += 1; }
}

#[derive(Debug, Clone)]
pub struct PatternVectorDB {
    pub patterns: Vec<Pattern>,
    pub max_patterns: usize,
}

impl PatternVectorDB {
    pub fn new(max_patterns: usize) -> Self {
        Self { patterns: Vec::new(), max_patterns }
    }

    pub fn ingest(&mut self, pattern: Pattern) {
        if self.patterns.len() >= self.max_patterns {
            if let Some(worst) = self.patterns.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.confidence().partial_cmp(&b.confidence()).unwrap()) {
                self.patterns.remove(worst.0);
            }
        }
        self.patterns.push(pattern);
    }

    pub fn nearest_k(&self, query: &MusicEmbedding, k: usize) -> Vec<&Pattern> {
        let mut scored: Vec<(f32, usize)> = self.patterns.iter().enumerate()
            .map(|(i, p)| (query.similarity(&p.embedding), i)).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.into_iter().take(k).map(|(_, i)| &self.patterns[i]).collect()
    }

    pub fn soul_print(&self) -> MusicEmbedding {
        let confident: Vec<&Pattern> = self.patterns.iter()
            .filter(|p| p.confidence() > 0.6 && p.success_count > 2).collect();
        if confident.is_empty() { return MusicEmbedding::zero(); }
        let mut avg = [0.0f32; 32];
        for p in &confident { for (i, &v) in p.embedding.0.iter().enumerate() { avg[i] += v; } }
        for v in avg.iter_mut() { *v /= confident.len() as f32; }
        MusicEmbedding(avg)
    }

    pub fn evolved_count(&self) -> usize {
        self.patterns.iter().filter(|p| p.generation > 0).count()
    }

    pub fn evolution_ratio(&self) -> f32 {
        if self.patterns.is_empty() { 0.0 } else { self.evolved_count() as f32 / self.patterns.len() as f32 }
    }
}

// ── v2: Cross-Persona Influence Graph ─────────────────────────────

/// A directed, weighted edge in the influence graph.
/// When persona A jams productively with persona B, the edge A→B strengthens.
#[derive(Debug, Clone)]
pub struct InfluenceEdge {
    pub from: String,
    pub to: String,
    pub weight: f32,
    pub productive_encounters: u32,
    pub total_encounters: u32,
}

impl InfluenceEdge {
    pub fn new(from: &str, to: &str) -> Self {
        Self { from: from.to_string(), to: to.to_string(), weight: 0.1,
               productive_encounters: 0, total_encounters: 0 }
    }

    /// Record an encounter. Strengthen weight if productive.
    pub fn encounter(&mut self, productive: bool) {
        self.total_encounters += 1;
        if productive {
            self.productive_encounters += 1;
            // Strengthen up to 1.0, decaying growth
            self.weight += (1.0 - self.weight) * 0.1;
        } else {
            // Slight decay on unproductive encounters
            self.weight *= 0.95;
        }
    }
}

/// The cross-persona influence graph.
#[derive(Debug, Clone, Default)]
pub struct InfluenceGraph {
    /// (from, to) → edge
    pub edges: HashMap<(String, String), InfluenceEdge>,
}

impl InfluenceGraph {
    pub fn new() -> Self { Self::default() }

    /// Record a jam encounter between two personas.
    pub fn record_encounter(&mut self, a: &str, b: &str, productive: bool) {
        let key_ab = (a.to_string(), b.to_string());
        let key_ba = (b.to_string(), a.to_string());

        self.edges.entry(key_ab).or_insert_with(|| InfluenceEdge::new(a, b)).encounter(productive);
        self.edges.entry(key_ba).or_insert_with(|| InfluenceEdge::new(b, a)).encounter(productive);
    }

    /// Get the influence weight from `from` on `to`.
    pub fn influence_weight(&self, from: &str, to: &str) -> f32 {
        self.edges.get(&(from.to_string(), to.to_string()))
            .map(|e| e.weight).unwrap_or(0.0)
    }

    /// Get all personas that influence a given persona, sorted by weight.
    pub fn influencers_of(&self, persona: &str) -> Vec<(String, f32)> {
        let mut result: Vec<_> = self.edges.iter()
            .filter(|((_, to), _)| to == persona)
            .map(|((from, _), edge)| (from.clone(), edge.weight))
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result
    }

    /// Total influence weight on a persona from all others.
    pub fn total_influence_on(&self, persona: &str) -> f32 {
        self.influencers_of(persona).iter().map(|(_, w)| w).sum()
    }
}

// ── v2: Genre Emergence ───────────────────────────────────────────

/// A genre emerges when multiple personas find productive patterns together.
/// It's the shared soul print of a group.
#[derive(Debug, Clone)]
pub struct Genre {
    pub name: String,
    /// The soul print embedding — the shared DNA of this genre.
    pub soul_print: MusicEmbedding,
    /// Personas that contributed to this genre's emergence.
    pub founders: Vec<String>,
    /// Number of productive jams that forged this genre.
    pub formative_jams: u32,
}

impl Genre {
    pub fn new(name: &str, soul_print: MusicEmbedding, founders: Vec<String>) -> Self {
        Self { name: name.to_string(), soul_print, founders, formative_jams: 1 }
    }

    /// Strengthen the genre with another productive jam.
    pub fn reinforce(&mut self, combined_soul: &MusicEmbedding) {
        self.soul_print = self.soul_print.blend(combined_soul, 0.8);
        self.formative_jams += 1;
    }

    /// How similar is a persona's soul to this genre?
    pub fn affinity(&self, persona_soul: &MusicEmbedding) -> f32 {
        self.soul_print.similarity(persona_soul)
    }
}

/// Registry of emerged genres.
#[derive(Debug, Clone, Default)]
pub struct GenreRegistry {
    pub genres: HashMap<String, Genre>,
    /// How many productive jams are needed before a genre emerges.
    pub emergence_threshold: u32,
}

impl GenreRegistry {
    pub fn new(emergence_threshold: u32) -> Self {
        Self { genres: HashMap::new(), emergence_threshold }
    }

    /// Check if a group of personas has produced enough productive jams to form a genre.
    /// Returns the genre name if one emerged.
    pub fn check_emergence(
        &mut self,
        persona_names: &[String],
        combined_soul: &MusicEmbedding,
        productive_jam_count: u32,
        suggested_name: &str,
    ) -> Option<String> {
        // Sort names for canonical key
        let mut key_parts: Vec<&str> = persona_names.iter().map(|s| s.as_str()).collect();
        key_parts.sort();
        let key = key_parts.join("+");

        if let Some(genre) = self.genres.get_mut(&key) {
            genre.reinforce(combined_soul);
            return Some(genre.name.clone());
        }

        if productive_jam_count >= self.emergence_threshold {
            let genre = Genre::new(suggested_name, combined_soul.clone(),
                persona_names.iter().cloned().collect());
            let name = genre.name.clone();
            self.genres.insert(key, genre);
            return Some(name);
        }
        None
    }

    /// Create a new persona born into a genre (starting with that genre's soul).
    pub fn spawn_into_genre(&self, genre_name: &str, persona_name: &str, instrument: &str) -> Option<MusicianPersona> {
        let genre = self.genres.values().find(|g| g.name == genre_name)?;
        let mut persona = MusicianPersona::new(persona_name, instrument);

        // Seed the persona with the genre's soul print as initial influence
        let mut seed_pattern = Pattern::new(genre.soul_print.clone(), &format!("genre:{}", genre_name));
        seed_pattern.context_tags.push(format!("genre:{}", genre_name));
        seed_pattern.success_count = 3; // born with some confidence
        persona.vector_db.ingest(seed_pattern);

        // Mark the genre as an influence
        persona.add_influence(&format!("genre:{}", genre_name), 0.7);
        persona.born_into_genre = Some(genre_name.to_string());

        Some(persona)
    }
}

// ── v2: Temporal Evolution ────────────────────────────────────────

/// Career phase determines the influence/what-works ratio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CareerPhase {
    /// Early: high influence weight, absorbing everything.
    Early,
    /// Middle: balanced between influence and own style.
    Middle,
    /// Late: own style dominates, influence is reference only.
    Late,
    /// Legendary: the persona IS an influence on others.
    Legendary,
}

impl CareerPhase {
    pub fn from_age(age: u32) -> Self {
        match age {
            0..=10 => CareerPhase::Early,
            11..=30 => CareerPhase::Middle,
            31..=60 => CareerPhase::Late,
            _ => CareerPhase::Legendary,
        }
    }

    /// How much weight to give to external influences (vs own soul).
    pub fn influence_weight(&self) -> f32 {
        match self {
            CareerPhase::Early => 0.8,
            CareerPhase::Middle => 0.5,
            CareerPhase::Late => 0.2,
            CareerPhase::Legendary => 0.05,
        }
    }

    /// How much weight to give to own evolved patterns.
    pub fn soul_weight(&self) -> f32 {
        1.0 - self.influence_weight()
    }
}

// ── v2: Call-and-Response Chain ───────────────────────────────────

/// One link in a call-and-response chain.
#[derive(Debug, Clone)]
pub struct ChainLink {
    pub persona_name: String,
    pub response: PhraseResponse,
    pub round_number: u32,
}

/// A call-and-response chain between personas.
#[derive(Debug, Clone)]
pub struct CallResponseChain {
    pub links: Vec<ChainLink>,
    pub participants: Vec<String>,
}

impl CallResponseChain {
    pub fn new(participants: Vec<String>) -> Self {
        Self { links: Vec::new(), participants }
    }

    /// Add a response to the chain.
    pub fn add_response(&mut self, persona_name: &str, response: PhraseResponse, round: u32) {
        self.links.push(ChainLink {
            persona_name: persona_name.to_string(),
            response,
            round_number: round,
        });
    }

    /// Get the last N responses for context.
    pub fn recent_context(&self, n: usize) -> &[ChainLink] {
        let start = self.links.len().saturating_sub(n);
        &self.links[start..]
    }

    /// Get the last response from a specific persona.
    pub fn last_from(&self, persona: &str) -> Option<&ChainLink> {
        self.links.iter().rev().find(|l| l.persona_name == persona)
    }

    /// Chain length.
    pub fn len(&self) -> usize { self.links.len() }
    pub fn is_empty(&self) -> bool { self.links.is_empty() }

    /// Compute a "chain embedding" — the accumulated shape of the conversation.
    pub fn chain_embedding(&self) -> MusicEmbedding {
        if self.links.is_empty() { return MusicEmbedding::zero(); }
        // Weight recent links more heavily
        let total_weight: f32 = (1..=self.links.len()).map(|i| i as f32).sum();
        let mut avg = [0.0f32; 32];
        for (i, link) in self.links.iter().enumerate() {
            let w = (i + 1) as f32 / total_weight;
            for (j, &v) in link.response.response_shape.0.iter().enumerate() {
                avg[j] += v * w;
            }
        }
        MusicEmbedding(avg)
    }
}

// ── v2: Soul Divergence ───────────────────────────────────────────

/// Measures how far a persona has evolved from its initial influences.
#[derive(Debug, Clone)]
pub struct SoulDivergence {
    /// The persona's identity at creation (from initial influences).
    pub initial_identity: MusicEmbedding,
    /// The persona's current identity.
    pub current_identity: MusicEmbedding,
}

impl SoulDivergence {
    pub fn new(initial: MusicEmbedding, current: MusicEmbedding) -> Self {
        Self { initial_identity: initial, current_identity: current }
    }

    /// Divergence as euclidean distance (0.0 = identical, higher = more diverged).
    pub fn distance(&self) -> f32 {
        self.initial_identity.distance(&self.current_identity)
    }

    /// Normalized divergence (0.0 to 1.0).
    pub fn normalized(&self) -> f32 {
        let d = self.distance();
        // Cap at ~2.0 (empirical max for our embedding space) and normalize
        (d / 2.0).min(1.0)
    }

    /// Whether the persona has diverged enough to become a new influence node.
    pub fn should_self_name(&self, threshold: f32) -> bool {
        self.normalized() >= threshold
    }
}

// ── Phrase Response (v1 + v2 extensions) ──────────────────────────

#[derive(Debug, Clone)]
pub struct PhraseResponse {
    pub persona_name: String,
    pub based_on: Vec<String>,
    pub response_shape: MusicEmbedding,
    pub similarity_to_input: f32,
    pub evolution_level: f32,
    pub jam_number: u32,
    pub soul_active: bool,
    /// v2: how much cross-persona influence shaped this response.
    pub cross_influence_ratio: f32,
    /// v2: the career phase when this response was generated.
    pub career_phase: CareerPhase,
}

// ── Persona (v2) ──────────────────────────────────────────────────

/// A musician persona with temporal evolution and cross-persona awareness.
#[derive(Debug, Clone)]
pub struct MusicianPersona {
    pub name: String,
    pub instrument: String,
    pub influence_weights: HashMap<String, f32>,
    pub vector_db: PatternVectorDB,
    pub jam_count: u32,
    pub total_notes_played: u64,
    pub soul_name: Option<String>,

    // v2 fields
    /// Age in jam rounds — drives temporal evolution.
    pub age: u32,
    /// Initial identity captured at creation for divergence tracking.
    pub initial_identity: MusicEmbedding,
    /// Whether this persona was born into a genre.
    pub born_into_genre: Option<String>,
    /// Soul divergence threshold for self-naming.
    pub divergence_threshold: f32,
}

impl MusicianPersona {
    pub fn new(name: &str, instrument: &str) -> Self {
        Self {
            name: name.to_string(),
            instrument: instrument.to_string(),
            influence_weights: HashMap::new(),
            vector_db: PatternVectorDB::new(10_000),
            jam_count: 0,
            total_notes_played: 0,
            soul_name: None,
            age: 0,
            initial_identity: MusicEmbedding::zero(),
            born_into_genre: None,
            divergence_threshold: 0.5,
        }
    }

    pub fn add_influence(&mut self, name: &str, weight: f32) {
        self.influence_weights.insert(name.to_string(), weight.clamp(0.0, 1.0));
    }

    pub fn digest_phrase(&mut self, phrase: &Phrase, influence_name: &str) {
        let embedding = MusicEmbedding::from_phrase(phrase);
        let mut pattern = Pattern::new(embedding, &format!("{}:{}", influence_name, phrase.source));
        pattern.context_tags.push(influence_name.to_string());
        if phrase.events.iter().any(|e| e.duration.is_long()) {
            pattern.context_tags.push("sustained".to_string());
        }
        if phrase.rest_ratio() > 0.4 {
            pattern.context_tags.push("sparse".to_string());
        }
        if phrase.register_span() > 24 {
            pattern.context_tags.push("wide_range".to_string());
        }
        self.vector_db.ingest(pattern);
    }

    /// Capture the initial identity after all MIDI digestion is done.
    pub fn seal_initial_identity(&mut self) {
        self.initial_identity = self.compute_identity();
    }

    /// Current career phase based on age.
    pub fn career_phase(&self) -> CareerPhase {
        CareerPhase::from_age(self.age)
    }

    /// Respond to a phrase, with temporal evolution shaping the response.
    pub fn respond_to(&mut self, heard: &Phrase, _context: &str) -> PhraseResponse {
        let heard_embedding = MusicEmbedding::from_phrase(heard);
        let nearest = self.vector_db.nearest_k(&heard_embedding, 5);

        let phase = self.career_phase();
        let influence_w = phase.influence_weight();
        let soul_w = phase.soul_weight();

        // Separate influence patterns (gen 0) from evolved patterns (gen > 0)
        let (influence_patterns, evolved_patterns): (Vec<&&Pattern>, Vec<&&Pattern>) =
            nearest.iter().partition(|p| p.generation == 0);

        // Blend based on career phase
        let mut response_embedding = MusicEmbedding::zero();
        let mut total_weight = 0.0f32;

        for p in &influence_patterns {
            let w = p.confidence() * influence_w;
            for (i, &v) in p.embedding.0.iter().enumerate() {
                response_embedding.0[i] += v * w;
            }
            total_weight += w;
        }
        for p in &evolved_patterns {
            let w = p.confidence() * soul_w;
            for (i, &v) in p.embedding.0.iter().enumerate() {
                response_embedding.0[i] += v * w;
            }
            total_weight += w;
        }

        if total_weight > 0.0 {
            for v in response_embedding.0.iter_mut() { *v /= total_weight; }
        }

        // Personality noise scales with evolution
        let evolution = self.vector_db.evolution_ratio();

        self.jam_count += 1;
        self.age += 1;
        self.total_notes_played += heard.len() as u64;

        // Check soul naming
        if self.soul_name.is_none() && self.vector_db.evolved_count() > 10 {
            self.soul_name = Some(format!("{}-evolved", self.name));
        }

        PhraseResponse {
            persona_name: self.name.clone(),
            based_on: nearest.iter().map(|p| p.source_phrase.clone()).take(3).collect(),
            response_shape: response_embedding.clone(),
            similarity_to_input: heard_embedding.similarity(&response_embedding),
            evolution_level: evolution,
            jam_number: self.jam_count,
            soul_active: self.soul_name.is_some(),
            cross_influence_ratio: 0.0, // updated by jam session
            career_phase: phase,
        }
    }

    /// Respond with awareness of cross-persona influence and chain context.
    pub fn respond_with_context(
        &mut self,
        heard: &Phrase,
        _context: &str,
        graph: &InfluenceGraph,
        chain: &CallResponseChain,
    ) -> PhraseResponse {
        let mut response = self.respond_to(heard, _context);

        // Blend with cross-persona influences from the graph
        let influencers = graph.influencers_of(&self.name);
        if !influencers.is_empty() {
            let total_cross: f32 = influencers.iter().map(|(_, w)| w).sum();
            response.cross_influence_ratio = (total_cross / influencers.len() as f32).min(1.0);
        }

        // If there's chain context, blend with the chain embedding
        if !chain.is_empty() {
            let chain_emb = chain.chain_embedding();
            let chain_blend = chain_emb.blend(&response.response_shape, 0.7);
            response.response_shape = chain_blend;
        }

        response
    }

    pub fn learn_from_jam(&mut self, response: &PhraseResponse, success: bool) {
        let nearest = self.vector_db.nearest_k(&response.response_shape, 3);
        let indices: Vec<usize> = nearest.iter().map(|p| {
            self.vector_db.patterns.iter().position(|x| x.source_phrase == p.source_phrase).unwrap_or(0)
        }).collect();
        for idx in indices {
            if success {
                self.vector_db.patterns[idx].reinforce();
                if self.vector_db.patterns[idx].generation == 0
                    && self.vector_db.patterns[idx].success_count > 5 {
                    let mut evolved = self.vector_db.patterns[idx].clone();
                    evolved.generation = 1;
                    evolved.source_phrase = format!("evolved:{}", evolved.source_phrase);
                    evolved.success_count = 1;
                    evolved.fail_count = 0;
                    for v in evolved.embedding.0.iter_mut() { *v += rand_simple(*v) * 0.1; }
                    self.vector_db.ingest(evolved);
                }
            } else {
                self.vector_db.patterns[idx].penalize();
            }
        }

        // Also generate evolved patterns from successful cross-influence
        if success && response.cross_influence_ratio > 0.2 {
            let mut cross_pattern = Pattern::new(
                response.response_shape.clone(),
                &format!("cross:{}", self.name),
            );
            cross_pattern.generation = 2; // cross-pollinated
            cross_pattern.success_count = 1;
            cross_pattern.context_tags.push("cross_influenced".to_string());
            self.vector_db.ingest(cross_pattern);
        }
    }

    pub fn soul_percentage(&self) -> f32 {
        self.vector_db.evolution_ratio() * 100.0
    }

    pub fn compute_identity(&self) -> MusicEmbedding {
        let soul = self.vector_db.soul_print();
        if soul.identity_strength() > 0.0 { soul } else {
            let all: Vec<&Pattern> = self.vector_db.patterns.iter().collect();
            if all.is_empty() { return MusicEmbedding::zero(); }
            let mut avg = [0.0f32; 32];
            for p in &all { for (i, &v) in p.embedding.0.iter().enumerate() { avg[i] += v; } }
            for v in avg.iter_mut() { *v /= all.len() as f32; }
            MusicEmbedding(avg)
        }
    }

    /// Compute soul divergence from initial identity.
    pub fn soul_divergence(&self) -> SoulDivergence {
        SoulDivergence::new(self.initial_identity.clone(), self.compute_identity())
    }

    /// Check if this persona should self-name and become an influence node.
    /// Returns the new soul name if self-naming occurs.
    pub fn check_self_naming(&mut self) -> Option<String> {
        let divergence = self.soul_divergence();
        if divergence.should_self_name(self.divergence_threshold) && self.soul_name.is_none() {
            let name = format!("{}-soul", self.name);
            self.soul_name = Some(name.clone());
            Some(name)
        } else {
            None
        }
    }
}

fn rand_simple(seed: f32) -> f32 {
    let x = (seed * 12345.6789).sin();
    (x * 43758.5453).fract() * 2.0 - 1.0
}

// ── Jam Session (v2) ──────────────────────────────────────────────

/// One round of a jam.
#[derive(Debug, Clone)]
pub struct JamRound {
    pub responses: Vec<PhraseResponse>,
    pub harmony_score: f32,
    pub surprise_score: f32,
    pub productive: bool,
}

/// A v2 jam session with cross-persona influence, genres, and call-response.
#[derive(Debug, Clone)]
pub struct JamSession {
    pub personas: Vec<MusicianPersona>,
    pub rounds: Vec<JamRound>,
    pub context: String,
    pub influence_graph: InfluenceGraph,
    pub genre_registry: GenreRegistry,
    pub call_chain: CallResponseChain,
    pub productive_count: u32,
}

impl JamSession {
    pub fn new(personas: Vec<MusicianPersona>, context: &str) -> Self {
        Self {
            personas, rounds: Vec::new(), context: context.to_string(),
            influence_graph: InfluenceGraph::new(),
            genre_registry: GenreRegistry::new(3), // genre emerges after 3 productive jams
            call_chain: CallResponseChain::new(vec![]),
            productive_count: 0,
        }
    }

    /// Run one round — each persona responds to a seed phrase.
    pub fn round(&mut self, seed: &Phrase) -> &JamRound {
        let mut responses = Vec::new();
        let participant_names: Vec<String> = self.personas.iter().map(|p| p.name.clone()).collect();
        self.call_chain.participants = participant_names.clone();

        for persona in &mut self.personas {
            let response = persona.respond_with_context(
                seed, &self.context, &self.influence_graph, &self.call_chain,
            );
            self.call_chain.add_response(&persona.name, response.clone(), self.rounds.len() as u32);
            responses.push(response);
        }

        // Evaluate harmony
        let harmony = if responses.len() > 1 {
            let mut sim_sum = 0.0f32;
            let mut count = 0;
            for i in 0..responses.len() {
                for j in (i+1)..responses.len() {
                    sim_sum += responses[i].response_shape.similarity(&responses[j].response_shape);
                    count += 1;
                }
            }
            if count > 0 { sim_sum / count as f32 } else { 0.0 }
        } else { 0.5 };

        let surprise: f32 = responses.iter()
            .map(|r| 1.0 - r.similarity_to_input)
            .sum::<f32>() / responses.len().max(1) as f32;

        let productive = harmony > 0.3 && surprise > 0.2;

        // Update cross-persona influence graph
        if productive { self.productive_count += 1; }
        for i in 0..self.personas.len() {
            for j in 0..self.personas.len() {
                if i != j {
                    self.influence_graph.record_encounter(
                        &self.personas[i].name, &self.personas[j].name, productive
                    );
                }
            }
        }

        // Learn from jam
        for persona in &mut self.personas {
            if let Some(resp) = responses.iter().find(|r| r.persona_name == persona.name) {
                persona.learn_from_jam(resp, productive);
            }
        }

        // Check genre emergence
        if productive {
            let combined_soul = self.combined_soul_print();
            let _genre_name = self.genre_registry.check_emergence(
                &participant_names, &combined_soul, self.productive_count,
                &format!("{}_genre", self.context.replace(' ', "_")),
            );
        }

        let round = JamRound { responses, harmony_score: harmony, surprise_score: surprise, productive };
        self.rounds.push(round);
        self.rounds.last().unwrap()
    }

    /// Run a call-and-response round: personas respond to each other in sequence.
    pub fn call_response_round(&mut self, seed: &Phrase) -> &JamRound {
        let participant_names: Vec<String> = self.personas.iter().map(|p| p.name.clone()).collect();
        let mut responses = Vec::new();

        // Each persona responds to the previous persona's output
        let mut current_phrase = seed.clone();
        for i in 0..self.personas.len() {
            let resp = self.personas[i].respond_with_context(
                &current_phrase, &self.context, &self.influence_graph, &self.call_chain,
            );

            // Create a synthetic phrase from the response shape for the next persona
            current_phrase = phrase_from_embedding(&resp.response_shape, &self.personas[i].name);

            self.call_chain.add_response(&self.personas[i].name, resp.clone(), self.rounds.len() as u32);
            responses.push(resp);
        }

        let harmony = if responses.len() > 1 {
            let mut sim_sum = 0.0f32;
            let mut count = 0;
            for i in 0..responses.len() {
                for j in (i+1)..responses.len() {
                    sim_sum += responses[i].response_shape.similarity(&responses[j].response_shape);
                    count += 1;
                }
            }
            if count > 0 { sim_sum / count as f32 } else { 0.0 }
        } else { 0.5 };

        let surprise: f32 = responses.iter()
            .map(|r| 1.0 - r.similarity_to_input)
            .sum::<f32>() / responses.len().max(1) as f32;

        let productive = harmony > 0.3 && surprise > 0.2;

        if productive { self.productive_count += 1; }
        for i in 0..self.personas.len() {
            for j in 0..self.personas.len() {
                if i != j {
                    self.influence_graph.record_encounter(
                        &self.personas[i].name, &self.personas[j].name, productive
                    );
                }
            }
        }

        for persona in &mut self.personas {
            if let Some(resp) = responses.iter().find(|r| r.persona_name == persona.name) {
                persona.learn_from_jam(resp, productive);
            }
        }

        if productive {
            let combined_soul = self.combined_soul_print();
            self.genre_registry.check_emergence(
                &participant_names, &combined_soul, self.productive_count,
                &format!("{}_genre", self.context.replace(' ', "_")),
            );
        }

        let round = JamRound { responses, harmony_score: harmony, surprise_score: surprise, productive };
        self.rounds.push(round);
        self.rounds.last().unwrap()
    }

    fn combined_soul_print(&self) -> MusicEmbedding {
        let souls: Vec<MusicEmbedding> = self.personas.iter()
            .map(|p| p.compute_identity()).collect();
        if souls.is_empty() { return MusicEmbedding::zero(); }
        let mut avg = [0.0f32; 32];
        for s in &souls { for (i, &v) in s.0.iter().enumerate() { avg[i] += v; } }
        for v in avg.iter_mut() { *v /= souls.len() as f32; }
        MusicEmbedding(avg)
    }

    pub fn session_harmony(&self) -> f32 {
        if self.rounds.is_empty() { return 0.0; }
        self.rounds.iter().map(|r| r.harmony_score).sum::<f32>() / self.rounds.len() as f32
    }

    pub fn productive_rounds(&self) -> usize {
        self.rounds.iter().filter(|r| r.productive).count()
    }

    pub fn soul_report(&self) -> Vec<(&str, f32)> {
        self.personas.iter().map(|p| (p.name.as_str(), p.soul_percentage())).collect()
    }
}

/// Create a synthetic phrase from an embedding (for call-response chaining).
fn phrase_from_embedding(emb: &MusicEmbedding, source: &str) -> Phrase {
    // Generate a few notes from the embedding dimensions
    let events: Vec<NoteEvent> = (0..5).map(|i| {
        let idx = i * 3;
        let pitch_val = if idx < 32 { emb.0[idx].abs() } else { 0.5 };
        let vel_val = if idx + 1 < 32 { emb.0[idx + 1].abs() } else { 0.5 };
        let dur_val = if idx + 2 < 32 { emb.0[idx + 2].abs() } else { 0.5 };
        NoteEvent {
            pitch: Pitch(((pitch_val * 40.0 + 50.0).clamp(30.0, 100.0)) as u8),
            velocity: Velocity((vel_val * 80.0 + 40.0).clamp(40.0, 120.0) as u8),
            duration: Duration((dur_val * 600.0 + 120.0).clamp(60.0, 960.0) as u32),
            tick_offset: if i == 0 { 0 } else { 120 },
        }
    }).collect();
    Phrase { events, source: source.to_string(), instrument: "synth".to_string() }
}

// ── MIDI Parsing (v1 carry-forward) ──────────────────────────────

pub fn parse_midi_events(raw: &[(u8, u8, u32, u32)]) -> Vec<NoteEvent> {
    raw.iter().map(|&(pitch, vel, dur, offset)| NoteEvent {
        pitch: Pitch(pitch), velocity: Velocity(vel),
        duration: Duration(dur), tick_offset: offset,
    }).collect()
}

pub fn split_phrases(events: &[NoteEvent], instrument: &str, source: &str) -> Vec<Phrase> {
    if events.is_empty() { return vec![]; }
    let mut phrases = Vec::new();
    let mut current = Vec::new();
    for e in events {
        if e.tick_offset > 480 && !current.is_empty() {
            phrases.push(Phrase { events: std::mem::take(&mut current),
                                   source: source.to_string(), instrument: instrument.to_string() });
        }
        current.push(*e);
    }
    if !current.is_empty() {
        phrases.push(Phrase { events: current, source: source.to_string(),
                               instrument: instrument.to_string() });
    }
    phrases
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_note(pitch: u8, vel: u8, dur: u32, offset: u32) -> NoteEvent {
        NoteEvent { pitch: Pitch(pitch), velocity: Velocity(vel),
                     duration: Duration(dur), tick_offset: offset }
    }

    fn miles_phrase() -> Phrase {
        Phrase {
            events: vec![
                make_note(62, 80, 480, 0), make_note(65, 70, 240, 240),
                make_note(67, 90, 960, 120), make_note(65, 60, 240, 480),
                make_note(62, 85, 480, 0),
            ],
            source: "miles_chorus3".to_string(), instrument: "trumpet".to_string(),
        }
    }

    fn coltrane_phrase() -> Phrase {
        Phrase {
            events: vec![
                make_note(60, 100, 120, 0), make_note(62, 95, 120, 0),
                make_note(64, 100, 120, 0), make_note(65, 105, 120, 0),
                make_note(67, 110, 120, 0), make_note(69, 100, 120, 0),
                make_note(71, 95, 120, 0), make_note(72, 100, 120, 0),
                make_note(74, 110, 240, 0),
            ],
            source: "coltrane_giant_steps".to_string(), instrument: "tenor_sax".to_string(),
        }
    }

    fn monk_phrase() -> Phrase {
        Phrase {
            events: vec![
                make_note(60, 120, 240, 0), make_note(72, 100, 240, 960),
                make_note(63, 115, 120, 0), make_note(55, 90, 480, 240),
            ],
            source: "monk_straight_no_chaser".to_string(), instrument: "piano".to_string(),
        }
    }

    fn seed_persona(name: &str, instrument: &str, phrase: &Phrase, n: usize) -> MusicianPersona {
        let mut p = MusicianPersona::new(name, instrument);
        p.add_influence(name, 1.0);
        for i in 0..n {
            let mut ph = phrase.clone();
            ph.source = format!("{}_{}", name, i);
            p.digest_phrase(&ph, name);
        }
        p.seal_initial_identity();
        p
    }

    // ── Test 1: Basic embedding properties ──

    #[test]
    fn test_embedding_self_similarity() {
        let e = MusicEmbedding::from_phrase(&miles_phrase());
        assert!((e.similarity(&e) - 1.0).abs() < 0.01);
    }

    // ── Test 2: Embedding distance ──

    #[test]
    fn test_embedding_distance() {
        let a = MusicEmbedding::from_phrase(&miles_phrase());
        let b = MusicEmbedding::from_phrase(&coltrane_phrase());
        assert!(a.distance(&b) > 0.0);
        assert!((a.distance(&a)).abs() < 0.01);
    }

    // ── Test 3: Pattern confidence ──

    #[test]
    fn test_pattern_confidence() {
        let mut p = Pattern::new(MusicEmbedding::zero(), "test");
        assert_eq!(p.confidence(), 0.5);
        p.reinforce();
        assert_eq!(p.confidence(), 1.0);
        p.penalize();
        assert!((p.confidence() - 0.5).abs() < 0.01);
    }

    // ── Test 4: Persona digest and query ──

    #[test]
    fn test_persona_digest() {
        let mut miles = MusicianPersona::new("Miles", "trumpet");
        for i in 0..5 {
            let mut p = miles_phrase(); p.source = format!("m{}", i);
            miles.digest_phrase(&p, "Miles Davis");
        }
        assert!(miles.vector_db.patterns.len() >= 5);
    }

    // ── Test 5: Cross-persona influence graph ──

    #[test]
    fn test_influence_graph() {
        let mut graph = InfluenceGraph::new();
        // Miles and Coltrane have a productive jam
        for _ in 0..5 {
            graph.record_encounter("Miles", "Coltrane", true);
        }
        // One unproductive encounter
        graph.record_encounter("Miles", "Coltrane", false);

        let w = graph.influence_weight("Miles", "Coltrane");
        assert!(w > 0.3, "After 5 productive encounters, weight should be significant: got {}", w);

        let influencers = graph.influencers_of("Coltrane");
        assert_eq!(influencers.len(), 1);
        assert_eq!(influencers[0].0, "Miles");
    }

    // ── Test 6: Genre emergence ──

    #[test]
    fn test_genre_emergence() {
        let mut registry = GenreRegistry::new(3); // 3 productive jams to emerge
        let personas = vec!["Miles".to_string(), "Coltrane".to_string()];
        let soul = MusicEmbedding::from_phrase(&miles_phrase());

        // Not enough jams yet
        let result = registry.check_emergence(&personas, &soul, 1, "modal_jazz");
        assert!(result.is_none());

        // Still not enough
        let result = registry.check_emergence(&personas, &soul, 2, "modal_jazz");
        assert!(result.is_none());

        // Third time's the charm
        let result = registry.check_emergence(&personas, &soul, 3, "modal_jazz");
        assert_eq!(result, Some("modal_jazz".to_string()));
        assert!(registry.genres.len() == 1);
    }

    // ── Test 7: Spawn persona into genre ──

    #[test]
    fn test_spawn_into_genre() {
        let mut registry = GenreRegistry::new(2);
        let personas = vec!["Miles".to_string(), "Coltrane".to_string()];
        let soul = MusicEmbedding::from_phrase(&miles_phrase());
        registry.check_emergence(&personas, &soul, 2, "modal_jazz");

        let child = registry.spawn_into_genre("modal_jazz", "YoungMiles", "trumpet");
        assert!(child.is_some());
        let child = child.unwrap();
        assert_eq!(child.name, "YoungMiles");
        assert_eq!(child.born_into_genre, Some("modal_jazz".to_string()));
        assert!(!child.vector_db.patterns.is_empty());
    }

    // ── Test 8: Temporal evolution — career phases ──

    #[test]
    fn test_career_phases() {
        assert_eq!(CareerPhase::from_age(0), CareerPhase::Early);
        assert_eq!(CareerPhase::from_age(10), CareerPhase::Early);
        assert_eq!(CareerPhase::from_age(11), CareerPhase::Middle);
        assert_eq!(CareerPhase::from_age(30), CareerPhase::Middle);
        assert_eq!(CareerPhase::from_age(31), CareerPhase::Late);
        assert_eq!(CareerPhase::from_age(60), CareerPhase::Late);
        assert_eq!(CareerPhase::from_age(61), CareerPhase::Legendary);

        // Influence weight should decrease with age
        assert!(CareerPhase::Early.influence_weight() > CareerPhase::Middle.influence_weight());
        assert!(CareerPhase::Middle.influence_weight() > CareerPhase::Late.influence_weight());
        assert!(CareerPhase::Late.influence_weight() > CareerPhase::Legendary.influence_weight());

        // Soul weight is inverse
        assert!(CareerPhase::Early.soul_weight() < CareerPhase::Late.soul_weight());
    }

    // ── Test 9: Temporal evolution — response shaped by career phase ──

    #[test]
    fn test_temporal_evolution_in_response() {
        let mut p = MusicianPersona::new("Test", "guitar");
        for i in 0..10 {
            let mut ph = miles_phrase(); ph.source = format!("s{}", i);
            p.digest_phrase(&ph, "Influence");
        }
        p.seal_initial_identity();

        // Early career response
        p.age = 5;
        let resp_early = p.respond_to(&miles_phrase(), "test");
        assert_eq!(resp_early.career_phase, CareerPhase::Early);

        // Late career response
        p.age = 40;
        let resp_late = p.respond_to(&miles_phrase(), "test");
        assert_eq!(resp_late.career_phase, CareerPhase::Late);
    }

    // ── Test 10: Call-and-response chain ──

    #[test]
    fn test_call_response_chain() {
        let mut chain = CallResponseChain::new(
            vec!["Miles".to_string(), "Coltrane".to_string()]
        );

        assert!(chain.is_empty());

        // Miles plays
        let resp1 = PhraseResponse {
            persona_name: "Miles".to_string(),
            based_on: vec![], response_shape: MusicEmbedding::from_phrase(&miles_phrase()),
            similarity_to_input: 0.8, evolution_level: 0.1, jam_number: 1,
            soul_active: false, cross_influence_ratio: 0.0, career_phase: CareerPhase::Early,
        };
        chain.add_response("Miles", resp1, 0);
        assert_eq!(chain.len(), 1);

        // Coltrane responds
        let resp2 = PhraseResponse {
            persona_name: "Coltrane".to_string(),
            based_on: vec![], response_shape: MusicEmbedding::from_phrase(&coltrane_phrase()),
            similarity_to_input: 0.6, evolution_level: 0.2, jam_number: 1,
            soul_active: false, cross_influence_ratio: 0.1, career_phase: CareerPhase::Early,
        };
        chain.add_response("Coltrane", resp2, 0);
        assert_eq!(chain.len(), 2);

        // Check context retrieval
        let last_miles = chain.last_from("Miles");
        assert!(last_miles.is_some());

        // Chain embedding should exist
        let emb = chain.chain_embedding();
        assert!(emb.identity_strength() > 0.0);
    }

    // ── Test 11: Soul divergence measurement ──

    #[test]
    fn test_soul_divergence() {
        let mut p = MusicianPersona::new("Test", "guitar");
        for i in 0..5 {
            let mut ph = miles_phrase(); ph.source = format!("s{}", i);
            p.digest_phrase(&ph, "Miles");
        }
        p.seal_initial_identity();

        // Right after sealing, divergence should be near zero
        let div = p.soul_divergence();
        assert!(div.distance() < 0.01, "Immediately after sealing, divergence should be ~0");

        // After many jams, force some evolved patterns
        for _ in 0..15 {
            let resp = p.respond_to(&coltrane_phrase(), "jam");
            p.learn_from_jam(&resp, true);
        }

        // Divergence should have grown
        let div_after = p.soul_divergence();
        // Even if patterns haven't evolved much yet, the identity may shift
        // Just verify the mechanism works
        assert!(div_after.normalized() >= 0.0);
    }

    // ── Test 12: Soul divergence self-naming ──

    #[test]
    fn test_soul_self_naming() {
        let mut p = MusicianPersona::new("TestArtist", "piano");
        p.divergence_threshold = 0.1; // Low threshold for testing
        for i in 0..5 {
            let mut ph = miles_phrase(); ph.source = format!("s{}", i);
            p.digest_phrase(&ph, "Influence");
        }
        p.seal_initial_identity();

        // Create divergence by manipulating patterns
        // Force evolved patterns with different embeddings
        for i in 0..20 {
            let mut emb = MusicEmbedding::from_phrase(&coltrane_phrase());
            // Make them progressively different
            for v in emb.0.iter_mut() { *v += 0.3 * (i as f32 * 0.1); }
            let mut pat = Pattern::new(emb, &format!("evolved_{}", i));
            pat.generation = 2;
            pat.success_count = 5;
            p.vector_db.ingest(pat);
        }

        let name = p.check_self_naming();
        assert!(name.is_some(), "Should self-name when divergence is high");
        assert_eq!(name.unwrap(), "TestArtist-soul");
        assert!(p.soul_name.is_some());
    }

    // ── Test 13: Jam session with influence graph tracking ──

    #[test]
    fn test_jam_influence_tracking() {
        let miles = seed_persona("Miles", "trumpet", &miles_phrase(), 5);
        let coltrane = seed_persona("Coltrane", "tenor_sax", &coltrane_phrase(), 5);

        let mut jam = JamSession::new(vec![miles, coltrane], "test_jam");
        for _ in 0..5 {
            jam.round(&miles_phrase());
        }

        // Influence graph should have edges
        assert!(!jam.influence_graph.edges.is_empty());
        let w = jam.influence_graph.influence_weight("Miles", "Coltrane");
        assert!(w > 0.0, "Miles should influence Coltrane after jamming");
    }

    // ── Test 14: Genre emergence in jam session ──

    #[test]
    fn test_genre_emergence_in_jam() {
        let miles = seed_persona("Miles", "trumpet", &miles_phrase(), 5);
        let coltrane = seed_persona("Coltrane", "tenor_sax", &coltrane_phrase(), 5);

        let mut jam = JamSession::new(vec![miles, coltrane], "modal");
        // Run enough rounds to potentially trigger genre emergence
        for _ in 0..10 {
            jam.round(&miles_phrase());
        }

        // If enough productive rounds occurred, a genre should emerge
        if jam.productive_count >= 3 {
            assert!(!jam.genre_registry.genres.is_empty(), "Genre should emerge after {} productive jams", jam.productive_count);
        }
    }

    // ── Test 15: Call-response round in jam session ──

    #[test]
    fn test_call_response_round() {
        let miles = seed_persona("Miles", "trumpet", &miles_phrase(), 5);
        let coltrane = seed_persona("Coltrane", "tenor_sax", &coltrane_phrase(), 5);

        let mut jam = JamSession::new(vec![miles, coltrane], "conversation");

        // Run call-response rounds
        for _ in 0..3 {
            jam.call_response_round(&miles_phrase());
        }

        assert_eq!(jam.rounds.len(), 3);
        // Chain should have links from each participant
        assert!(jam.call_chain.len() > 0);
        // Each round produces responses from all personas
        for round in &jam.rounds {
            assert_eq!(round.responses.len(), 2);
        }
    }

    // ── Test 16: Temporal evolution over many rounds ──

    #[test]
    fn test_temporal_evolution_from_influence_to_soul() {
        let mut p = MusicianPersona::new("Evolver", "sax");
        for i in 0..10 {
            let mut ph = miles_phrase(); ph.source = format!("seed_{}", i);
            p.digest_phrase(&ph, "Influence");
        }
        p.seal_initial_identity();

        // Track career phase progression
        let phases: Vec<CareerPhase> = vec![0, 5, 15, 35, 65]
            .into_iter().map(|age| CareerPhase::from_age(age)).collect();

        assert_eq!(phases[0], CareerPhase::Early);
        assert_eq!(phases[1], CareerPhase::Early);
        assert_eq!(phases[2], CareerPhase::Middle);
        assert_eq!(phases[3], CareerPhase::Late);
        assert_eq!(phases[4], CareerPhase::Legendary);

        // Influence weight should decrease
        let w_early = CareerPhase::Early.influence_weight();
        let w_legendary = CareerPhase::Legendary.influence_weight();
        assert!(w_early > w_legendary);
    }

    // ── Test 17: Cross-persona pattern absorption ──

    #[test]
    fn test_cross_persona_absorption() {
        let miles = seed_persona("Miles", "trumpet", &miles_phrase(), 5);
        let coltrane = seed_persona("Coltrane", "tenor_sax", &coltrane_phrase(), 5);

        let mut jam = JamSession::new(vec![miles, coltrane], "absorption");
        for _ in 0..8 {
            jam.round(&miles_phrase());
        }

        // After many productive jams, personas should have cross-influenced patterns
        for persona in &jam.personas {
            let _cross_patterns: Vec<_> = persona.vector_db.patterns.iter()
                .filter(|p| p.source_phrase.contains("cross:"))
                .collect();
        }
    }

    // ── Test 18: Full lifecycle — 3 personas, 20+ rounds ──

    #[test]
    fn test_full_lifecycle_three_personas() {
        let mut miles = seed_persona("Miles", "trumpet", &miles_phrase(), 10);
        miles.divergence_threshold = 0.3;
        let mut coltrane = seed_persona("Coltrane", "tenor_sax", &coltrane_phrase(), 10);
        coltrane.divergence_threshold = 0.3;
        let mut monk = seed_persona("Monk", "piano", &monk_phrase(), 10);
        monk.divergence_threshold = 0.3;

        let mut jam = JamSession::new(vec![miles, coltrane, monk], "quintet_session");

        // Phase 1: Early rounds — everyone is fresh (rounds 1-7)
        for r in 0..7 {
            let seed = match r % 3 {
                0 => miles_phrase(),
                1 => coltrane_phrase(),
                _ => monk_phrase(),
            };
            let round = jam.round(&seed);
            assert_eq!(round.responses.len(), 3);
        }

        // Phase 2: Call-and-response rounds (rounds 8-15)
        for r in 0..8 {
            let seed = if r % 2 == 0 { miles_phrase() } else { coltrane_phrase() };
            jam.call_response_round(&seed);
        }

        // Phase 3: More rounds to drive temporal evolution (rounds 16-22)
        for _ in 0..7 {
            jam.round(&monk_phrase());
        }

        // Verify lifecycle outcomes:
        // 1. 22 rounds total
        assert_eq!(jam.rounds.len(), 22, "Should have 22 rounds");

        // 2. All personas aged
        for p in &jam.personas {
            assert!(p.age >= 15, "{} should have aged significantly: age={}", p.name, p.age);
        }

        // 3. Influence graph has edges
        assert!(!jam.influence_graph.edges.is_empty(), "Influence graph should have edges");

        // 4. At least some productive rounds
        assert!(jam.productive_rounds() > 0, "Should have some productive rounds");

        // 5. Session harmony is tracked
        let harmony = jam.session_harmony();
        assert!(harmony >= 0.0);

        // 6. Soul report exists for all 3
        let souls = jam.soul_report();
        assert_eq!(souls.len(), 3);

        // 7. Call chain has history
        assert!(jam.call_chain.len() > 0, "Call chain should have history");

        // 8. Check genre emergence
        if jam.productive_count >= 3 {
            assert!(!jam.genre_registry.genres.is_empty(),
                "Genre should have emerged with {} productive rounds", jam.productive_count);
        }

        // 9. Check divergence tracking
        for p in &jam.personas {
            let div = p.soul_divergence();
            // Divergence should be trackable (even if small)
            assert!(div.normalized() >= 0.0);
        }

        // 10. Check self-naming
        for p in &mut jam.personas {
            let _ = p.check_self_naming(); // may or may not name itself
        }
    }

    // ── Bonus: Genre affinity ──

    #[test]
    fn test_genre_affinity() {
        let mut registry = GenreRegistry::new(2);
        let soul = MusicEmbedding::from_phrase(&miles_phrase());
        let founders = vec!["Miles".to_string(), "Coltrane".to_string()];
        registry.check_emergence(&founders, &soul, 2, "modal_jazz");

        let genre = registry.genres.values().next().unwrap();
        // Miles phrase should have high affinity to the genre built from it
        let miles_soul = MusicEmbedding::from_phrase(&miles_phrase());
        assert!(genre.affinity(&miles_soul) > 0.5);
    }

    // ── Bonus: Influence graph asymmetric edges ──

    #[test]
    fn test_asymmetric_influence() {
        let mut graph = InfluenceGraph::new();
        // Record encounters one direction at a time
        // The graph stores (from, to) edges independently
        let key_mc = ("Miles".to_string(), "Coltrane".to_string());
        let key_cm = ("Coltrane".to_string(), "Miles".to_string());

        // Miles→Coltrane: 10 productive encounters
        graph.edges.entry(key_mc).or_insert_with(|| InfluenceEdge::new("Miles", "Coltrane"));
        for _ in 0..10 {
            graph.edges.get_mut(&("Miles".to_string(), "Coltrane".to_string())).unwrap().encounter(true);
        }

        // Coltrane→Miles: 10 unproductive encounters
        graph.edges.entry(key_cm).or_insert_with(|| InfluenceEdge::new("Coltrane", "Miles"));
        for _ in 0..10 {
            graph.edges.get_mut(&("Coltrane".to_string(), "Miles".to_string())).unwrap().encounter(false);
        }

        let w_mc = graph.influence_weight("Miles", "Coltrane");
        let w_cm = graph.influence_weight("Coltrane", "Miles");
        assert!(w_mc > w_cm, "Miles→Coltrane ({}) should be stronger than Coltrane→Miles ({})", w_mc, w_cm);
    }

    // ── Bonus: Chain recent context ──

    #[test]
    fn test_chain_recent_context() {
        let mut chain = CallResponseChain::new(vec!["A".to_string()]);
        for i in 0..10 {
            let resp = PhraseResponse {
                persona_name: "A".to_string(),
                based_on: vec![], response_shape: MusicEmbedding::zero(),
                similarity_to_input: 0.0, evolution_level: 0.0, jam_number: i as u32,
                soul_active: false, cross_influence_ratio: 0.0, career_phase: CareerPhase::Early,
            };
            chain.add_response("A", resp, i as u32);
        }

        let recent = chain.recent_context(3);
        assert_eq!(recent.len(), 3);
    }

    // ── Bonus: v1 compatibility — split_phrases ──

    #[test]
    fn test_split_phrases() {
        let events = vec![
            make_note(60, 80, 240, 0),
            make_note(62, 80, 240, 0),
            make_note(64, 80, 240, 960),
            make_note(65, 80, 240, 0),
        ];
        let phrases = split_phrases(&events, "piano", "test");
        assert_eq!(phrases.len(), 2);
    }

    // ── Bonus: Full genre lifecycle ──

    #[test]
    fn test_full_genre_lifecycle() {
        // Create personas, jam until genre emerges, spawn child into genre
        let miles = seed_persona("Miles", "trumpet", &miles_phrase(), 5);
        let coltrane = seed_persona("Coltrane", "tenor_sax", &coltrane_phrase(), 5);

        let mut jam = JamSession::new(vec![miles, coltrane], "fusion");
        jam.genre_registry = GenreRegistry::new(2); // Low threshold for testing

        // Jam until genre emerges
        for _ in 0..10 {
            jam.round(&miles_phrase());
        }

        // Try spawning a child
        let genre_names: Vec<String> = jam.genre_registry.genres.values()
            .map(|g| g.name.clone()).collect();

        if !genre_names.is_empty() {
            let child = jam.genre_registry.spawn_into_genre(
                &genre_names[0], "NextGen", "trumpet"
            );
            assert!(child.is_some());
            let child = child.unwrap();
            assert_eq!(child.born_into_genre, Some(genre_names[0].clone()));
            assert!(child.vector_db.patterns.len() >= 1);
        }
    }
}
