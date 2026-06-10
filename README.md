# musician-soul-v2

vector DB persona system for musical AI — adds cross-persona influence graphs, genre emergence, temporal career phases, call-and-response chains, and soul divergence tracking.

## Why This Exists

V1 proved that personas can develop independent musical identity through jam sessions. But real musical ecosystems are richer: musicians influence *each other*, genres emerge when enough musicians find common ground, and artists evolve through career phases (mimic → find voice → become influence). V2 models all of this.

The five additions over v1:

1. **Influence graph** — Directed, weighted edges between personas. Productive jams strengthen edges; unproductive ones decay them. Asymmetric: Miles might influence Coltrane more than Coltrane influences Miles.

2. **Genre emergence** — When N personas have enough productive jams together, a genre crystallizes: a shared soul print becomes a named genre. New personas can be *born into* a genre, starting with its DNA.

3. **Career phases** — `Early` (80% influence, 20% soul), `Middle` (50/50), `Late` (20/80), `Legendary` (5/95). A persona's response blending shifts over its lifetime.

4. **Call-and-response chains** — Instead of everyone responding to the same seed simultaneously, personas respond *to each other* in sequence, creating conversational musical dialogue.

5. **Soul divergence** — Measures how far a persona has evolved from its initial influences. When divergence exceeds a threshold, the persona "self-names" and becomes a new influence node for others.

## Architecture

```text
Personas ──► Influence Graph ──► Cross-Pollination
   │                                    │
   ▼                                    ▼
Call-Response Chains              Genre Emergence
   │                                    │
   ▼                                    ▼
Temporal Evolution ──────────► Soul Divergence
                                      │
                                      ▼
                              Self-Naming (New Influence Node)
```

### Key New Types

- **`InfluenceGraph`** — Directed weighted graph between persona names. Edges strengthen/decay per encounter.
- **`Genre`** — Named soul print shared by a group of founders. Can spawn new personas.
- **`GenreRegistry`** — Tracks emerged genres. Configurable emergence threshold.
- **`CareerPhase`** — `Early` / `Middle` / `Late` / `Legendary`. Drives influence-vs-soul blending.
- **`CallResponseChain`** — Ordered sequence of responses with recency-weighted chain embedding.
- **`SoulDivergence`** — Euclidean distance between initial and current identity. Drives self-naming.
- **`JamSession`** (v2) — Extended with graph tracking, genre emergence, and call-response rounds.

### Career Phase Mechanics

| Phase | Age Range | Influence Weight | Soul Weight | Behavior |
|-------|-----------|-----------------|-------------|----------|
| Early | 0–10 | 80% | 20% | Absorbs everything |
| Middle | 11–30 | 50% | 50% | Balanced exploration |
| Late | 31–60 | 20% | 80% | Own style dominates |
| Legendary | 61+ | 5% | 95% | Minimal external influence |

## Usage

```rust
use musician_soul_v2::*;

// Create and seed personas
let mut miles = MusicianPersona::new("Miles", "trumpet");
miles.add_influence("Miles Davis", 1.0);
for i in 0..10 {
    let mut ph = miles_phrase();
    ph.source = format!("m{}", i);
    miles.digest_phrase(&ph, "Miles Davis");
}
miles.seal_initial_identity(); // Capture starting point for divergence tracking

// Similar setup for Coltrane, Monk...
// (see tests for full setup with seed_persona helper)

// Jam session with all v2 features
let mut jam = JamSession::new(vec![miles, coltrane, monk], "quintet_session");

// Standard rounds
for _ in 0..5 {
    jam.round(&seed_phrase());
}

// Call-and-response rounds (sequential dialogue)
for _ in 0..3 {
    jam.call_response_round(&seed_phrase());
}

// Check influence graph
let influencers = jam.influence_graph.influencers_of("Coltrane");
for (name, weight) in influencers {
    println!("{} influences Coltrane: {:.2}", name, weight);
}

// Check genre emergence
if !jam.genre_registry.genres.is_empty() {
    for (key, genre) in &jam.genre_registry.genres {
        println!("Genre '{}' emerged from {}", genre.name, key);
    }

    // Spawn a child persona born into a genre
    let child = jam.genre_registry.spawn_into_genre("modal_jazz", "YoungMiles", "trumpet");
}

// Check soul divergence
for persona in &jam.personas {
    let div = persona.soul_divergence();
    println!("{} divergence: {:.3} (normalized: {:.3})",
        persona.name, div.distance(), div.normalized());
}

// Check self-naming
for persona in &mut jam.personas {
    if let Some(soul_name) = persona.check_self_naming() {
        println!("{} self-named as: {}", persona.name, soul_name);
    }
}
```

## API Reference

### V1 Types (carried forward)
All v1 types are present: `Pitch`, `Velocity`, `Duration`, `NoteEvent`, `Phrase`, `MusicEmbedding`, `Pattern`, `PatternVectorDB`, `MusicianPersona` (extended).

### `MusicianPersona` (v2 extensions)
- `.seal_initial_identity()` — Capture starting identity for divergence tracking
- `.career_phase()` → `CareerPhase` — Current phase based on age
- `.respond_with_context(heard, context, graph, chain)` → `PhraseResponse` — Context-aware response
- `.soul_divergence()` → `SoulDivergence` — Distance from initial identity
- `.check_self_naming()` → `Option<String>` — Self-name if diverged enough
- `.compute_identity()` → `MusicEmbedding` — Current identity vector
- Fields: `age`, `initial_identity`, `born_into_genre`, `divergence_threshold`

### `InfluenceGraph`
- `new()` — Empty graph
- `.record_encounter(a, b, productive)` — Bidirectional edge update
- `.influence_weight(from, to)` → `f32`
- `.influencers_of(persona)` → `Vec<(String, f32)>` — Sorted by weight
- `.total_influence_on(persona)` → `f32`

### `Genre`
- `new(name, soul_print, founders)` — Create genre
- `.reinforce(combined_soul)` — Strengthen with another productive jam
- `.affinity(persona_soul)` → `f32` — Cosine similarity to genre

### `GenreRegistry`
- `new(emergence_threshold)` — N productive jams before genre emerges
- `.check_emergence(personas, combined_soul, count, name)` → `Option<String>`
- `.spawn_into_genre(name, persona_name, instrument)` → `Option<MusicianPersona>`

### `CareerPhase`
- `from_age(age)` — Map age to phase
- `.influence_weight()` / `.soul_weight()` — Blending ratios

### `CallResponseChain`
- `new(participants)` — Empty chain
- `.add_response(persona, response, round)` — Append link
- `.recent_context(n)` → `&[ChainLink]` — Last N responses
- `.last_from(persona)` → `Option<&ChainLink>`
- `.chain_embedding()` → `MusicEmbedding` — Recency-weighted centroid

### `SoulDivergence`
- `.distance()` → `f32` — Euclidean distance
- `.normalized()` → `f32` — Clamped to [0, 1]
- `.should_self_name(threshold)` → `bool`

### `JamSession` (v2)
- `new(personas, context)` — With graph, genres, chain, all initialized
- `.round(seed)` → `&JamRound` — Standard round with graph/genre updates
- `.call_response_round(seed)` → `&JamRound` — Sequential persona responses
- `.session_harmony()`, `.productive_rounds()`, `.soul_report()` — Analytics
- Fields: `influence_graph`, `genre_registry`, `call_chain`, `productive_count`

### `PhraseResponse` (v2 extensions)
- Fields: `cross_influence_ratio`, `career_phase` (in addition to v1 fields)

## The Deeper Idea

V2 asks: what happens when you put multiple evolving personas in a room and let them jam for hundreds of rounds? The answer, empirically: influence graphs grow asymmetrically, genres emerge from shared patterns, personas self-name when they've diverged far enough from their influences, and children born into genres start with an advantage.

The self-naming mechanism is the philosophical core. A persona starts by mimicking Miles Davis. Over enough jams, its soul diverges. When the distance exceeds a threshold, it *names itself* — becoming a new influence node that future personas can learn from. This is how genres actually form: someone copies, deviates, and becomes the thing others copy.

## Related Crates

- [`musician-soul`](../musician-soul) — V1: single-persona evolution with jam sessions and soul prints
- [`ternary-cuda-kernels-v2`](../ternary-cuda-kernels-v2) — GPU kernels for harmony computation, voice leading, and groove scheduling
- [`character-sheet`](../character-sheet) — The `.nail` bundle format for persisting persona state
