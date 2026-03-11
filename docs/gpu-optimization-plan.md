# GPU Optimization Plan

This branch documents and tracks GPU-focused optimization attempts for proving.

## Branch

- Name: `gpu-attempts-sumcheck`
- Purpose: isolate experimental GPU work from stable proving code.

## Primary Targets

1. Sumcheck hypercube reduction
- File: `crates/sumcheck/src/prove.rs`
- Function: `compute_over_hypercube`
- Why: largest regular map-reduce kernel over `2^n` points; best expected GPU payoff.

2. Sumcheck fold kernels
- File: `crates/utils/src/multilinear.rs`
- Functions: `fold_multilinear_in_small_field`, `fold_multilinear_in_large_field`, and batch wrappers.
- Why: repeatedly called in each sumcheck round; arithmetic-heavy and data-parallel.

3. Prover-side witness linear algebra
- File: `crates/air/src/prove.rs`
- Operations: `multilinears_linear_combination`, folded evaluations, and related per-point vector ops.
- Why: naturally parallel vector operations and contributes to proving latency.

4. Commitment path (phase 2)
- File: `crates/air/src/prove.rs` (call site) and `whir_p3` commit internals (implementation).
- Operation: `committer.commit(&dft, ...)` and related DFT/hash work.
- Why: likely major cost center, but integration is broader and should follow sumcheck kernels.

## Optimization Strategy

1. Instrument first
- Add timers around proving stages and sumcheck rounds.
- Capture baseline throughput by `log_n_rows`.

2. Keep a CPU fallback
- Introduce feature-gated GPU backend (`gpu` feature).
- Preserve current Rayon path for correctness and portability.

3. Port kernels in order
- Step A: `compute_over_hypercube` map-reduce kernel.
- Step B: multilinear folding kernels.
- Step C: witness linear-combination kernels.
- Step D: commitment/DFT path.

4. Validate correctness at each step
- Deterministic test vectors and proof equivalence checks.
- Per-kernel tolerance policy (exact match where field ops permit it).

## Initial Deliverables

- Baseline benchmark table for current CPU prover.
- Feature-gated kernel abstraction for sumcheck compute path.
- First GPU implementation of hypercube reduction with parity tests against CPU.
- Performance report: wall time, speedup, and memory transfer overhead.

## Out of Scope (for now)

- Verifier-side GPU acceleration.
- Protocol-level changes.
- Aggressive unsafe refactors unrelated to measured bottlenecks.

