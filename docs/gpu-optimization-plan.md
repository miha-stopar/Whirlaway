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

- Baseline benchmark table for current CPU prover. Status: `benches/poseidon2_benchmarks.rs` now captures prover-only timings from the Poseidon2 example.
- Feature-gated kernel abstraction for sumcheck compute path. Status: landed as a dispatch seam with CPU fallback in `crates/sumcheck/src/backend.rs`.
- Witness-side batching helper for AIR prover. Status: landed as `crates/air/src/backend.rs`, including a smaller-domain combine path for `sub_evals` and `inner_sum`.
- Feature-gated AIR hypercube evaluator based on compiled constraint IR. Status: landed for the outer zerocheck path under the `gpu` feature, via `sumcheck::prove_with_hypercube_evaluator` and `air::kernel_ir`.
- First GPU implementation of hypercube reduction with parity tests against CPU.
- Performance report: wall time, speedup, and memory transfer overhead.

## Current Baseline

- Date: 2026-03-17
- Command: `cargo bench --bench poseidon2_benchmarks -- --noplot`
- Scope: prover-only timing from the Poseidon2 example, with `security_bits = 128`, `log_inv_rate = 1`, `univariate_skips = 4`, and no preprocessed columns.
- `log_n_rows = 16`: `time = [1.3415 s, 1.3778 s, 1.4181 s]`, roughly `47.6k` hashes/s at the median.
- `log_n_rows = 18`: `time = [7.3457 s, 8.2344 s, 9.2595 s]`, roughly `31.8k` hashes/s at the median.
- Note: `log_n_rows = 14` was removed from the baseline set because full prove+verify with the same parameters hit `Sumcheck(InvalidRound)` in verification.

## Current Constraint

- Generic `compute_over_hypercube` is still host-only even with the `gpu` feature enabled.
- Reason: the hot loop evaluates an arbitrary Rust `SumcheckComputation`/`Air` callback, which cannot be shipped to a device kernel without a lower-level circuit or expression representation.
- Practical GPU targets remain the fold kernels and witness-side batching path; a real device implementation for hypercube reduction needs a serialized kernel IR or a computation-specific backend.
- Progress: `air::kernel_ir` now compiles AIR constraints into a flat instruction program from Plonky3's symbolic builder, giving a concrete starting point for a computation-specific GPU backend.
- Progress: that IR is now executable on CPU via `ConstraintProgram::evaluate`, so we can validate compiled constraint programs against the existing symbolic AIR path before introducing device kernels.
- Progress: the zerocheck hot loop can now consume that compiled IR through a custom hypercube evaluator, but only when building with `--features gpu`.
- Progress: the GPU-feature evaluator now runs against a lowered flat-input program layout, so the AIR kernel is represented as opcode stream plus input slots rather than repeated `KernelInput` decoding.
- Validation: the Poseidon2 AIR remains a GPU-candidate subset (`148` constraints, `0` unsupported features), and the compiled program now matches the original symbolic constraints on generated trace rows.
- Caveat: using the compiled IR interpreter as the default CPU prover path regressed the `2^16` Poseidon2 benchmark by about 25%, so the new evaluator is currently feature-gated while the host fallback is tuned or replaced with a real device backend.
- Caveat: even after lowering to the flat-input plan, the `gpu`-feature host fallback still measured about `1.86s` at `log_n_rows = 16`, so further host-side interpreter work is unlikely to beat the existing callback path by itself.

## Protocol Reference

The end-to-end protocol walkthrough, concrete Poseidon2 worked example, and comparison with `/Users/miha/projects/csp/miha-whir-p3` now live in [`docs/whirlaway-protocol-walkthrough.md`](./whirlaway-protocol-walkthrough.md).

## Out of Scope (for now)

- Verifier-side GPU acceleration.
- Protocol-level changes.
- Aggressive unsafe refactors unrelated to measured bottlenecks.
