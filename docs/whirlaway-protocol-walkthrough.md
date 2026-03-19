# Whirlaway Protocol Walkthrough

This note explains the end-to-end proving flow in Whirlaway, then works through the concrete `log_n_rows = 16` Poseidon2 benchmark so the sumcheck and WHIR sizes are easy to reason about.

## End-to-End Protocol Picture

At a high level, Whirlaway is:

```text
AIR trace
    -> batch the AIR constraints
    -> use sumcheck to reduce "all rows satisfy the AIR" to a few random-point claims
    -> use WHIR as the multilinear PCS that binds those claims to one committed witness polynomial
```

Said differently:

- `sumcheck` is the algebraic reduction layer.
- `whir-p3` is the commitment/opening layer.
- Whirlaway is the AIR-specific glue that turns a trace table into one multilinear polynomial, runs AIR zerochecks, then hands the final opening claim to WHIR.

## Two Different "k" Parameters

There are two unrelated folding knobs in the current Poseidon2 benchmark, and it is easy to mix them up:

| Name | Where it appears | Current benchmark value | What it controls |
|---|---|---:|---|
| `univariate_skips` | AIR zerocheck sumcheck | `4` | Collapse the first `4` row variables of the AIR zerocheck into one larger univariate round. |
| `whir_folding_factor` | WHIR PCS | `ConstantFromSecondRound(7, 4)` | Fold `7` variables in the initial WHIR sumcheck, then `4` variables in each later WHIR round. |

These solve different problems:

- `univariate_skips` speeds up the AIR zerocheck.
- `whir_folding_factor` controls how aggressively the PCS shrinks the committed multilinear polynomial across WHIR rounds.

## Whole Prover Flow

The current prover path in `crates/air/src/prove.rs` is:

```text
Poseidon2 witness table (16 columns x 2^log_n_rows rows)
        |
        | pack all witness columns into one multilinear polynomial P
        v
WHIR commitment to P
        |
        | sample constraint batching randomness
        v
outer AIR zerocheck:
prove the batched AIR residual H is zero on all rows
        |
        | end of zerocheck gives claims for shifted columns:
        |   c_i^up(beta), c_i^down(beta)
        v
batch shifted-column claims into one witness-side claim
        |
        v
inner sumcheck:
reduce shifted-column claims back to one batched evaluation of P
        |
        v
final WHIR opening:
open P at one random point and check it matches the inner-sumcheck claim
```

The verifier mirrors the same Fiat-Shamir transcript:

```text
read WHIR commitment
    -> replay zerocheck verification
    -> replay inner-sumcheck verification
    -> derive one final evaluation constraint on P
    -> ask WHIR to verify that opening
```

## Where Zerocheck Fits

The AIR constraints are local: they talk about the current row and the next row. Whirlaway therefore builds, for every trace column `c`, two shifted copies:

- `c^up`: the "current row" view
- `c^down`: the "next row" view

Then the prover batches the `148` Poseidon2 constraints into one combined polynomial:

```text
H = h_0 + alpha * h_1 + alpha^2 * h_2 + ... + alpha^147 * h_147
```

The core statement becomes:

```text
for every row b in {0,1}^n:
H(c_0^up(b), ..., c_{M-1}^up(b), c_0^down(b), ..., c_{M-1}^down(b)) = 0
```

Zerocheck proves this global statement by running sumcheck on:

```text
sum_{b in {0,1}^n} eq(b, r) * H(...)
```

for a random verifier point `r`.

This is why zerocheck is the hot path for GPU work:

- it iterates over the full row hypercube,
- it evaluates the AIR computation at each point,
- and in the current code that evaluation still happens through an AIR-specific Rust callback.

## Where WHIR Fits

WHIR never sees the original AIR logic. By the time WHIR is used, the AIR argument has already been reduced to:

```text
"the committed multilinear witness polynomial P evaluates to value v at point z"
```

WHIR is responsible for:

- committing to `P`,
- maintaining soundness of the multilinear commitment through its own DFT + Merkle + STIR rounds,
- and verifying the final opening claim.

So the division of labor is:

- AIR layer: "why should the witness satisfy the computation?"
- WHIR layer: "are the claimed witness values really evaluations of the committed multilinear polynomial?"

## Concrete Poseidon2 Example

This section uses the smaller benchmark from `benches/poseidon2_benchmarks.rs`:

- `log_n_rows = 16`
- `security_bits = 128`
- `univariate_skips = 4`
- `whir_folding_factor = ConstantFromSecondRound(7, 4)`
- `whir_log_inv_rate = 1`
- `whir_initial_domain_reduction_factor = 5`
- `n_preprocessed_columns = 0`

The benchmark currently measures about `1.38 s` prover time at the median on this case.

### 1. Witness Table and Packed Polynomial

In `src/examples/poseidon2.rs`, the benchmark constructs:

- `WIDTH = 16` witness columns
- `n_rows = 2^16 = 65,536`
- no preprocessed columns

So the raw committed witness table has:

```text
16 * 65,536 = 1,048,576 = 2^20
```

field elements.

Whirlaway then packs those 16 columns into one multilinear polynomial `P` using `packed_multilinear`:

```text
P : {0,1}^20 -> F
```

with:

- `4` column-selection bits because `log2(16) = 4`
- `16` row-selection bits because there are `2^16` rows

The intended interpretation is:

```text
P(z_3, z_2, z_1, z_0, r_15, ..., r_0)
    = witness_column[z][row]
```

where `z` is the 4-bit column index and `row` is the 16-bit row index.

Because `16` is already a power of two, there is no extra zero-padding in the column dimension for this benchmark. The packed witness polynomial therefore has exactly `2^20` evaluations.

Visualization:

```text
column c0, 2^16 evals
column c1, 2^16 evals
...
column c15, 2^16 evals
        |
        | concatenate in column order
        v
flat evaluation vector of length 16 * 2^16 = 2^20
        |
        v
one multilinear polynomial P on 20 Boolean variables
```

### 2. AIR-Side Zerocheck Sizes

For this benchmark:

- number of witness columns `M' = 16`
- number of shifted columns entering zerocheck = `16 up + 16 down = 32`
- number of batched constraints = `148`
- AIR constraint degree parameter in the table = `3`

The outer zerocheck still reasons over the row hypercube of size `2^16`, but `univariate_skips = 4` changes the first sumcheck step:

- instead of folding one row variable immediately,
- the protocol treats the first `4` row variables as one larger univariate block of size `2^4 = 16`,
- and only the remaining `12` row variables stay Boolean in the tail.

That is why the prover samples:

```text
log_length + 1 - univariate_skips = 16 + 1 - 4 = 13
```

zerocheck challenges:

- `1` challenge for the skipped first block
- `12` more challenges for the remaining Boolean variables

This `univariate_skips = 4` parameter belongs to the AIR zerocheck only. It is not the WHIR folding factor.

### 3. WHIR Variables for the Same Benchmark

The WHIR commitment sees the packed witness polynomial `P` with:

```text
num_variables = log_n_rows + log_n_witness_columns = 16 + 4 = 20
```

This is smaller than the standalone `miha-whir-p3` default benchmark, which starts from `24` variables, because here the committed object is the packed Poseidon2 witness rather than a free-standing random polynomial on `{0,1}^24`.

### 4. Initial WHIR Commitment DFT

The current benchmark uses:

- initial WHIR folding factor `k0 = 7`
- starting inverse rate `rho = 1`, so initial blowup factor `2^rho = 2`

The commitment code first repeats the `2^20` evaluations by a factor of `2`:

```text
2^20 -> 2^21
```

Then it reshapes that repeated vector into a matrix of width `2^k0 = 128` before running the batched DFT:

```text
repeated evaluation vector, length 2^21
        |
        | width = 2^7 = 128
        v
matrix of height 2^(21 - 7) = 2^14 = 16,384
        |
        | one DFT per column
        v
128 DFT streams, each on a two-adic subgroup of size 2^14
```

So the concrete first commitment numbers are:

| Quantity | Value |
|---|---:|
| committed polynomial variables | `20` |
| raw evaluation count | `2^20 = 1,048,576` |
| repeated count after inverse-rate blowup | `2^21 = 2,097,152` |
| first WHIR folding factor `k0` | `7` |
| number of DFT streams | `2^7 = 128` |
| subgroup size per stream | `2^14 = 16,384` |
| total Reed-Solomon codeword size | `128 * 2^14 = 2^21` |

One subtle point in the current code: `EvalsDft::new(2^18)` is a twiddle-cache upper bound, not the actual first FFT size. The actual first commitment DFT for this benchmark is the `128 x 2^14` batched transform above.

### 5. Why the DFT Subgroup Has To Be This Large

The DFT is not just interpolating the raw witness table. It is building a larger Reed-Solomon-style codeword so that WHIR can later do:

- OOD sampling,
- STIR queries,
- Merkle openings,
- and proximity-style checks.

That means there are always two sizes to track:

| Object | Meaning |
|---|---|
| raw multilinear table size | the actual polynomial table, here `2^20` |
| committed domain size | the redundant codeword size used by WHIR, here `2^21` |

The DFT works on the second size, not the first one. The initial subgroup size is large because:

- the protocol intentionally blows the table up by the inverse rate (`2` here),
- and then splits the resulting codeword into `2^k0 = 128` interleaved streams.

### 6. Round-By-Round WHIR Sizes in This Benchmark

For this benchmark, the WHIR folding schedule is:

```text
initial fold: 7 variables
then: 4 variables
then: 4 variables
final direct phase on the remaining 5 variables
```

The first domain reduction factor is aggressive:

```text
whir_initial_domain_reduction_factor = 5
```

so after the initial fold, the RS domain size drops by `2^5 = 32`.

This gives the following concrete WHIR progression:

| Stage | Polynomial vars at that stage | Raw table size | New committed domain size | Effective inverse rate | Next fold width | Per-stream DFT size |
|---|---:|---:|---:|---:|---:|---:|
| Initial commitment | `20` | `2^20` | `2^21` | `2` | `2^7 = 128` | `2^14` |
| After initial WHIR fold | `13` | `2^13` | no new commitment yet | not applicable | not applicable | not applicable |
| Round 0 commitment | `13` | `2^13` | `2^16` | `2^16 / 2^13 = 8` | `2^4 = 16` | `2^12` |
| Round 1 commitment | `9` | `2^9` | `2^15` | `2^15 / 2^9 = 64` | `2^4 = 16` | `2^11` |
| Final direct phase | `5` | `2^5` | no new Merkle layer | previous domain still `2^15` for final checks | no new DFT | no new DFT |

There are only two post-commitment WHIR Merkle/DFT rounds here because the large first fold `7` gets the polynomial down to `13` variables immediately, and two more folds by `4` leave only `5` variables, which is below WHIR's threshold for "send the final polynomial directly".

## Relation To `miha-whir-p3`

The local repository at `/Users/miha/projects/csp/miha-whir-p3` is a standalone WHIR implementation and documentation set. Conceptually, Whirlaway sits on top of the same ideas, but they are not the same codebase or the same API surface.

### Short Version

```text
miha-whir-p3:
    starts from an arbitrary multilinear polynomial and proves PCS consistency for it

Whirlaway:
    starts from an AIR trace,
    reduces AIR correctness to one final multilinear opening claim,
    then uses WHIR as the PCS for that final claim
```

### Side-By-Side Comparison

| Aspect | Whirlaway | `miha-whir-p3` |
|---|---|---|
| Starting object | AIR witness columns plus AIR constraints | A standalone multilinear polynomial plus an initial statement |
| What gets committed | One packed witness polynomial over `(column bits, row bits)` | One user-supplied polynomial over `{0,1}^n` |
| Why sumcheck appears | First to prove AIR zerocheck and shifted-column consistency, then again inside WHIR | Mainly as WHIR's own folding/consistency machinery |
| Role of WHIR | Final multilinear PCS used after AIR reductions | The main protocol itself |
| Preprocessed columns | Not committed; verifier computes them directly | No AIR-specific notion of preprocessed columns |
| Default benchmark shape | `20` committed variables for `log_n_rows = 16` Poseidon2 (`4` column bits + `16` row bits) | Standalone WHIR doc uses `24` variables |
| Default folding schedule in the compared benchmark docs | `7, 4, 4` for the Poseidon2 example here | `4, 4, 4, 4` in `docs/whir-protocol-visualization.md` |
| First domain reduction factor in the compared docs | `5` here | `3` in the standalone WHIR visualization doc |
| Number of Merkle/DFT layers in the compared examples | initial commitment + `2` round commitments | initial commitment + `4` round commitments |

### What Is The Same

These are the core ideas shared by both:

- Commit to multilinear evaluations with a DFT-based Reed-Solomon-style encoding.
- Merkle-commit to the encoded matrix.
- Sample OOD points from the Fiat-Shamir transcript.
- Use query openings and sumcheck-based folding to shrink the polynomial round by round.
- Stop once the folded polynomial is small enough and send it directly.

This is why the DFT/commitment part of Whirlaway looks so close to standalone WHIR: the PCS layer really is WHIR.

### What Whirlaway Adds On Top

Whirlaway adds the AIR-specific reduction layer on top of WHIR:

- packing witness columns into one multilinear polynomial,
- batching many AIR constraints into one residual polynomial,
- zerocheck over row constraints,
- the shifted-column machinery (`up` / `down`),
- and the inner sumcheck that turns shifted-column claims into one final opening claim.

Those pieces are not built into `miha-whir-p3` because that repository implements the generic multilinear commitment/proof machinery, not an AIR proof system. If another project wanted to prove AIR constraints on top of `miha-whir-p3`, it would need to build a similar AIR-specific layer outside the WHIR library.

So, if you mentally separate the protocol into:

```text
AIR reduction layer
    +
multilinear PCS layer
```

then:

- Whirlaway owns the AIR reduction layer,
- `whir-p3` owns the PCS layer.

### Current Dependency vs Local Standalone Repo

This repository currently depends on the upstream `whir-p3` git dependency, not directly on `/Users/miha/projects/csp/miha-whir-p3`. The concepts are aligned, but the APIs differ.

The most visible code-structure differences are:

- The local standalone repo has an explicit `DftBatchLayout` helper and a dedicated `docs/whir-protocol-visualization.md`, which make the matrix reshape formulas very explicit.
- The dependency currently used by Whirlaway keeps the same underlying formulas but expresses them more compactly through `parallel_repeat`, `RowMajorMatrix::new(..., width)`, and the specialized `EvalsDft` implementation.
- The local standalone repo's prover and commitment APIs revolve around `InitialStatement`, `WhirProof`, and explicit round-state objects.
- The dependency used here exposes a more compact path through `CommitmentWriter`, `ProverState`, `Statement`, and `Witness`.

So the cleanest way to think about them is:

- `miha-whir-p3` is the standalone WHIR engine with clearer WHIR-only documentation,
- Whirlaway is the AIR proof system that uses WHIR as its multilinear commitment backend.

### Compared To The Standalone `miha-whir-p3` Benchmark

The standalone visualization doc in `/Users/miha/projects/csp/miha-whir-p3/docs/whir-protocol-visualization.md` uses:

- `num_variables = 24`
- `folding_factor = Constant(4)`
- `starting_log_inv_rate = 1`
- `rs_domain_initial_reduction_factor = 3`

That leads to:

- a larger initial polynomial (`2^24` evals instead of `2^20` here),
- a smaller first fold (`4` instead of `7`),
- and more round commitments (`4` instead of `2` after the initial commitment).

In other words, the Poseidon2 benchmark in Whirlaway is not "a bigger WHIR benchmark"; it is a more structured benchmark where:

- the committed polynomial is smaller,
- the AIR reductions happen before WHIR,
- and the WHIR schedule is tuned to get from `20` variables down to a direct `5`-variable final polynomial quickly.
