use ::air::{
    AirSettings,
    kernel_ir::{ConstraintProgramStats, compile_air_constraint_program},
};
use air::table::AirTable;
use p3_challenger::DuplexChallenger;
use p3_field::PrimeField64;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear, Poseidon2KoalaBear};
use p3_matrix::Matrix;
use p3_poseidon2_air::{Poseidon2Air, RoundConstants, generate_trace_rows};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::fmt;
use std::time::{Duration, Instant};
use tracing::level_filters::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use whir_p3::{
    fiat_shamir::domain_separator::DomainSeparator, parameters::FoldingFactor,
    whir::parameters::WhirConfig,
};

// Koalabear
type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;

type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>; // leaf hashing
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>; // 2-to-1 compression
type MyChallenger = DuplexChallenger<F, Poseidon16, 16, 8>;

// Koalabear
type F = KoalaBear;
type EF = BinomialExtensionField<F, 8>;
type LinearLayers = GenericPoseidon2LinearLayersKoalaBear;
const SBOX_DEGREE: u64 = 3;
const SBOX_REGISTERS: usize = 0;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS: usize = 20;

// BabyBear
// type F = BabyBear;
// type EF = BinomialExtensionField<F, 4>;
// type LinearLayers = GenericPoseidon2LinearLayersBabyBear;
// const SBOX_DEGREE: u64 = 7;
// const SBOX_REGISTERS: usize = 1;
// const HALF_FULL_ROUNDS: usize = 4;
// const PARTIAL_ROUNDS: usize = 13;

const WIDTH: usize = 16;

#[derive(Clone, Debug)]
pub struct Poseidon2Benchmark {
    pub log_n_rows: usize,
    pub settings: AirSettings,
    pub prover_time: Duration,
    pub verifier_time: Duration,
    pub proof_size: f64, // in bytes
}

impl fmt::Display for Poseidon2Benchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Security level: {} bits ({:?}), starting rate: 1/{}, folding factor: {}",
            self.settings.security_bits,
            self.settings.whir_soudness_type,
            1 << self.settings.whir_log_inv_rate,
            match self.settings.whir_folding_factor {
                FoldingFactor::Constant(factor) => format!("{factor}"),
                FoldingFactor::ConstantFromSecondRound(first, then) =>
                    format!("1st: {first} then {then}"),
            }
        )?;
        let n_rows = 1 << self.log_n_rows;
        writeln!(
            f,
            "Proved {} poseidon2 hashes in {:.3} s ({} / s)",
            n_rows,
            self.prover_time.as_millis() as f64 / 1000.0,
            (n_rows as f64 / self.prover_time.as_secs_f64()).round() as usize
        )?;
        writeln!(f, "Proof size: {:.1} KiB", self.proof_size / 1024.0)?;
        writeln!(f, "Verification: {} ms", self.verifier_time.as_millis())
    }
}

pub fn prove_poseidon2(
    log_n_rows: usize,
    settings: AirSettings,
    n_preprocessed_columns: usize,
    display_logs: bool,
) -> Poseidon2Benchmark {
    run_poseidon2(
        log_n_rows,
        settings,
        n_preprocessed_columns,
        display_logs,
        true,
    )
}

pub fn poseidon2_kernel_program_stats() -> ConstraintProgramStats {
    let mut rng = StdRng::seed_from_u64(0);
    let constants =
        RoundConstants::<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>::from_rng(&mut rng);

    let poseidon_air = Poseidon2Air::<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >::new(constants);

    compile_air_constraint_program::<F, _>(&poseidon_air, 0, 0).stats()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use air::kernel_ir::{ConstraintProgramInputs, KernelInput, RowSelector};
    use p3_uni_stark::{Entry, SymbolicExpression, get_symbolic_constraints};

    use super::*;

    fn evaluate_symbolic_expression(
        expr: &SymbolicExpression<F>,
        inputs: &ConstraintProgramInputs<'_, F>,
        memo: &mut HashMap<usize, F>,
    ) -> F {
        let key = expr as *const SymbolicExpression<F> as usize;
        if let Some(value) = memo.get(&key) {
            return *value;
        }

        let value = match expr {
            SymbolicExpression::Variable(var) => inputs.read(match var.entry {
                Entry::Main { offset } => KernelInput::Main {
                    row_offset: offset,
                    column: var.index,
                },
                Entry::Preprocessed { offset } => KernelInput::Preprocessed {
                    row_offset: offset,
                    column: var.index,
                },
                Entry::Permutation { offset } => KernelInput::Permutation {
                    row_offset: offset,
                    column: var.index,
                },
                Entry::Public => KernelInput::Public { index: var.index },
                Entry::Challenge => KernelInput::Challenge { index: var.index },
            }),
            SymbolicExpression::IsFirstRow => {
                inputs.read(KernelInput::Selector(RowSelector::First))
            }
            SymbolicExpression::IsLastRow => inputs.read(KernelInput::Selector(RowSelector::Last)),
            SymbolicExpression::IsTransition => {
                inputs.read(KernelInput::Selector(RowSelector::Transition))
            }
            SymbolicExpression::Constant(value) => *value,
            SymbolicExpression::Add { x, y, .. } => {
                evaluate_symbolic_expression(x.as_ref(), inputs, memo)
                    + evaluate_symbolic_expression(y.as_ref(), inputs, memo)
            }
            SymbolicExpression::Sub { x, y, .. } => {
                evaluate_symbolic_expression(x.as_ref(), inputs, memo)
                    - evaluate_symbolic_expression(y.as_ref(), inputs, memo)
            }
            SymbolicExpression::Neg { x, .. } => {
                -evaluate_symbolic_expression(x.as_ref(), inputs, memo)
            }
            SymbolicExpression::Mul { x, y, .. } => {
                evaluate_symbolic_expression(x.as_ref(), inputs, memo)
                    * evaluate_symbolic_expression(y.as_ref(), inputs, memo)
            }
        };

        memo.insert(key, value);
        value
    }

    fn evaluate_symbolic_constraints(
        constraints: &[SymbolicExpression<F>],
        inputs: &ConstraintProgramInputs<'_, F>,
    ) -> Vec<F> {
        let mut memo = HashMap::new();
        constraints
            .iter()
            .map(|constraint| evaluate_symbolic_expression(constraint, inputs, &mut memo))
            .collect()
    }

    #[test]
    fn poseidon2_constraints_match_current_gpu_candidate_subset() {
        let mut rng = StdRng::seed_from_u64(0);
        let constants =
            RoundConstants::<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>::from_rng(&mut rng);

        let poseidon_air = Poseidon2Air::<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >::new(constants);

        let program = compile_air_constraint_program::<F, _>(&poseidon_air, 0, 0);
        let stats = program.stats();

        assert!(program.is_gpu_candidate());
        assert_eq!(stats.constraints, 148);
        assert_eq!(stats.unsupported_features, 0);
        assert!(stats.muls > 0);
    }

    #[test]
    fn poseidon2_kernel_program_matches_symbolic_constraints_on_trace_rows() {
        let mut rng = StdRng::seed_from_u64(0);
        let constants =
            RoundConstants::<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>::from_rng(&mut rng);

        let poseidon_air = Poseidon2Air::<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >::new(constants.clone());

        let inputs: Vec<[F; WIDTH]> = (0..4)
            .map(|_| std::array::from_fn(|_| rng.random()))
            .collect();

        let trace = generate_trace_rows::<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(inputs, &constants, 0);

        let symbolic_constraints = get_symbolic_constraints(&poseidon_air, 0, 0);
        let program = compile_air_constraint_program::<F, _>(&poseidon_air, 0, 0);

        for row_index in 0..trace.height() {
            let local = trace.row_slice(row_index).expect("row is in bounds");
            let next = trace
                .row_slice((row_index + 1) % trace.height())
                .expect("row is in bounds");
            let inputs =
                ConstraintProgramInputs::for_main_row(&local, &next, row_index, trace.height());

            let expected = evaluate_symbolic_constraints(&symbolic_constraints, &inputs);
            let actual = program.evaluate(&inputs);

            assert_eq!(actual, expected, "constraint mismatch on row {row_index}");
        }
    }
}

pub fn prove_poseidon2_prover_only(
    log_n_rows: usize,
    settings: AirSettings,
    n_preprocessed_columns: usize,
    display_logs: bool,
) -> Poseidon2Benchmark {
    run_poseidon2(
        log_n_rows,
        settings,
        n_preprocessed_columns,
        display_logs,
        false,
    )
}

fn run_poseidon2(
    log_n_rows: usize,
    settings: AirSettings,
    n_preprocessed_columns: usize,
    display_logs: bool,
    verify_proof: bool,
) -> Poseidon2Benchmark {
    if display_logs {
        let env_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();

        Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .init();
    }

    let n_rows = 1 << log_n_rows;

    let mut rng = StdRng::seed_from_u64(0);
    let constants =
        RoundConstants::<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>::from_rng(&mut rng);

    let poseidon_air = Poseidon2Air::<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >::new(constants.clone());

    let inputs: Vec<[F; WIDTH]> = (0..n_rows)
        .map(|_| std::array::from_fn(|_| rng.random()))
        .collect();

    let witness_matrix = generate_trace_rows::<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >(inputs, &constants, 0)
    .transpose();

    let mut witness = witness_matrix
        .rows()
        .map(|col| whir_p3::poly::evals::EvaluationsList::new(col.collect()))
        .collect::<Vec<_>>();

    let preprocessed_columns = witness.drain(..n_preprocessed_columns).collect::<Vec<_>>();

    let table = AirTable::<F, EF, _>::new(
        poseidon_air,
        log_n_rows,
        settings.univariate_skips,
        preprocessed_columns,
        3,
    );

    let poseidon16 = Poseidon16::new_from_rng_128(&mut rng);
    let poseidon24 = Poseidon24::new_from_rng_128(&mut rng);
    let merkle_hash = MerkleHash::new(poseidon24);
    let merkle_compress = MerkleCompress::new(poseidon16.clone());

    let t = Instant::now();

    let whir_params: WhirConfig<_, _, _, _, MyChallenger> =
        table.build_whir_params(&settings, merkle_hash.clone(), merkle_compress.clone());
    let mut domainsep: DomainSeparator<EF, F> = DomainSeparator::new(vec![]);
    domainsep.commit_statement::<_, _, _, 8>(&whir_params);
    domainsep.add_whir_proof::<_, _, _, 8>(&whir_params);

    let challenger = MyChallenger::new(poseidon16);

    let mut prover_state = domainsep.to_prover_state(challenger.clone());

    table.prove(
        &settings,
        merkle_hash.clone(),
        merkle_compress.clone(),
        &mut prover_state,
        witness,
    );
    // let proof_size = prover_state.narg_string().len();

    let prover_time = t.elapsed();
    let verifier_time = if verify_proof {
        let time = Instant::now();
        let mut verifier_state =
            domainsep.to_verifier_state(prover_state.proof_data().to_vec(), challenger);

        table
            .verify(
                &settings,
                merkle_hash,
                merkle_compress,
                &mut verifier_state,
                log_n_rows,
            )
            .unwrap();
        time.elapsed()
    } else {
        Duration::ZERO
    };

    let proof_size = prover_state.proof_data().len() as f64 * (F::ORDER_U64 as f64).log2() / 8.0;

    Poseidon2Benchmark {
        log_n_rows,
        settings,
        prover_time,
        verifier_time,
        proof_size,
    }
}
