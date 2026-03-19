use p3_field::{ExtensionField, Field};
use tracing::instrument;
use utils::{add_multilinears, multilinears_linear_combination};
use whir_p3::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

use crate::utils::{column_down, column_up};

#[cfg(feature = "gpu")]
use p3_field::{Packable, PackedFieldExtension, PackedValue};
#[cfg(feature = "gpu")]
use rayon::prelude::*;
#[cfg(feature = "gpu")]
use sumcheck::HypercubeEvaluator;
#[cfg(feature = "gpu")]
use crate::kernel_ir::{ConstraintProgramInputs, LoweredConstraintProgram, RowPair};

pub(crate) struct BatchedWitness<EF> {
    pub(crate) batched_column: EvaluationsList<EF>,
    pub(crate) sub_evals: EvaluationsList<EF>,
    pub(crate) inner_sum: EF,
}

#[cfg(feature = "gpu")]
pub(crate) struct ConstraintProgramHypercubeEvaluator<'a, F> {
    program: &'a LoweredConstraintProgram<F>,
}

#[cfg(feature = "gpu")]
impl<'a, F> ConstraintProgramHypercubeEvaluator<'a, F> {
    #[must_use]
    pub(crate) const fn new(program: &'a LoweredConstraintProgram<F>) -> Self {
        Self { program }
    }
}

#[cfg(feature = "gpu")]
pub(crate) struct ConstraintProgramExtensionHypercubeEvaluator<'a, F> {
    program: &'a LoweredConstraintProgram<F>,
}

#[cfg(feature = "gpu")]
impl<'a, F> ConstraintProgramExtensionHypercubeEvaluator<'a, F> {
    #[must_use]
    pub(crate) const fn new(program: &'a LoweredConstraintProgram<F>) -> Self {
        Self { program }
    }
}

#[cfg(feature = "gpu")]
impl<F, EF, A> HypercubeEvaluator<F, F, EF, A> for ConstraintProgramHypercubeEvaluator<'_, F>
where
    F: Field + Packable,
    EF: ExtensionField<F>,
    A: Sync,
{
    fn compute(
        &self,
        pols: &[EvaluationsList<F>],
        _computation: &A,
        batching_scalars: &[EF],
        eq_mle: Option<&EvaluationsList<EF>>,
    ) -> EF {
        assert!(
            pols.iter()
                .all(|p| p.num_variables() == pols[0].num_variables())
        );
        assert!(
            pols.len().is_multiple_of(2),
            "compiled AIR evaluator expects local/next column pairs",
        );

        let n_chunks = pols[0].num_evals() / F::Packing::WIDTH;
        let width = pols.len() / 2;
        let packed_pols = pols
            .iter()
            .map(|poly| F::Packing::pack_slice(poly.evals()))
            .collect::<Vec<_>>();
        let packed_zero = F::Packing::from_fn(|_| F::ZERO);

        (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let mut local = Vec::with_capacity(width);
                let mut next = Vec::with_capacity(width);
                local.extend(packed_pols[..width].iter().map(|poly| poly[chunk_idx]));
                next.extend(packed_pols[width..].iter().map(|poly| poly[chunk_idx]));

                let inputs = ConstraintProgramInputs {
                    main: RowPair::new(&local, &next),
                    preprocessed: RowPair::empty(),
                    permutation: RowPair::empty(),
                    public_values: &[],
                    challenges: &[],
                    is_first_row: packed_zero,
                    is_last_row: packed_zero,
                    is_transition: packed_zero,
                };
                let flat_inputs = self.program.input_layout.flatten_inputs(&inputs);
                let mut value = self
                    .program
                    .evaluate_batched_packed(&flat_inputs, batching_scalars);
                if let Some(eq_mle) = eq_mle {
                    let start = chunk_idx * F::Packing::WIDTH;
                    let end = start + F::Packing::WIDTH;
                    value *= EF::ExtensionPacking::from_ext_slice(&eq_mle.evals()[start..end]);
                }
                EF::ExtensionPacking::to_ext_iter(std::iter::once(value)).sum::<EF>()
            })
            .sum()
    }
}

#[cfg(feature = "gpu")]
impl<F, EF, A> HypercubeEvaluator<F, EF, EF, A>
    for ConstraintProgramExtensionHypercubeEvaluator<'_, F>
where
    F: Field,
    EF: ExtensionField<F>,
    A: Sync,
{
    fn compute(
        &self,
        pols: &[EvaluationsList<EF>],
        _computation: &A,
        batching_scalars: &[EF],
        eq_mle: Option<&EvaluationsList<EF>>,
    ) -> EF {
        assert!(
            pols.iter()
                .all(|p| p.num_variables() == pols[0].num_variables())
        );
        assert!(
            pols.len().is_multiple_of(2),
            "compiled AIR evaluator expects local/next column pairs",
        );

        let width = pols.len() / 2;

        (0..pols[0].num_evals())
            .into_par_iter()
            .map(|idx| {
                let mut local = Vec::with_capacity(width);
                let mut next = Vec::with_capacity(width);
                local.extend(pols[..width].iter().map(|poly| poly.evals()[idx]));
                next.extend(pols[width..].iter().map(|poly| poly.evals()[idx]));
                let inputs = ConstraintProgramInputs::new(&local, &next, EF::ZERO, EF::ZERO, EF::ZERO);
                let flat_inputs = self.program.input_layout.flatten_inputs(&inputs);
                let value = self
                    .program
                    .evaluate_batched_in_extension(&flat_inputs, batching_scalars);
                eq_mle.map_or(value, |eq| value * eq.evals()[idx])
            })
            .sum()
    }
}

#[instrument(name = "air: prepare_batched_witness", skip_all)]
pub(crate) fn prepare_batched_witness<F: Field, EF: ExtensionField<F>>(
    witness: &[EvaluationsList<F>],
    columns_batching_scalars: &[EF],
    alpha: EF,
    zerocheck_tail: &[EF],
    epsilons: &[EF],
) -> BatchedWitness<EF> {
    dispatch::prepare_batched_witness(
        witness,
        columns_batching_scalars,
        alpha,
        zerocheck_tail,
        epsilons,
    )
}

mod cpu {
    use super::*;

    pub(super) fn prepare_batched_witness<F: Field, EF: ExtensionField<F>>(
        witness: &[EvaluationsList<F>],
        columns_batching_scalars: &[EF],
        alpha: EF,
        zerocheck_tail: &[EF],
        epsilons: &[EF],
    ) -> BatchedWitness<EF> {
        let batched_column = multilinears_linear_combination(
            witness,
            &EvaluationsList::eval_eq(columns_batching_scalars).evals()[..witness.len()],
        );

        let zerocheck_tail = MultilinearPoint(zerocheck_tail.to_vec());

        // Fold the shifted columns before mixing them. This keeps the expensive add/scale
        // work on the smaller `2^univariate_skips` domain instead of the full witness column.
        let up_folded = column_up(&batched_column).fold(&zerocheck_tail);
        let down_folded = column_down(&batched_column).fold(&zerocheck_tail);
        let sub_evals = add_multilinears(&up_folded, &down_folded.scale(alpha));
        let inner_sum = sub_evals.evaluate(&MultilinearPoint(epsilons.to_vec()));

        BatchedWitness {
            batched_column,
            sub_evals,
            inner_sum,
        }
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;

    // Placeholder until witness-side kernels move to a real device backend.
    pub(super) fn prepare_batched_witness<F: Field, EF: ExtensionField<F>>(
        witness: &[EvaluationsList<F>],
        columns_batching_scalars: &[EF],
        alpha: EF,
        zerocheck_tail: &[EF],
        epsilons: &[EF],
    ) -> BatchedWitness<EF> {
        cpu::prepare_batched_witness(
            witness,
            columns_batching_scalars,
            alpha,
            zerocheck_tail,
            epsilons,
        )
    }
}

#[cfg(feature = "gpu")]
mod dispatch {
    use super::*;

    const MIN_VARS_FOR_GPU: usize = 0;

    fn should_use_gpu(log_length: usize) -> bool {
        log_length >= MIN_VARS_FOR_GPU
    }

    pub(super) fn prepare_batched_witness<F: Field, EF: ExtensionField<F>>(
        witness: &[EvaluationsList<F>],
        columns_batching_scalars: &[EF],
        alpha: EF,
        zerocheck_tail: &[EF],
        epsilons: &[EF],
    ) -> BatchedWitness<EF> {
        if should_use_gpu(witness.first().map_or(0, EvaluationsList::num_variables)) {
            gpu::prepare_batched_witness(
                witness,
                columns_batching_scalars,
                alpha,
                zerocheck_tail,
                epsilons,
            )
        } else {
            cpu::prepare_batched_witness(
                witness,
                columns_batching_scalars,
                alpha,
                zerocheck_tail,
                epsilons,
            )
        }
    }
}

#[cfg(not(feature = "gpu"))]
mod dispatch {
    use super::*;

    pub(super) fn prepare_batched_witness<F: Field, EF: ExtensionField<F>>(
        witness: &[EvaluationsList<F>],
        columns_batching_scalars: &[EF],
        alpha: EF,
        zerocheck_tail: &[EF],
        epsilons: &[EF],
    ) -> BatchedWitness<EF> {
        cpu::prepare_batched_witness(
            witness,
            columns_batching_scalars,
            alpha,
            zerocheck_tail,
            epsilons,
        )
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "gpu")]
    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    #[cfg(feature = "gpu")]
    use p3_field::PrimeCharacteristicRing;
    #[cfg(feature = "gpu")]
    use p3_matrix::Matrix;

    use super::*;
    #[cfg(feature = "gpu")]
    use crate::kernel_ir::{ConstraintProgramInputs, compile_air_constraint_program};

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 8>;

    #[cfg(feature = "gpu")]
    struct ArithmeticAir;

    #[cfg(feature = "gpu")]
    impl BaseAir<F> for ArithmeticAir {
        fn width(&self) -> usize {
            2
        }
    }

    #[cfg(feature = "gpu")]
    impl<AB> Air<AB> for ArithmeticAir
    where
        AB: AirBuilder<F = F>,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.row_slice(0).expect("matrix is empty");
            let next = main.row_slice(1).expect("matrix is empty");
            builder.assert_zero(local[0].clone() * local[1].clone() + next[0].clone() - F::ONE);
        }
    }

    fn base(value: u32) -> F {
        F::new(value)
    }

    fn ext(value: u32) -> EF {
        EF::from(base(value))
    }

    fn base_poly(values: &[u32]) -> EvaluationsList<F> {
        EvaluationsList::new(values.iter().copied().map(base).collect())
    }

    #[test]
    fn prepared_batched_witness_matches_previous_path() {
        let witness = vec![
            base_poly(&[1, 2, 3, 4, 5, 6, 7, 8]),
            base_poly(&[9, 10, 11, 12, 13, 14, 15, 16]),
            base_poly(&[17, 18, 19, 20, 21, 22, 23, 24]),
        ];
        let columns_batching_scalars = vec![ext(3), ext(5)];
        let alpha = ext(7);
        let zerocheck_tail = vec![ext(11)];
        let epsilons = vec![ext(13), ext(17)];

        let expected_batched_column = multilinears_linear_combination(
            &witness,
            &EvaluationsList::eval_eq(&columns_batching_scalars).evals()[..witness.len()],
        );
        let expected_mixed = add_multilinears(
            &column_up(&expected_batched_column),
            &column_down(&expected_batched_column).scale(alpha),
        );
        let expected_sub_evals = expected_mixed.fold(&MultilinearPoint(zerocheck_tail.clone()));
        let expected_inner_sum = expected_mixed.evaluate(&MultilinearPoint(
            [epsilons.clone(), zerocheck_tail.clone()].concat(),
        ));

        let actual = prepare_batched_witness(
            &witness,
            &columns_batching_scalars,
            alpha,
            &zerocheck_tail,
            &epsilons,
        );

        assert_eq!(
            actual.batched_column.evals(),
            expected_batched_column.evals()
        );
        assert_eq!(actual.sub_evals.evals(), expected_sub_evals.evals());
        assert_eq!(actual.inner_sum, expected_inner_sum);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn compiled_constraint_program_evaluator_matches_scalar_sum() {
        let program = compile_air_constraint_program::<F, _>(&ArithmeticAir, 0, 0);
        let lowered_program = program.lower();
        let evaluator = ConstraintProgramHypercubeEvaluator::new(&lowered_program);
        let pols = vec![
            base_poly(&[2, 3, 4, 5]),
            base_poly(&[6, 7, 8, 9]),
            base_poly(&[10, 11, 12, 13]),
            base_poly(&[14, 15, 16, 17]),
        ];
        let batching_scalars = vec![ext(19)];
        let eq_mle = EvaluationsList::new(vec![ext(23), ext(29), ext(31), ext(37)]);

        let actual =
            evaluator.compute(&pols, &ArithmeticAir, &batching_scalars, Some(&eq_mle));

        let expected = (0..pols[0].num_evals())
            .map(|idx| {
                let local = [pols[0].evals()[idx], pols[1].evals()[idx]];
                let next = [pols[2].evals()[idx], pols[3].evals()[idx]];
                let inputs =
                    ConstraintProgramInputs::new(&local, &next, F::ZERO, F::ZERO, F::ONE);
                program.evaluate_batched(&inputs, &batching_scalars) * eq_mle.evals()[idx]
            })
            .sum::<EF>();

        assert_eq!(actual, expected);
    }
}
