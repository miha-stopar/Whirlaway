use p3_field::{ExtensionField, Field};
use tracing::instrument;
use utils::{add_multilinears, multilinears_linear_combination};
use whir_p3::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

use crate::utils::{column_down, column_up};

pub(crate) struct BatchedWitness<EF> {
    pub(crate) batched_column: EvaluationsList<EF>,
    pub(crate) sub_evals: EvaluationsList<EF>,
    pub(crate) inner_sum: EF,
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
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;

    use super::*;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 8>;

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
}
