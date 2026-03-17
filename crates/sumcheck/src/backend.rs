use std::any::TypeId;

use p3_field::{BasedVectorSpace, ExtensionField, Field, PackedValue};
use rayon::prelude::*;
use smallvec::SmallVec;
use utils::{
    batch_fold_multilinear_in_large_field as batch_fold_multilinear_in_large_field_cpu,
    batch_fold_multilinear_in_small_field as batch_fold_multilinear_in_small_field_cpu,
};
use whir_p3::poly::evals::EvaluationsList;

use crate::{SumcheckComputation, SumcheckComputationPacked, prove::eval_sumcheck_computation};

const INLINE_POINT_WIDTH: usize = 64;

pub(crate) fn batch_fold_multilinear_in_large_field<F: Field, EF: ExtensionField<F>>(
    polys: &[&EvaluationsList<F>],
    scalars: &[EF],
) -> Vec<EvaluationsList<EF>> {
    dispatch::batch_fold_multilinear_in_large_field(polys, scalars)
}

pub(crate) fn batch_fold_multilinear_in_small_field<F: Field, EF: ExtensionField<F>>(
    polys: &[&EvaluationsList<EF>],
    scalars: &[F],
) -> Vec<EvaluationsList<EF>> {
    dispatch::batch_fold_multilinear_in_small_field(polys, scalars)
}

pub(crate) fn compute_over_hypercube<F, NF, EF, SC>(
    pols: &[EvaluationsList<NF>],
    computation: &SC,
    batching_scalars: &[EF],
    eq_mle: Option<&EvaluationsList<EF>>,
) -> EF
where
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
    SC: SumcheckComputation<F, NF, EF> + SumcheckComputationPacked<F, EF>,
{
    dispatch::compute_over_hypercube(pols, computation, batching_scalars, eq_mle)
}

mod cpu {
    use super::*;

    pub(super) fn batch_fold_multilinear_in_large_field<F: Field, EF: ExtensionField<F>>(
        polys: &[&EvaluationsList<F>],
        scalars: &[EF],
    ) -> Vec<EvaluationsList<EF>> {
        batch_fold_multilinear_in_large_field_cpu(polys, scalars)
    }

    pub(super) fn batch_fold_multilinear_in_small_field<F: Field, EF: ExtensionField<F>>(
        polys: &[&EvaluationsList<EF>],
        scalars: &[F],
    ) -> Vec<EvaluationsList<EF>> {
        batch_fold_multilinear_in_small_field_cpu(polys, scalars)
    }

    pub(super) fn compute_over_hypercube<F, NF, EF, SC>(
        pols: &[EvaluationsList<NF>],
        computation: &SC,
        batching_scalars: &[EF],
        eq_mle: Option<&EvaluationsList<EF>>,
    ) -> EF
    where
        F: Field,
        NF: ExtensionField<F>,
        EF: ExtensionField<NF> + ExtensionField<F>,
        SC: SumcheckComputation<F, NF, EF> + SumcheckComputationPacked<F, EF>,
    {
        assert!(
            pols.iter()
                .all(|p| p.num_variables() == pols[0].num_variables())
        );
        let n_vars = pols[0].num_variables();
        if TypeId::of::<NF>() == TypeId::of::<F>() {
            let pols: &[EvaluationsList<F>] = unsafe { std::mem::transmute(pols) };
            let packed_pols = pols
                .iter()
                .map(|p| F::Packing::pack_slice(p.evals()))
                .collect::<Vec<_>>();

            let decomposed_batching_scalars: Vec<_> = (0..<EF as BasedVectorSpace<F>>::DIMENSION)
                .map(|i| {
                    batching_scalars
                        .iter()
                        .map(|x| x.as_basis_coefficients_slice()[i])
                        .collect()
                })
                .collect();

            (0..(1 << n_vars) / F::Packing::WIDTH)
                .into_par_iter()
                .enumerate()
                .map(|(x, i)| {
                    let mut point = SmallVec::<[F::Packing; INLINE_POINT_WIDTH]>::with_capacity(
                        packed_pols.len(),
                    );
                    point.extend(packed_pols.iter().map(|pol| pol[x]));
                    let res = computation.eval_packed(
                        &point,
                        batching_scalars,
                        &decomposed_batching_scalars,
                    );
                    if let Some(eq_mle) = eq_mle {
                        res.enumerate()
                            .map(|(idx_in_packing, res)| {
                                res * eq_mle.evals()[i * F::Packing::WIDTH + idx_in_packing]
                            })
                            .sum()
                    } else {
                        res.sum()
                    }
                })
                .sum()
        } else {
            assert_eq!(TypeId::of::<NF>(), TypeId::of::<EF>());
            (0..1 << n_vars)
                .into_par_iter()
                .map(|x| {
                    let mut point = SmallVec::<[NF; INLINE_POINT_WIDTH]>::with_capacity(pols.len());
                    point.extend(pols.iter().map(|pol| pol.evals()[x]));
                    let eq_mle_eval = eq_mle.map(|p| p.evals()[x]);
                    eval_sumcheck_computation(computation, batching_scalars, &point, eq_mle_eval)
                })
                .sum()
        }
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;

    // Placeholder until a real GPU kernel backend lands. Keeping the module and
    // signatures in place avoids another prover refactor once device kernels exist.
    pub(super) fn batch_fold_multilinear_in_large_field<F: Field, EF: ExtensionField<F>>(
        polys: &[&EvaluationsList<F>],
        scalars: &[EF],
    ) -> Vec<EvaluationsList<EF>> {
        cpu::batch_fold_multilinear_in_large_field(polys, scalars)
    }

    pub(super) fn batch_fold_multilinear_in_small_field<F: Field, EF: ExtensionField<F>>(
        polys: &[&EvaluationsList<EF>],
        scalars: &[F],
    ) -> Vec<EvaluationsList<EF>> {
        cpu::batch_fold_multilinear_in_small_field(polys, scalars)
    }
}

#[cfg(feature = "gpu")]
mod dispatch {
    use super::*;
    use crate::prove::MIN_VARS_FOR_GPU;

    fn should_use_gpu(n_vars: usize) -> bool {
        n_vars >= MIN_VARS_FOR_GPU
    }

    pub(super) fn batch_fold_multilinear_in_large_field<F: Field, EF: ExtensionField<F>>(
        polys: &[&EvaluationsList<F>],
        scalars: &[EF],
    ) -> Vec<EvaluationsList<EF>> {
        if should_use_gpu(polys.first().map_or(0, |poly| poly.num_variables())) {
            gpu::batch_fold_multilinear_in_large_field(polys, scalars)
        } else {
            cpu::batch_fold_multilinear_in_large_field(polys, scalars)
        }
    }

    pub(super) fn batch_fold_multilinear_in_small_field<F: Field, EF: ExtensionField<F>>(
        polys: &[&EvaluationsList<EF>],
        scalars: &[F],
    ) -> Vec<EvaluationsList<EF>> {
        if should_use_gpu(polys.first().map_or(0, |poly| poly.num_variables())) {
            gpu::batch_fold_multilinear_in_small_field(polys, scalars)
        } else {
            cpu::batch_fold_multilinear_in_small_field(polys, scalars)
        }
    }

    pub(super) fn compute_over_hypercube<F, NF, EF, SC>(
        pols: &[EvaluationsList<NF>],
        computation: &SC,
        batching_scalars: &[EF],
        eq_mle: Option<&EvaluationsList<EF>>,
    ) -> EF
    where
        F: Field,
        NF: ExtensionField<F>,
        EF: ExtensionField<NF> + ExtensionField<F>,
        SC: SumcheckComputation<F, NF, EF> + SumcheckComputationPacked<F, EF>,
    {
        let _ = should_use_gpu(pols.first().map_or(0, EvaluationsList::num_variables));
        // `SC` is an arbitrary Rust callback today, so there is no generic way to serialize it
        // into a device program. Keep this on the CPU until sumcheck computations get a
        // GPU-targetable representation.
        cpu::compute_over_hypercube(pols, computation, batching_scalars, eq_mle)
    }
}

#[cfg(not(feature = "gpu"))]
mod dispatch {
    use super::*;

    pub(super) fn batch_fold_multilinear_in_large_field<F: Field, EF: ExtensionField<F>>(
        polys: &[&EvaluationsList<F>],
        scalars: &[EF],
    ) -> Vec<EvaluationsList<EF>> {
        cpu::batch_fold_multilinear_in_large_field(polys, scalars)
    }

    pub(super) fn batch_fold_multilinear_in_small_field<F: Field, EF: ExtensionField<F>>(
        polys: &[&EvaluationsList<EF>],
        scalars: &[F],
    ) -> Vec<EvaluationsList<EF>> {
        cpu::batch_fold_multilinear_in_small_field(polys, scalars)
    }

    pub(super) fn compute_over_hypercube<F, NF, EF, SC>(
        pols: &[EvaluationsList<NF>],
        computation: &SC,
        batching_scalars: &[EF],
        eq_mle: Option<&EvaluationsList<EF>>,
    ) -> EF
    where
        F: Field,
        NF: ExtensionField<F>,
        EF: ExtensionField<NF> + ExtensionField<F>,
        SC: SumcheckComputation<F, NF, EF> + SumcheckComputationPacked<F, EF>,
    {
        cpu::compute_over_hypercube(pols, computation, batching_scalars, eq_mle)
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{PackedValue, extension::BinomialExtensionField};
    use p3_koala_bear::KoalaBear;

    use super::*;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 8>;

    struct PackedLaneSum;

    impl SumcheckComputation<F, F, EF> for PackedLaneSum {
        fn eval(&self, point: &[F], batching_scalars: &[EF]) -> EF {
            EF::from(point[0]) + batching_scalars[0] * point[1]
        }
    }

    impl SumcheckComputationPacked<F, EF> for PackedLaneSum {
        fn eval_packed(
            &self,
            point: &[<F as p3_field::Field>::Packing],
            batching_scalars: &[EF],
            _decomposed_alpha_powers: &[Vec<F>],
        ) -> impl Iterator<Item = EF> + Send + Sync {
            let scalar = batching_scalars[0];
            (0..<F as p3_field::Field>::Packing::WIDTH).map(move |idx| {
                EF::from(point[0].as_slice()[idx]) + scalar * point[1].as_slice()[idx]
            })
        }
    }

    struct ExtensionLaneSum;

    impl SumcheckComputation<F, EF, EF> for ExtensionLaneSum {
        fn eval(&self, point: &[EF], batching_scalars: &[EF]) -> EF {
            point[0] + batching_scalars[0] * point[1]
        }
    }

    impl SumcheckComputationPacked<F, EF> for ExtensionLaneSum {
        fn eval_packed(
            &self,
            _point: &[<F as p3_field::Field>::Packing],
            _batching_scalars: &[EF],
            _decomposed_alpha_powers: &[Vec<F>],
        ) -> impl Iterator<Item = EF> + Send + Sync {
            std::iter::empty()
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

    fn ext_poly(values: &[u32]) -> EvaluationsList<EF> {
        EvaluationsList::new(values.iter().copied().map(ext).collect())
    }

    #[test]
    fn backend_matches_cpu_for_small_field_folds() {
        let polys = [base_poly(&[1, 2, 3, 4]), base_poly(&[5, 6, 7, 8])];
        let refs = polys.iter().collect::<Vec<_>>();
        let scalars = [base(3), base(7)];

        let expected = cpu::batch_fold_multilinear_in_small_field(&refs, &scalars);
        let actual = batch_fold_multilinear_in_small_field(&refs, &scalars);

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.evals(), expected.evals());
        }
    }

    #[test]
    fn backend_matches_cpu_for_large_field_folds() {
        let polys = [base_poly(&[1, 2, 3, 4]), base_poly(&[5, 6, 7, 8])];
        let refs = polys.iter().collect::<Vec<_>>();
        let scalars = [ext(3), ext(7)];

        let expected = cpu::batch_fold_multilinear_in_large_field(&refs, &scalars);
        let actual = batch_fold_multilinear_in_large_field(&refs, &scalars);

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.evals(), expected.evals());
        }
    }

    #[test]
    fn backend_matches_cpu_for_packed_hypercube_reduction() {
        let polys = [base_poly(&[1, 2, 3, 4]), base_poly(&[5, 6, 7, 8])];
        let batching_scalars = [ext(11)];
        let eq_mle = EvaluationsList::eval_eq(&[ext(2), ext(3)]);

        let expected = cpu::compute_over_hypercube::<F, F, EF, _>(
            &polys,
            &PackedLaneSum,
            &batching_scalars,
            Some(&eq_mle),
        );
        let actual = compute_over_hypercube::<F, F, EF, _>(
            &polys,
            &PackedLaneSum,
            &batching_scalars,
            Some(&eq_mle),
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn backend_matches_cpu_for_extension_hypercube_reduction() {
        let polys = [ext_poly(&[1, 2, 3, 4]), ext_poly(&[5, 6, 7, 8])];
        let batching_scalars = [ext(11)];
        let eq_mle = EvaluationsList::eval_eq(&[ext(2), ext(3)]);

        let expected = cpu::compute_over_hypercube::<F, EF, EF, _>(
            &polys,
            &ExtensionLaneSum,
            &batching_scalars,
            Some(&eq_mle),
        );
        let actual = compute_over_hypercube::<F, EF, EF, _>(
            &polys,
            &ExtensionLaneSum,
            &batching_scalars,
            Some(&eq_mle),
        );

        assert_eq!(actual, expected);
    }
}
