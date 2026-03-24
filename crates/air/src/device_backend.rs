use p3_field::{ExtensionField, Field, Packable};
use whir_p3::poly::evals::EvaluationsList;

use crate::kernel_ir::DeviceConstraintProgram;

#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "gpu")]
use p3_field::{BasedVectorSpace, PrimeField32};
#[cfg(feature = "gpu")]
use tracing::warn;

#[cfg(feature = "gpu")]
use crate::kernel_ir::encode_canonical_basis_u32;

#[cfg(all(feature = "gpu", target_os = "macos"))]
mod metal_backend;
#[cfg(feature = "gpu")]
mod wgpu_backend;

const INLINE_COLUMN_WIDTH: usize = 64;

#[cfg(feature = "gpu")]
const MIN_POINTS_FOR_GPU: usize = 0;
#[cfg(feature = "gpu")]
const MAX_DEVICE_OPCODES: usize = 2_048;
#[cfg(feature = "gpu")]
const MAX_EXTENSION_DEGREE: usize = 8;

#[cfg(feature = "gpu")]
pub(crate) fn evaluate_packed_main_pairs<F, EF>(
    program: &DeviceConstraintProgram<F>,
    pols: &[EvaluationsList<F>],
    batching_scalars: &[EF],
    eq_mle: Option<&EvaluationsList<EF>>,
) -> EF
where
    F: Field + Packable + PrimeField32,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
{
    dispatch::evaluate_packed_main_pairs(program, pols, batching_scalars, eq_mle)
}

#[cfg(not(feature = "gpu"))]
pub(crate) fn evaluate_packed_main_pairs<F, EF>(
    program: &DeviceConstraintProgram<F>,
    pols: &[EvaluationsList<F>],
    batching_scalars: &[EF],
    eq_mle: Option<&EvaluationsList<EF>>,
) -> EF
where
    F: Field + Packable,
    EF: ExtensionField<F>,
{
    dispatch::evaluate_packed_main_pairs(program, pols, batching_scalars, eq_mle)
}

pub(crate) fn evaluate_extension_main_pairs<F, EF>(
    program: &DeviceConstraintProgram<F>,
    pols: &[EvaluationsList<EF>],
    batching_scalars: &[EF],
    eq_mle: Option<&EvaluationsList<EF>>,
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    dispatch::evaluate_extension_main_pairs(program, pols, batching_scalars, eq_mle)
}

mod cpu {
    use super::*;
    use p3_field::{PackedFieldExtension, PackedValue};
    use rayon::prelude::*;
    use smallvec::SmallVec;

    pub(super) fn evaluate_packed_main_pairs<F, EF>(
        program: &DeviceConstraintProgram<F>,
        pols: &[EvaluationsList<F>],
        batching_scalars: &[EF],
        eq_mle: Option<&EvaluationsList<EF>>,
    ) -> EF
    where
        F: Field + Packable,
        EF: ExtensionField<F>,
    {
        assert!(pols
            .iter()
            .all(|p| p.num_variables() == pols[0].num_variables()));
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
                let mut local = SmallVec::<[F::Packing; INLINE_COLUMN_WIDTH]>::with_capacity(width);
                let mut next = SmallVec::<[F::Packing; INLINE_COLUMN_WIDTH]>::with_capacity(width);
                local.extend(packed_pols[..width].iter().map(|poly| poly[chunk_idx]));
                next.extend(packed_pols[width..].iter().map(|poly| poly[chunk_idx]));
                let flat_inputs = program.input_layout.flatten_main_pair(
                    &local,
                    &next,
                    packed_zero,
                    packed_zero,
                    packed_zero,
                );
                let mut value = program.evaluate_batched_packed(&flat_inputs, batching_scalars);
                if let Some(eq_mle) = eq_mle {
                    let start = chunk_idx * F::Packing::WIDTH;
                    let end = start + F::Packing::WIDTH;
                    value *= EF::ExtensionPacking::from_ext_slice(&eq_mle.evals()[start..end]);
                }
                EF::ExtensionPacking::to_ext_iter(std::iter::once(value)).sum::<EF>()
            })
            .sum()
    }

    pub(super) fn evaluate_extension_main_pairs<F, EF>(
        program: &DeviceConstraintProgram<F>,
        pols: &[EvaluationsList<EF>],
        batching_scalars: &[EF],
        eq_mle: Option<&EvaluationsList<EF>>,
    ) -> EF
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        assert!(pols
            .iter()
            .all(|p| p.num_variables() == pols[0].num_variables()));
        assert!(
            pols.len().is_multiple_of(2),
            "compiled AIR evaluator expects local/next column pairs",
        );

        let width = pols.len() / 2;

        (0..pols[0].num_evals())
            .into_par_iter()
            .map(|idx| {
                let mut local = SmallVec::<[EF; INLINE_COLUMN_WIDTH]>::with_capacity(width);
                let mut next = SmallVec::<[EF; INLINE_COLUMN_WIDTH]>::with_capacity(width);
                local.extend(pols[..width].iter().map(|poly| poly.evals()[idx]));
                next.extend(pols[width..].iter().map(|poly| poly.evals()[idx]));
                let flat_inputs = program.input_layout.flatten_main_pair(
                    &local,
                    &next,
                    EF::ZERO,
                    EF::ZERO,
                    EF::ZERO,
                );
                let value = program.evaluate_batched_in_extension(&flat_inputs, batching_scalars);
                eq_mle.map_or(value, |eq| value * eq.evals()[idx])
            })
            .sum()
    }
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct GpuKernelHeader {
    pub(super) order: u32,
    pub(super) point_count: u32,
    pub(super) total_slots: u32,
    pub(super) instruction_count: u32,
    pub(super) output_count: u32,
    pub(super) extension_degree: u32,
    pub(super) reserved0: u32,
    pub(super) reserved1: u32,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct GpuInstruction {
    pub(super) opcode: u32,
    pub(super) arg0: u32,
    pub(super) arg1: u32,
    pub(super) reserved: u32,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub(super) struct PackedMainGpuPayload {
    pub(super) header: GpuKernelHeader,
    pub(super) instructions: Vec<GpuInstruction>,
    pub(super) constants: Vec<u32>,
    pub(super) outputs: Vec<u32>,
    pub(super) batching_scalars: Vec<u32>,
    pub(super) flat_inputs: Vec<u32>,
}

#[cfg(feature = "gpu")]
impl PackedMainGpuPayload {
    #[must_use]
    pub(super) fn result_len_u32s(&self) -> usize {
        self.header.point_count as usize * self.header.extension_degree as usize
    }
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuBackendPreference {
    Auto,
    Metal,
    Wgpu,
}

#[cfg(feature = "gpu")]
impl GpuBackendPreference {
    #[must_use]
    fn from_env() -> Self {
        match std::env::var("WHIRLAWAY_GPU_BACKEND")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("metal") => Self::Metal,
            Some("wgpu") => Self::Wgpu,
            _ => Self::Auto,
        }
    }
}

#[cfg(feature = "gpu")]
fn build_packed_main_payload<F, EF>(
    program: &DeviceConstraintProgram<F>,
    pols: &[EvaluationsList<F>],
    batching_scalars: &[EF],
) -> Result<PackedMainGpuPayload, String>
where
    F: Field + PrimeField32,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
{
    assert!(pols
        .iter()
        .all(|p| p.num_variables() == pols[0].num_variables()));
    assert!(
        pols.len().is_multiple_of(2),
        "compiled AIR evaluator expects local/next column pairs",
    );

    let canonical = program.encode_canonical_u32();
    if canonical.opcodes.len() > MAX_DEVICE_OPCODES {
        return Err(format!(
            "device runtime supports at most {MAX_DEVICE_OPCODES} AIR instructions, got {}",
            canonical.opcodes.len(),
        ));
    }
    if EF::DIMENSION > MAX_EXTENSION_DEGREE {
        return Err(format!(
            "device runtime supports extension degree at most {MAX_EXTENSION_DEGREE}, got {}",
            EF::DIMENSION,
        ));
    }

    let total_slots = canonical.input_layout.total_slots();
    let width = pols.len() / 2;
    let expected_main_only_slots = width * 2 + 3;
    if total_slots != expected_main_only_slots {
        return Err(format!(
            "device runtime only supports main-column AIR inputs; expected {expected_main_only_slots} flat slots, got {total_slots}",
        ));
    }

    let point_count = pols[0].num_evals();
    let mut flat_inputs = Vec::with_capacity(point_count * total_slots);
    for point_idx in 0..point_count {
        flat_inputs.extend(
            pols[..width]
                .iter()
                .map(|poly| poly.evals()[point_idx].as_canonical_u32()),
        );
        flat_inputs.extend(
            pols[width..]
                .iter()
                .map(|poly| poly.evals()[point_idx].as_canonical_u32()),
        );
        flat_inputs.extend_from_slice(&[0, 0, 0]);
    }

    let instructions = canonical
        .opcodes
        .iter()
        .zip(&canonical.arg0)
        .zip(&canonical.arg1)
        .map(|((&opcode, &arg0), &arg1)| GpuInstruction {
            opcode: opcode as u32,
            arg0,
            arg1,
            reserved: 0,
        })
        .collect::<Vec<_>>();

    Ok(PackedMainGpuPayload {
        header: GpuKernelHeader {
            order: canonical.field.order,
            point_count: point_count as u32,
            total_slots: total_slots as u32,
            instruction_count: canonical.opcodes.len() as u32,
            output_count: canonical.outputs.len() as u32,
            extension_degree: EF::DIMENSION as u32,
            reserved0: 0,
            reserved1: 0,
        },
        instructions,
        constants: canonical.constants,
        outputs: canonical.outputs,
        batching_scalars: encode_canonical_basis_u32::<F, EF>(batching_scalars),
        flat_inputs,
    })
}

#[cfg(feature = "gpu")]
fn reduce_gpu_output<F, EF>(coefficients: &[u32], eq_mle: Option<&EvaluationsList<EF>>) -> EF
where
    F: Field + PrimeField32,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
{
    assert!(
        coefficients.len().is_multiple_of(EF::DIMENSION),
        "device output length must be divisible by the extension degree",
    );

    match eq_mle {
        Some(eq_mle) => coefficients
            .chunks_exact(EF::DIMENSION)
            .zip(eq_mle.evals())
            .map(|(chunk, &eq)| {
                EF::from_basis_coefficients_iter(chunk.iter().copied().map(F::from_int))
                    .expect("device result chunk should match extension dimension")
                    * eq
            })
            .sum(),
        None => coefficients
            .chunks_exact(EF::DIMENSION)
            .map(|chunk| {
                EF::from_basis_coefficients_iter(chunk.iter().copied().map(F::from_int))
                    .expect("device result chunk should match extension dimension")
            })
            .sum(),
    }
}

#[cfg(feature = "gpu")]
fn execute_packed_main_payload(payload: &PackedMainGpuPayload) -> Result<Vec<u32>, String> {
    match GpuBackendPreference::from_env() {
        GpuBackendPreference::Auto => execute_packed_main_auto(payload),
        GpuBackendPreference::Metal => execute_packed_main_metal(payload),
        GpuBackendPreference::Wgpu => wgpu_backend::execute(payload),
    }
}

#[cfg(feature = "gpu")]
fn execute_packed_main_auto(payload: &PackedMainGpuPayload) -> Result<Vec<u32>, String> {
    let mut errors = Vec::new();

    #[cfg(target_os = "macos")]
    match metal_backend::execute(payload) {
        Ok(result) => return Ok(result),
        Err(err) => errors.push(format!("metal: {err}")),
    }

    match wgpu_backend::execute(payload) {
        Ok(result) => Ok(result),
        Err(err) => {
            errors.push(format!("wgpu: {err}"));
            Err(errors.join("; "))
        }
    }
}

#[cfg(feature = "gpu")]
fn execute_packed_main_metal(payload: &PackedMainGpuPayload) -> Result<Vec<u32>, String> {
    #[cfg(target_os = "macos")]
    {
        metal_backend::execute(payload)
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = payload;
        Err("metal backend is only available on macOS".to_string())
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;

    pub(super) fn evaluate_packed_main_pairs<F, EF>(
        program: &DeviceConstraintProgram<F>,
        pols: &[EvaluationsList<F>],
        batching_scalars: &[EF],
        eq_mle: Option<&EvaluationsList<EF>>,
    ) -> EF
    where
        F: Field + Packable + PrimeField32,
        EF: ExtensionField<F> + BasedVectorSpace<F>,
    {
        let payload = match build_packed_main_payload(program, pols, batching_scalars) {
            Ok(payload) => payload,
            Err(err) => {
                warn!(reason = %err, "gpu payload build failed; falling back to cpu");
                return cpu::evaluate_packed_main_pairs(program, pols, batching_scalars, eq_mle);
            }
        };

        match execute_packed_main_payload(&payload) {
            Ok(coefficients) => reduce_gpu_output::<F, EF>(&coefficients, eq_mle),
            Err(err) => {
                warn!(reason = %err, "gpu runtime failed; falling back to cpu");
                cpu::evaluate_packed_main_pairs(program, pols, batching_scalars, eq_mle)
            }
        }
    }

    // Extension-valued local/next rows still use the host path until the AIR kernel tape
    // grows a serialized extension-input representation.
    pub(super) fn evaluate_extension_main_pairs<F, EF>(
        program: &DeviceConstraintProgram<F>,
        pols: &[EvaluationsList<EF>],
        batching_scalars: &[EF],
        eq_mle: Option<&EvaluationsList<EF>>,
    ) -> EF
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        cpu::evaluate_extension_main_pairs(program, pols, batching_scalars, eq_mle)
    }
}

#[cfg(feature = "gpu")]
mod dispatch {
    use super::*;

    fn should_use_gpu(point_count: usize) -> bool {
        point_count >= MIN_POINTS_FOR_GPU
    }

    pub(super) fn evaluate_packed_main_pairs<F, EF>(
        program: &DeviceConstraintProgram<F>,
        pols: &[EvaluationsList<F>],
        batching_scalars: &[EF],
        eq_mle: Option<&EvaluationsList<EF>>,
    ) -> EF
    where
        F: Field + Packable + PrimeField32,
        EF: ExtensionField<F> + BasedVectorSpace<F>,
    {
        if should_use_gpu(pols.first().map_or(0, EvaluationsList::num_evals)) {
            gpu::evaluate_packed_main_pairs(program, pols, batching_scalars, eq_mle)
        } else {
            cpu::evaluate_packed_main_pairs(program, pols, batching_scalars, eq_mle)
        }
    }

    pub(super) fn evaluate_extension_main_pairs<F, EF>(
        program: &DeviceConstraintProgram<F>,
        pols: &[EvaluationsList<EF>],
        batching_scalars: &[EF],
        eq_mle: Option<&EvaluationsList<EF>>,
    ) -> EF
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        if should_use_gpu(pols.first().map_or(0, EvaluationsList::num_evals)) {
            gpu::evaluate_extension_main_pairs(program, pols, batching_scalars, eq_mle)
        } else {
            cpu::evaluate_extension_main_pairs(program, pols, batching_scalars, eq_mle)
        }
    }
}

#[cfg(not(feature = "gpu"))]
mod dispatch {
    use super::*;

    pub(super) fn evaluate_packed_main_pairs<F, EF>(
        program: &DeviceConstraintProgram<F>,
        pols: &[EvaluationsList<F>],
        batching_scalars: &[EF],
        eq_mle: Option<&EvaluationsList<EF>>,
    ) -> EF
    where
        F: Field + Packable,
        EF: ExtensionField<F>,
    {
        cpu::evaluate_packed_main_pairs(program, pols, batching_scalars, eq_mle)
    }

    pub(super) fn evaluate_extension_main_pairs<F, EF>(
        program: &DeviceConstraintProgram<F>,
        pols: &[EvaluationsList<EF>],
        batching_scalars: &[EF],
        eq_mle: Option<&EvaluationsList<EF>>,
    ) -> EF
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        cpu::evaluate_extension_main_pairs(program, pols, batching_scalars, eq_mle)
    }
}
