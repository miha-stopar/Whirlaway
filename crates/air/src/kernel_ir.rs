use std::collections::HashMap;

use p3_air::Air;
use p3_field::{ExtensionField, Field, PackedValue, PrimeCharacteristicRing};
use p3_uni_stark::{
    Entry, SymbolicAirBuilder, SymbolicExpression, SymbolicVariable, get_symbolic_constraints,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowSelector {
    First,
    Last,
    Transition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelInput {
    Main { row_offset: usize, column: usize },
    Preprocessed { row_offset: usize, column: usize },
    Public { index: usize },
    Challenge { index: usize },
    Permutation { row_offset: usize, column: usize },
    Selector(RowSelector),
}

#[derive(Debug, Clone, Copy)]
pub struct RowPair<'a, F> {
    pub local: &'a [F],
    pub next: &'a [F],
}

impl<'a, F> RowPair<'a, F> {
    #[must_use]
    pub const fn new(local: &'a [F], next: &'a [F]) -> Self {
        Self { local, next }
    }

    #[must_use]
    pub const fn empty() -> Self {
        Self {
            local: &[],
            next: &[],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConstraintProgramInputs<'a, F> {
    pub main: RowPair<'a, F>,
    pub preprocessed: RowPair<'a, F>,
    pub permutation: RowPair<'a, F>,
    pub public_values: &'a [F],
    pub challenges: &'a [F],
    pub is_first_row: F,
    pub is_last_row: F,
    pub is_transition: F,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnsupportedGpuFeature {
    PublicInput,
    ChallengeInput,
    PermutationInput,
    RowSelector(RowSelector),
}

#[derive(Debug, Clone)]
pub enum KernelInstruction<F> {
    Constant(F),
    Input(KernelInput),
    Add { lhs: usize, rhs: usize },
    Sub { lhs: usize, rhs: usize },
    Neg { src: usize },
    Mul { lhs: usize, rhs: usize },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConstraintProgramStats {
    pub constraints: usize,
    pub instructions: usize,
    pub constants: usize,
    pub inputs: usize,
    pub adds: usize,
    pub subs: usize,
    pub negs: usize,
    pub muls: usize,
    pub max_constraint_degree: usize,
    pub unsupported_features: usize,
}

#[derive(Debug, Clone)]
pub struct ConstraintProgram<F> {
    pub instructions: Vec<KernelInstruction<F>>,
    pub outputs: Vec<usize>,
    pub unsupported_features: Vec<UnsupportedGpuFeature>,
    pub max_constraint_degree: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConstraintInputLayout {
    pub main_width: usize,
    pub preprocessed_width: usize,
    pub permutation_width: usize,
    pub public_values: usize,
    pub challenges: usize,
}

#[derive(Debug, Clone)]
pub enum LoweredKernelInstruction<F> {
    Constant(F),
    Input { slot: usize },
    Add { lhs: usize, rhs: usize },
    Sub { lhs: usize, rhs: usize },
    Neg { src: usize },
    Mul { lhs: usize, rhs: usize },
}

#[derive(Debug, Clone)]
pub struct LoweredConstraintProgram<F> {
    pub input_layout: ConstraintInputLayout,
    pub instructions: Vec<LoweredKernelInstruction<F>>,
    pub outputs: Vec<usize>,
    pub unsupported_features: Vec<UnsupportedGpuFeature>,
    pub max_constraint_degree: usize,
}

impl<F> ConstraintProgram<F> {
    #[must_use]
    pub fn is_gpu_candidate(&self) -> bool {
        self.unsupported_features.is_empty()
    }

    #[must_use]
    pub fn stats(&self) -> ConstraintProgramStats {
        let mut stats = ConstraintProgramStats {
            constraints: self.outputs.len(),
            instructions: self.instructions.len(),
            max_constraint_degree: self.max_constraint_degree,
            unsupported_features: self.unsupported_features.len(),
            ..ConstraintProgramStats::default()
        };

        for instruction in &self.instructions {
            match instruction {
                KernelInstruction::Constant(_) => stats.constants += 1,
                KernelInstruction::Input(_) => stats.inputs += 1,
                KernelInstruction::Add { .. } => stats.adds += 1,
                KernelInstruction::Sub { .. } => stats.subs += 1,
                KernelInstruction::Neg { .. } => stats.negs += 1,
                KernelInstruction::Mul { .. } => stats.muls += 1,
            }
        }

        stats
    }
}

impl ConstraintInputLayout {
    #[must_use]
    pub const fn total_slots(&self) -> usize {
        self.selectors_offset() + 3
    }

    #[must_use]
    pub const fn main_local_offset(&self) -> usize {
        0
    }

    #[must_use]
    pub const fn main_next_offset(&self) -> usize {
        self.main_local_offset() + self.main_width
    }

    #[must_use]
    pub const fn preprocessed_local_offset(&self) -> usize {
        self.main_next_offset() + self.main_width
    }

    #[must_use]
    pub const fn preprocessed_next_offset(&self) -> usize {
        self.preprocessed_local_offset() + self.preprocessed_width
    }

    #[must_use]
    pub const fn permutation_local_offset(&self) -> usize {
        self.preprocessed_next_offset() + self.preprocessed_width
    }

    #[must_use]
    pub const fn permutation_next_offset(&self) -> usize {
        self.permutation_local_offset() + self.permutation_width
    }

    #[must_use]
    pub const fn public_values_offset(&self) -> usize {
        self.permutation_next_offset() + self.permutation_width
    }

    #[must_use]
    pub const fn challenges_offset(&self) -> usize {
        self.public_values_offset() + self.public_values
    }

    #[must_use]
    pub const fn selectors_offset(&self) -> usize {
        self.challenges_offset() + self.challenges
    }

    #[must_use]
    pub fn slot_for_input(&self, input: KernelInput) -> usize {
        match input {
            KernelInput::Main { row_offset, column } => match row_offset {
                0 => self.main_local_offset() + column,
                1 => self.main_next_offset() + column,
                _ => panic!("unsupported row offset {row_offset} for main input"),
            },
            KernelInput::Preprocessed { row_offset, column } => match row_offset {
                0 => self.preprocessed_local_offset() + column,
                1 => self.preprocessed_next_offset() + column,
                _ => panic!("unsupported row offset {row_offset} for preprocessed input"),
            },
            KernelInput::Permutation { row_offset, column } => match row_offset {
                0 => self.permutation_local_offset() + column,
                1 => self.permutation_next_offset() + column,
                _ => panic!("unsupported row offset {row_offset} for permutation input"),
            },
            KernelInput::Public { index } => self.public_values_offset() + index,
            KernelInput::Challenge { index } => self.challenges_offset() + index,
            KernelInput::Selector(RowSelector::First) => self.selectors_offset(),
            KernelInput::Selector(RowSelector::Last) => self.selectors_offset() + 1,
            KernelInput::Selector(RowSelector::Transition) => self.selectors_offset() + 2,
        }
    }

    #[must_use]
    pub fn flatten_inputs<T: Copy>(&self, inputs: &ConstraintProgramInputs<'_, T>) -> Vec<T> {
        let mut flat_inputs = Vec::with_capacity(self.total_slots());
        self.push_slice(&mut flat_inputs, inputs.main.local, self.main_width, "main local");
        self.push_slice(&mut flat_inputs, inputs.main.next, self.main_width, "main next");
        self.push_slice(
            &mut flat_inputs,
            inputs.preprocessed.local,
            self.preprocessed_width,
            "preprocessed local",
        );
        self.push_slice(
            &mut flat_inputs,
            inputs.preprocessed.next,
            self.preprocessed_width,
            "preprocessed next",
        );
        self.push_slice(
            &mut flat_inputs,
            inputs.permutation.local,
            self.permutation_width,
            "permutation local",
        );
        self.push_slice(
            &mut flat_inputs,
            inputs.permutation.next,
            self.permutation_width,
            "permutation next",
        );
        self.push_slice(
            &mut flat_inputs,
            inputs.public_values,
            self.public_values,
            "public values",
        );
        self.push_slice(
            &mut flat_inputs,
            inputs.challenges,
            self.challenges,
            "challenges",
        );
        flat_inputs.push(inputs.is_first_row);
        flat_inputs.push(inputs.is_last_row);
        flat_inputs.push(inputs.is_transition);
        flat_inputs
    }

    fn push_slice<T: Copy>(
        &self,
        flat_inputs: &mut Vec<T>,
        values: &[T],
        width: usize,
        label: &str,
    ) {
        assert!(
            values.len() >= width,
            "{label} requires at least {width} values, got {}",
            values.len(),
        );
        flat_inputs.extend_from_slice(&values[..width]);
    }
}

impl<'a, F> ConstraintProgramInputs<'a, F> {
    #[must_use]
    pub const fn with_preprocessed(mut self, local: &'a [F], next: &'a [F]) -> Self {
        self.preprocessed = RowPair::new(local, next);
        self
    }

    #[must_use]
    pub const fn with_permutation(mut self, local: &'a [F], next: &'a [F]) -> Self {
        self.permutation = RowPair::new(local, next);
        self
    }

    #[must_use]
    pub const fn with_public_values(mut self, public_values: &'a [F]) -> Self {
        self.public_values = public_values;
        self
    }

    #[must_use]
    pub const fn with_challenges(mut self, challenges: &'a [F]) -> Self {
        self.challenges = challenges;
        self
    }

    #[must_use]
    pub fn read(&self, input: KernelInput) -> F
    where
        F: Copy,
    {
        match input {
            KernelInput::Main { row_offset, column } => {
                self.read_row_pair(self.main, row_offset, column, "main")
            }
            KernelInput::Preprocessed { row_offset, column } => {
                self.read_row_pair(self.preprocessed, row_offset, column, "preprocessed")
            }
            KernelInput::Public { index } => self
                .public_values
                .get(index)
                .copied()
                .unwrap_or_else(|| panic!("public input {index} is out of bounds")),
            KernelInput::Challenge { index } => self
                .challenges
                .get(index)
                .copied()
                .unwrap_or_else(|| panic!("challenge input {index} is out of bounds")),
            KernelInput::Permutation { row_offset, column } => {
                self.read_row_pair(self.permutation, row_offset, column, "permutation")
            }
            KernelInput::Selector(RowSelector::First) => self.is_first_row,
            KernelInput::Selector(RowSelector::Last) => self.is_last_row,
            KernelInput::Selector(RowSelector::Transition) => self.is_transition,
        }
    }

    fn read_row_pair(
        &self,
        rows: RowPair<'_, F>,
        row_offset: usize,
        column: usize,
        label: &str,
    ) -> F
    where
        F: Copy,
    {
        let row = match row_offset {
            0 => rows.local,
            1 => rows.next,
            _ => panic!("unsupported row offset {row_offset} for {label} input"),
        };

        row.get(column)
            .copied()
            .unwrap_or_else(|| panic!("{label} column {column} is out of bounds"))
    }
}

impl<'a, F: Field> ConstraintProgramInputs<'a, F> {
    #[must_use]
    pub fn new(
        main_local: &'a [F],
        main_next: &'a [F],
        is_first_row: F,
        is_last_row: F,
        is_transition: F,
    ) -> Self {
        Self {
            main: RowPair::new(main_local, main_next),
            preprocessed: RowPair::empty(),
            permutation: RowPair::empty(),
            public_values: &[],
            challenges: &[],
            is_first_row,
            is_last_row,
            is_transition,
        }
    }

    #[must_use]
    pub fn for_main_row(
        main_local: &'a [F],
        main_next: &'a [F],
        row_index: usize,
        trace_height: usize,
    ) -> Self {
        assert!(trace_height > 0, "trace height must be non-zero");
        assert!(
            row_index < trace_height,
            "row index {row_index} is out of bounds for trace height {trace_height}"
        );

        Self::new(
            main_local,
            main_next,
            if row_index == 0 { F::ONE } else { F::ZERO },
            if row_index + 1 == trace_height {
                F::ONE
            } else {
                F::ZERO
            },
            if row_index + 1 == trace_height {
                F::ZERO
            } else {
                F::ONE
            },
        )
    }
}

impl<F: Field> ConstraintProgram<F> {
    #[must_use]
    pub fn lower(&self) -> LoweredConstraintProgram<F> {
        let input_layout = self.derive_input_layout();
        let instructions = self
            .instructions
            .iter()
            .map(|instruction| match instruction {
                KernelInstruction::Constant(value) => LoweredKernelInstruction::Constant(*value),
                KernelInstruction::Input(input) => LoweredKernelInstruction::Input {
                    slot: input_layout.slot_for_input(*input),
                },
                KernelInstruction::Add { lhs, rhs } => LoweredKernelInstruction::Add {
                    lhs: *lhs,
                    rhs: *rhs,
                },
                KernelInstruction::Sub { lhs, rhs } => LoweredKernelInstruction::Sub {
                    lhs: *lhs,
                    rhs: *rhs,
                },
                KernelInstruction::Neg { src } => LoweredKernelInstruction::Neg { src: *src },
                KernelInstruction::Mul { lhs, rhs } => LoweredKernelInstruction::Mul {
                    lhs: *lhs,
                    rhs: *rhs,
                },
            })
            .collect();

        LoweredConstraintProgram {
            input_layout,
            instructions,
            outputs: self.outputs.clone(),
            unsupported_features: self.unsupported_features.clone(),
            max_constraint_degree: self.max_constraint_degree,
        }
    }

    fn derive_input_layout(&self) -> ConstraintInputLayout {
        let mut layout = ConstraintInputLayout::default();

        for instruction in &self.instructions {
            let KernelInstruction::Input(input) = instruction else {
                continue;
            };

            match input {
                KernelInput::Main { column, .. } => {
                    layout.main_width = layout.main_width.max(column + 1);
                }
                KernelInput::Preprocessed { column, .. } => {
                    layout.preprocessed_width = layout.preprocessed_width.max(column + 1);
                }
                KernelInput::Permutation { column, .. } => {
                    layout.permutation_width = layout.permutation_width.max(column + 1);
                }
                KernelInput::Public { index } => {
                    layout.public_values = layout.public_values.max(index + 1);
                }
                KernelInput::Challenge { index } => {
                    layout.challenges = layout.challenges.max(index + 1);
                }
                KernelInput::Selector(_) => {}
            }
        }

        layout
    }

    #[must_use]
    pub fn evaluate(&self, inputs: &ConstraintProgramInputs<'_, F>) -> Vec<F> {
        let mut values: Vec<F> = Vec::with_capacity(self.instructions.len());

        for instruction in &self.instructions {
            let value = match instruction {
                KernelInstruction::Constant(value) => *value,
                KernelInstruction::Input(input) => inputs.read(*input),
                KernelInstruction::Add { lhs, rhs } => values[*lhs] + values[*rhs],
                KernelInstruction::Sub { lhs, rhs } => values[*lhs] - values[*rhs],
                KernelInstruction::Neg { src } => -values[*src],
                KernelInstruction::Mul { lhs, rhs } => values[*lhs] * values[*rhs],
            };
            values.push(value);
        }

        self.outputs.iter().map(|&output| values[output]).collect()
    }

    #[must_use]
    pub fn evaluate_batched<EF>(
        &self,
        inputs: &ConstraintProgramInputs<'_, F>,
        batching_scalars: &[EF],
    ) -> EF
    where
        EF: ExtensionField<F>,
    {
        self.evaluate_batched_in_extension(inputs, batching_scalars)
    }

    #[must_use]
    pub fn evaluate_batched_in_extension<NF, EF>(
        &self,
        inputs: &ConstraintProgramInputs<'_, NF>,
        batching_scalars: &[EF],
    ) -> EF
    where
        NF: ExtensionField<F>,
        EF: ExtensionField<NF>,
    {
        assert_eq!(
            self.outputs.len(),
            batching_scalars.len(),
            "constraint program output count must match batching scalar count",
        );

        let mut values: Vec<NF> = Vec::with_capacity(self.instructions.len());

        for instruction in &self.instructions {
            let value = match instruction {
                KernelInstruction::Constant(value) => NF::from(*value),
                KernelInstruction::Input(input) => inputs.read(*input),
                KernelInstruction::Add { lhs, rhs } => values[*lhs] + values[*rhs],
                KernelInstruction::Sub { lhs, rhs } => values[*lhs] - values[*rhs],
                KernelInstruction::Neg { src } => -values[*src],
                KernelInstruction::Mul { lhs, rhs } => values[*lhs] * values[*rhs],
            };
            values.push(value);
        }

        self.outputs
            .iter()
            .zip(batching_scalars)
            .map(|(&output, &alpha)| alpha * values[output])
            .sum()
    }

    #[must_use]
    pub fn evaluate_batched_packed<EF>(
        &self,
        inputs: &ConstraintProgramInputs<'_, F::Packing>,
        batching_scalars: &[EF],
    ) -> EF::ExtensionPacking
    where
        EF: ExtensionField<F>,
    {
        assert_eq!(
            self.outputs.len(),
            batching_scalars.len(),
            "constraint program output count must match batching scalar count",
        );

        let mut values: Vec<F::Packing> = Vec::with_capacity(self.instructions.len());

        for instruction in &self.instructions {
            let value = match instruction {
                KernelInstruction::Constant(value) => F::Packing::from_fn(|_| *value),
                KernelInstruction::Input(input) => inputs.read(*input),
                KernelInstruction::Add { lhs, rhs } => values[*lhs] + values[*rhs],
                KernelInstruction::Sub { lhs, rhs } => values[*lhs] - values[*rhs],
                KernelInstruction::Neg { src } => -values[*src],
                KernelInstruction::Mul { lhs, rhs } => values[*lhs] * values[*rhs],
            };
            values.push(value);
        }

        self.outputs
            .iter()
            .zip(batching_scalars)
            .fold(EF::ExtensionPacking::ZERO, |acc, (&output, &alpha)| {
                acc + Into::<EF::ExtensionPacking>::into(alpha)
                    * Into::<EF::ExtensionPacking>::into(values[output])
            })
    }
}

impl<F: Field> LoweredConstraintProgram<F> {
    #[must_use]
    pub fn evaluate_batched<EF>(&self, flat_inputs: &[F], batching_scalars: &[EF]) -> EF
    where
        EF: ExtensionField<F>,
    {
        self.evaluate_batched_in_extension(flat_inputs, batching_scalars)
    }

    #[must_use]
    pub fn evaluate_batched_in_extension<NF, EF>(
        &self,
        flat_inputs: &[NF],
        batching_scalars: &[EF],
    ) -> EF
    where
        NF: ExtensionField<F>,
        EF: ExtensionField<NF>,
    {
        assert_eq!(
            self.outputs.len(),
            batching_scalars.len(),
            "constraint program output count must match batching scalar count",
        );
        assert!(
            flat_inputs.len() >= self.input_layout.total_slots(),
            "flat input buffer requires at least {} values, got {}",
            self.input_layout.total_slots(),
            flat_inputs.len(),
        );

        let mut values: Vec<NF> = Vec::with_capacity(self.instructions.len());

        for instruction in &self.instructions {
            let value = match instruction {
                LoweredKernelInstruction::Constant(value) => NF::from(*value),
                LoweredKernelInstruction::Input { slot } => flat_inputs[*slot],
                LoweredKernelInstruction::Add { lhs, rhs } => values[*lhs] + values[*rhs],
                LoweredKernelInstruction::Sub { lhs, rhs } => values[*lhs] - values[*rhs],
                LoweredKernelInstruction::Neg { src } => -values[*src],
                LoweredKernelInstruction::Mul { lhs, rhs } => values[*lhs] * values[*rhs],
            };
            values.push(value);
        }

        self.outputs
            .iter()
            .zip(batching_scalars)
            .map(|(&output, &alpha)| alpha * values[output])
            .sum()
    }

    #[must_use]
    pub fn evaluate_batched_packed<EF>(
        &self,
        flat_inputs: &[F::Packing],
        batching_scalars: &[EF],
    ) -> EF::ExtensionPacking
    where
        EF: ExtensionField<F>,
    {
        assert_eq!(
            self.outputs.len(),
            batching_scalars.len(),
            "constraint program output count must match batching scalar count",
        );
        assert!(
            flat_inputs.len() >= self.input_layout.total_slots(),
            "flat input buffer requires at least {} values, got {}",
            self.input_layout.total_slots(),
            flat_inputs.len(),
        );

        let mut values: Vec<F::Packing> = Vec::with_capacity(self.instructions.len());

        for instruction in &self.instructions {
            let value = match instruction {
                LoweredKernelInstruction::Constant(value) => F::Packing::from_fn(|_| *value),
                LoweredKernelInstruction::Input { slot } => flat_inputs[*slot],
                LoweredKernelInstruction::Add { lhs, rhs } => values[*lhs] + values[*rhs],
                LoweredKernelInstruction::Sub { lhs, rhs } => values[*lhs] - values[*rhs],
                LoweredKernelInstruction::Neg { src } => -values[*src],
                LoweredKernelInstruction::Mul { lhs, rhs } => values[*lhs] * values[*rhs],
            };
            values.push(value);
        }

        self.outputs
            .iter()
            .zip(batching_scalars)
            .fold(EF::ExtensionPacking::ZERO, |acc, (&output, &alpha)| {
                acc + Into::<EF::ExtensionPacking>::into(alpha)
                    * Into::<EF::ExtensionPacking>::into(values[output])
            })
    }
}

#[must_use]
pub fn compile_air_constraint_program<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
) -> ConstraintProgram<F>
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    let constraints = get_symbolic_constraints(air, preprocessed_width, num_public_values);
    let mut program = ConstraintProgram {
        instructions: Vec::new(),
        outputs: Vec::with_capacity(constraints.len()),
        unsupported_features: Vec::new(),
        max_constraint_degree: constraints
            .iter()
            .map(SymbolicExpression::degree_multiple)
            .max()
            .unwrap_or(0),
    };
    let mut memo = HashMap::new();

    for constraint in &constraints {
        let output = compile_expression(constraint, &mut program, &mut memo);
        program.outputs.push(output);
    }

    program
}

fn compile_expression<F: Field>(
    expr: &SymbolicExpression<F>,
    program: &mut ConstraintProgram<F>,
    memo: &mut HashMap<usize, usize>,
) -> usize {
    let key = expr as *const SymbolicExpression<F> as usize;
    if let Some(&cached) = memo.get(&key) {
        return cached;
    }

    let instruction = match expr {
        SymbolicExpression::Variable(var) => {
            let input = kernel_input_for_variable(var, &mut program.unsupported_features);
            KernelInstruction::Input(input)
        }
        SymbolicExpression::IsFirstRow => {
            push_unsupported(
                &mut program.unsupported_features,
                UnsupportedGpuFeature::RowSelector(RowSelector::First),
            );
            KernelInstruction::Input(KernelInput::Selector(RowSelector::First))
        }
        SymbolicExpression::IsLastRow => {
            push_unsupported(
                &mut program.unsupported_features,
                UnsupportedGpuFeature::RowSelector(RowSelector::Last),
            );
            KernelInstruction::Input(KernelInput::Selector(RowSelector::Last))
        }
        SymbolicExpression::IsTransition => {
            push_unsupported(
                &mut program.unsupported_features,
                UnsupportedGpuFeature::RowSelector(RowSelector::Transition),
            );
            KernelInstruction::Input(KernelInput::Selector(RowSelector::Transition))
        }
        SymbolicExpression::Constant(value) => KernelInstruction::Constant(*value),
        SymbolicExpression::Add { x, y, .. } => {
            let lhs = compile_expression(x.as_ref(), program, memo);
            let rhs = compile_expression(y.as_ref(), program, memo);
            KernelInstruction::Add { lhs, rhs }
        }
        SymbolicExpression::Sub { x, y, .. } => {
            let lhs = compile_expression(x.as_ref(), program, memo);
            let rhs = compile_expression(y.as_ref(), program, memo);
            KernelInstruction::Sub { lhs, rhs }
        }
        SymbolicExpression::Neg { x, .. } => {
            let src = compile_expression(x.as_ref(), program, memo);
            KernelInstruction::Neg { src }
        }
        SymbolicExpression::Mul { x, y, .. } => {
            let lhs = compile_expression(x.as_ref(), program, memo);
            let rhs = compile_expression(y.as_ref(), program, memo);
            KernelInstruction::Mul { lhs, rhs }
        }
    };

    let id = push_instruction(program, instruction);
    memo.insert(key, id);
    id
}

fn kernel_input_for_variable<F: Field>(
    var: &SymbolicVariable<F>,
    unsupported_features: &mut Vec<UnsupportedGpuFeature>,
) -> KernelInput {
    match var.entry {
        Entry::Main { offset } => KernelInput::Main {
            row_offset: offset,
            column: var.index,
        },
        Entry::Preprocessed { offset } => KernelInput::Preprocessed {
            row_offset: offset,
            column: var.index,
        },
        Entry::Permutation { offset } => {
            push_unsupported(
                unsupported_features,
                UnsupportedGpuFeature::PermutationInput,
            );
            KernelInput::Permutation {
                row_offset: offset,
                column: var.index,
            }
        }
        Entry::Public => {
            push_unsupported(unsupported_features, UnsupportedGpuFeature::PublicInput);
            KernelInput::Public { index: var.index }
        }
        Entry::Challenge => {
            push_unsupported(unsupported_features, UnsupportedGpuFeature::ChallengeInput);
            KernelInput::Challenge { index: var.index }
        }
    }
}

fn push_instruction<F>(
    program: &mut ConstraintProgram<F>,
    instruction: KernelInstruction<F>,
) -> usize {
    let id = program.instructions.len();
    program.instructions.push(instruction);
    id
}

fn push_unsupported(
    unsupported_features: &mut Vec<UnsupportedGpuFeature>,
    feature: UnsupportedGpuFeature,
) {
    if !unsupported_features.contains(&feature) {
        unsupported_features.push(feature);
    }
}

#[cfg(test)]
mod tests {
    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_field::{
        BasedVectorSpace, ExtensionField, Field, PackedValue, PrimeCharacteristicRing,
        extension::BinomialExtensionField,
    };
    use p3_koala_bear::KoalaBear;
    use p3_matrix::Matrix;

    use super::*;

    type EF = BinomialExtensionField<KoalaBear, 8>;
    type PF = <KoalaBear as Field>::Packing;
    type PEF = <EF as ExtensionField<KoalaBear>>::ExtensionPacking;

    struct ArithmeticAir;

    impl BaseAir<KoalaBear> for ArithmeticAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl Air<SymbolicAirBuilder<KoalaBear>> for ArithmeticAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<KoalaBear>) {
            let main = builder.main();
            let local = main.row_slice(0).expect("matrix is empty");
            let local = &*local;
            builder.assert_zero(local[0] * local[1] + local[0] - KoalaBear::ONE);
        }
    }

    struct FirstRowAir;

    impl BaseAir<KoalaBear> for FirstRowAir {
        fn width(&self) -> usize {
            1
        }
    }

    impl Air<SymbolicAirBuilder<KoalaBear>> for FirstRowAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<KoalaBear>) {
            let main = builder.main();
            let local = main.row_slice(0).expect("matrix is empty");
            let local = &*local;
            builder.when_first_row().assert_zero(local[0]);
        }
    }

    #[test]
    fn compiles_simple_air_to_kernel_program() {
        let program = compile_air_constraint_program::<KoalaBear, _>(&ArithmeticAir, 0, 0);
        let stats = program.stats();

        assert!(program.is_gpu_candidate());
        assert_eq!(stats.constraints, 1);
        assert_eq!(stats.inputs, 3);
        assert_eq!(stats.muls, 1);
        assert_eq!(stats.adds, 1);
        assert_eq!(stats.subs, 1);
        assert_eq!(stats.unsupported_features, 0);
        assert_eq!(program.max_constraint_degree, 2);
    }

    #[test]
    fn executable_program_matches_simple_air_constraint_value() {
        let program = compile_air_constraint_program::<KoalaBear, _>(&ArithmeticAir, 0, 0);
        let local = [KoalaBear::new(2), KoalaBear::new(3)];
        let next = [KoalaBear::ZERO, KoalaBear::ZERO];

        let outputs = program.evaluate(&ConstraintProgramInputs::for_main_row(&local, &next, 0, 2));

        assert_eq!(outputs, vec![KoalaBear::new(7)]);
    }

    #[test]
    fn batched_execution_matches_constraint_outputs() {
        let program = compile_air_constraint_program::<KoalaBear, _>(&ArithmeticAir, 0, 0);
        let lowered = program.lower();
        let local = [KoalaBear::new(2), KoalaBear::new(3)];
        let next = [KoalaBear::ZERO, KoalaBear::ZERO];
        let inputs = ConstraintProgramInputs::for_main_row(&local, &next, 0, 2);
        let batching_scalars = [EF::from(KoalaBear::new(5))];

        let outputs = program.evaluate(&inputs);
        let batched = program.evaluate_batched(&inputs, &batching_scalars);
        let flat_inputs = lowered.input_layout.flatten_inputs(&inputs);
        let lowered_batched = lowered.evaluate_batched(&flat_inputs, &batching_scalars);

        assert_eq!(batched, batching_scalars[0] * outputs[0]);
        assert_eq!(lowered_batched, batched);
    }

    #[test]
    fn packed_batched_execution_matches_scalar_lanes() {
        let program = compile_air_constraint_program::<KoalaBear, _>(&ArithmeticAir, 0, 0);
        let lowered = program.lower();
        let local0 = PF::from_fn(|idx| KoalaBear::new(idx as u32 + 2));
        let local1 = PF::from_fn(|idx| KoalaBear::new(idx as u32 + 3));
        let next0 = PF::from_fn(|_| KoalaBear::ZERO);
        let next1 = PF::from_fn(|_| KoalaBear::ZERO);
        let local_rows = [local0, local1];
        let next_rows = [next0, next1];
        let packed_inputs = ConstraintProgramInputs {
            main: RowPair::new(&local_rows, &next_rows),
            preprocessed: RowPair::empty(),
            permutation: RowPair::empty(),
            public_values: &[],
            challenges: &[],
            is_first_row: PF::from_fn(|_| KoalaBear::ZERO),
            is_last_row: PF::from_fn(|_| KoalaBear::ZERO),
            is_transition: PF::from_fn(|_| KoalaBear::ONE),
        };
        let batching_scalars = [EF::from(KoalaBear::new(5))];

        let packed = program.evaluate_batched_packed(&packed_inputs, &batching_scalars);
        let flat_inputs = lowered.input_layout.flatten_inputs(&packed_inputs);
        let lowered_packed = lowered.evaluate_batched_packed(&flat_inputs, &batching_scalars);
        let actual = (0..PF::WIDTH)
            .map(|idx| {
                let packed = lowered_packed;
                EF::from_basis_coefficients_fn(|coeff_idx| {
                    <PEF as BasedVectorSpace<PF>>::as_basis_coefficients_slice(&packed)[coeff_idx]
                        .as_slice()[idx]
                })
            })
            .collect::<Vec<_>>();
        let expected = (0..PF::WIDTH)
            .map(|idx| {
                let local = [local0.as_slice()[idx], local1.as_slice()[idx]];
                let next = [next0.as_slice()[idx], next1.as_slice()[idx]];
                let inputs = ConstraintProgramInputs::new(
                    &local,
                    &next,
                    KoalaBear::ZERO,
                    KoalaBear::ZERO,
                    KoalaBear::ONE,
                );
                program.evaluate_batched(&inputs, &batching_scalars)
            })
            .collect::<Vec<_>>();

        assert_eq!(lowered_packed, packed);
        assert_eq!(actual, expected);
    }

    #[test]
    fn flags_row_selectors_as_unsupported() {
        let program = compile_air_constraint_program::<KoalaBear, _>(&FirstRowAir, 0, 0);

        assert!(!program.is_gpu_candidate());
        assert_eq!(
            program.unsupported_features,
            vec![UnsupportedGpuFeature::RowSelector(RowSelector::First)]
        );
    }

    #[test]
    fn executable_program_respects_row_selectors() {
        let program = compile_air_constraint_program::<KoalaBear, _>(&FirstRowAir, 0, 0);
        let local = [KoalaBear::new(5)];
        let next = [KoalaBear::ZERO];

        let first_row =
            program.evaluate(&ConstraintProgramInputs::for_main_row(&local, &next, 0, 2));
        let last_row =
            program.evaluate(&ConstraintProgramInputs::for_main_row(&local, &next, 1, 2));

        assert_eq!(first_row, vec![KoalaBear::new(5)]);
        assert_eq!(last_row, vec![KoalaBear::ZERO]);
    }
}
