use bytemuck::{bytes_of, cast_slice};
use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};

use super::PackedMainGpuPayload;

const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Header {
    uint order;
    uint point_count;
    uint total_slots;
    uint instruction_count;
    uint output_count;
    uint extension_degree;
    uint reserved0;
    uint reserved1;
};

struct Instruction {
    uint opcode;
    uint arg0;
    uint arg1;
    uint reserved;
};

constant uint MAX_OPCODES = 2048;
constant uint MAX_EXTENSION_DEGREE = 8;

inline uint add_mod(uint lhs, uint rhs, uint order) {
    uint threshold = order - rhs;
    if (lhs >= threshold) {
        return lhs - threshold;
    }
    return lhs + rhs;
}

inline uint sub_mod(uint lhs, uint rhs, uint order) {
    if (lhs >= rhs) {
        return lhs - rhs;
    }
    return order - (rhs - lhs);
}

inline uint neg_mod(uint value, uint order) {
    if (value == 0) {
        return 0;
    }
    return order - value;
}

inline uint mul_mod(uint lhs, uint rhs, uint order) {
    uint a = lhs;
    uint b = rhs;
    uint acc = 0;
    while (b != 0) {
        if ((b & 1) != 0) {
            acc = add_mod(acc, a, order);
        }
        a = add_mod(a, a, order);
        b >>= 1;
    }
    return acc;
}

kernel void evaluate_air(
    constant Header& header [[buffer(0)]],
    device const Instruction* instructions [[buffer(1)]],
    device const uint* constants [[buffer(2)]],
    device const uint* outputs [[buffer(3)]],
    device const uint* batching_scalars [[buffer(4)]],
    device const uint* flat_inputs [[buffer(5)]],
    device uint* results [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= header.point_count) {
        return;
    }

    uint values[MAX_OPCODES];
    uint acc[MAX_EXTENSION_DEGREE];

    for (uint coeff = 0; coeff < header.extension_degree; coeff++) {
        acc[coeff] = 0;
    }

    uint point_offset = gid * header.total_slots;
    for (uint inst_idx = 0; inst_idx < header.instruction_count; inst_idx++) {
        Instruction inst = instructions[inst_idx];
        switch (inst.opcode) {
            case 0:
                values[inst_idx] = constants[inst.arg0];
                break;
            case 1:
                values[inst_idx] = flat_inputs[point_offset + inst.arg0];
                break;
            case 2:
                values[inst_idx] = add_mod(values[inst.arg0], values[inst.arg1], header.order);
                break;
            case 3:
                values[inst_idx] = sub_mod(values[inst.arg0], values[inst.arg1], header.order);
                break;
            case 4:
                values[inst_idx] = neg_mod(values[inst.arg0], header.order);
                break;
            default:
                values[inst_idx] = mul_mod(values[inst.arg0], values[inst.arg1], header.order);
                break;
        }
    }

    for (uint output_idx = 0; output_idx < header.output_count; output_idx++) {
        uint value = values[outputs[output_idx]];
        uint scalar_base = output_idx * header.extension_degree;
        for (uint coeff = 0; coeff < header.extension_degree; coeff++) {
            acc[coeff] = add_mod(
                acc[coeff],
                mul_mod(value, batching_scalars[scalar_base + coeff], header.order),
                header.order
            );
        }
    }

    uint result_base = gid * header.extension_degree;
    for (uint coeff = 0; coeff < header.extension_degree; coeff++) {
        results[result_base + coeff] = acc[coeff];
    }
}
"#;

pub(super) fn execute(payload: &PackedMainGpuPayload) -> Result<Vec<u32>, String> {
    let device = Device::system_default().ok_or("no Metal device available")?;
    let library = device
        .new_library_with_source(MSL_SOURCE, &CompileOptions::new())
        .map_err(|err| format!("new_library_with_source failed: {err}"))?;
    let function = library
        .get_function("evaluate_air", None)
        .map_err(|err| format!("get_function failed: {err}"))?;

    let pipeline_descriptor = ComputePipelineDescriptor::new();
    pipeline_descriptor.set_compute_function(Some(&function));
    let pipeline = device
        .new_compute_pipeline_state_with_function(
            pipeline_descriptor
                .compute_function()
                .ok_or("compute function missing from pipeline descriptor")?,
        )
        .map_err(|err| format!("new_compute_pipeline_state_with_function failed: {err}"))?;

    let command_queue = device.new_command_queue();
    let header_buffer = device.new_buffer_with_data(
        bytes_of(&payload.header).as_ptr().cast(),
        std::mem::size_of::<super::GpuKernelHeader>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let instruction_buffer = device.new_buffer_with_data(
        cast_slice::<super::GpuInstruction, u8>(&payload.instructions)
            .as_ptr()
            .cast(),
        std::mem::size_of_val(payload.instructions.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let constants_buffer = new_data_buffer(&device, &payload.constants);
    let outputs_buffer = device.new_buffer_with_data(
        cast_slice::<u32, u8>(&payload.outputs).as_ptr().cast(),
        std::mem::size_of_val(payload.outputs.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let batching_buffer = device.new_buffer_with_data(
        cast_slice::<u32, u8>(&payload.batching_scalars)
            .as_ptr()
            .cast(),
        std::mem::size_of_val(payload.batching_scalars.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let flat_inputs_buffer = device.new_buffer_with_data(
        cast_slice::<u32, u8>(&payload.flat_inputs).as_ptr().cast(),
        std::mem::size_of_val(payload.flat_inputs.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let results_buffer = device.new_buffer(
        (payload.result_len_u32s() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&header_buffer), 0);
    encoder.set_buffer(1, Some(&instruction_buffer), 0);
    encoder.set_buffer(2, Some(&constants_buffer), 0);
    encoder.set_buffer(3, Some(&outputs_buffer), 0);
    encoder.set_buffer(4, Some(&batching_buffer), 0);
    encoder.set_buffer(5, Some(&flat_inputs_buffer), 0);
    encoder.set_buffer(6, Some(&results_buffer), 0);

    let threads_per_group = pipeline.thread_execution_width().min(64);
    encoder.dispatch_threads(
        MTLSize::new(payload.header.point_count as u64, 1, 1),
        MTLSize::new(threads_per_group as u64, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results = unsafe {
        std::slice::from_raw_parts(
            results_buffer.contents().cast::<u32>(),
            payload.result_len_u32s(),
        )
    };
    Ok(results.to_vec())
}

fn new_data_buffer(device: &Device, data: &[u32]) -> metal::Buffer {
    let owned;
    let source = if data.is_empty() {
        owned = [0u32];
        owned.as_slice()
    } else {
        data
    };

    device.new_buffer_with_data(
        cast_slice::<u32, u8>(source).as_ptr().cast(),
        std::mem::size_of_val(source) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}
