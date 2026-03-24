use std::sync::{Mutex, MutexGuard, OnceLock, mpsc};

use bytemuck::cast_slice;
use pollster::block_on;
use wgpu::util::DeviceExt;

use super::PackedMainGpuPayload;

const WORKGROUP_SIZE: u32 = 64;

struct WgpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    program_buffers: Mutex<Option<WgpuProgramBuffers>>,
}

struct WgpuProgramBuffers {
    fingerprint: u64,
    instruction_buffer: wgpu::Buffer,
    constants_buffer: wgpu::Buffer,
    outputs_buffer: wgpu::Buffer,
}

fn runtime() -> Result<&'static WgpuRuntime, String> {
    static RUNTIME: OnceLock<Result<WgpuRuntime, String>> = OnceLock::new();

    match RUNTIME.get_or_init(init_runtime) {
        Ok(runtime) => Ok(runtime),
        Err(err) => Err(err.clone()),
    }
}

fn init_runtime() -> Result<WgpuRuntime, String> {
    let instance = wgpu::Instance::default();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .ok_or("request_adapter returned no compatible device")?;

    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("whirlaway-air-wgpu-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .map_err(|err| format!("request_device failed: {err}"))?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("whirlaway-air-wgpu-shader"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("whirlaway-air-wgpu-pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("evaluate_air"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group_layout = pipeline.get_bind_group_layout(0);

    Ok(WgpuRuntime {
        device,
        queue,
        pipeline,
        bind_group_layout,
        program_buffers: Mutex::new(None),
    })
}

fn static_program_buffers<'a>(
    runtime: &'a WgpuRuntime,
    payload: &PackedMainGpuPayload<'_>,
) -> MutexGuard<'a, Option<WgpuProgramBuffers>> {
    let mut cached = runtime
        .program_buffers
        .lock()
        .expect("wgpu program buffer cache should not be poisoned");

    if cached
        .as_ref()
        .is_none_or(|buffers| buffers.fingerprint != payload.program.fingerprint)
    {
        let device = &runtime.device;
        let instruction_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("whirlaway-air-wgpu-instructions"),
            contents: cast_slice(payload.program.instructions.as_slice()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let constants_data = if payload.program.constants.is_empty() {
            &[0u32][..]
        } else {
            payload.program.constants.as_slice()
        };
        let constants_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("whirlaway-air-wgpu-constants"),
            contents: cast_slice(constants_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let outputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("whirlaway-air-wgpu-outputs"),
            contents: cast_slice(payload.program.outputs.as_slice()),
            usage: wgpu::BufferUsages::STORAGE,
        });

        *cached = Some(WgpuProgramBuffers {
            fingerprint: payload.program.fingerprint,
            instruction_buffer,
            constants_buffer,
            outputs_buffer,
        });
    }

    cached
}

const WGSL_SOURCE: &str = r#"
struct Header {
    order: u32,
    point_count: u32,
    total_slots: u32,
    instruction_count: u32,
    output_count: u32,
    extension_degree: u32,
    reserved0: u32,
    reserved1: u32,
};

struct Instruction {
    opcode: u32,
    arg0: u32,
    arg1: u32,
    reserved: u32,
};

@group(0) @binding(0)
var<uniform> header: Header;

@group(0) @binding(1)
var<storage, read> instructions: array<Instruction>;

@group(0) @binding(2)
var<storage, read> constants: array<u32>;

@group(0) @binding(3)
var<storage, read> outputs: array<u32>;

@group(0) @binding(4)
var<storage, read> batching_scalars: array<u32>;

@group(0) @binding(5)
var<storage, read> flat_inputs: array<u32>;

@group(0) @binding(6)
var<storage, read_write> results: array<u32>;

const MAX_OPCODES: u32 = 2048u;
const MAX_EXTENSION_DEGREE: u32 = 8u;

fn add_mod(lhs: u32, rhs: u32, order: u32) -> u32 {
    let threshold = order - rhs;
    if lhs >= threshold {
        return lhs - threshold;
    }
    return lhs + rhs;
}

fn sub_mod(lhs: u32, rhs: u32, order: u32) -> u32 {
    if lhs >= rhs {
        return lhs - rhs;
    }
    return order - (rhs - lhs);
}

fn neg_mod(value: u32, order: u32) -> u32 {
    if value == 0u {
        return 0u;
    }
    return order - value;
}

fn mul_mod(lhs: u32, rhs: u32, order: u32) -> u32 {
    var a = lhs;
    var b = rhs;
    var acc = 0u;
    loop {
        if b == 0u {
            break;
        }
        if (b & 1u) != 0u {
            acc = add_mod(acc, a, order);
        }
        a = add_mod(a, a, order);
        b = b >> 1u;
    }
    return acc;
}

@compute @workgroup_size(64)
fn evaluate_air(@builtin(global_invocation_id) gid: vec3<u32>) {
    let point_idx = gid.x;
    if point_idx >= header.point_count {
        return;
    }

    var values: array<u32, MAX_OPCODES>;
    var acc: array<u32, MAX_EXTENSION_DEGREE>;

    for (var coeff = 0u; coeff < header.extension_degree; coeff++) {
        acc[coeff] = 0u;
    }

    let point_offset = point_idx * header.total_slots;
    for (var inst_idx = 0u; inst_idx < header.instruction_count; inst_idx++) {
        let inst = instructions[inst_idx];
        switch inst.opcode {
            case 0u: {
                values[inst_idx] = constants[inst.arg0];
            }
            case 1u: {
                values[inst_idx] = flat_inputs[point_offset + inst.arg0];
            }
            case 2u: {
                values[inst_idx] = add_mod(values[inst.arg0], values[inst.arg1], header.order);
            }
            case 3u: {
                values[inst_idx] = sub_mod(values[inst.arg0], values[inst.arg1], header.order);
            }
            case 4u: {
                values[inst_idx] = neg_mod(values[inst.arg0], header.order);
            }
            default: {
                values[inst_idx] = mul_mod(values[inst.arg0], values[inst.arg1], header.order);
            }
        }
    }

    for (var output_idx = 0u; output_idx < header.output_count; output_idx++) {
        let value = values[outputs[output_idx]];
        let scalar_base = output_idx * header.extension_degree;
        for (var coeff = 0u; coeff < header.extension_degree; coeff++) {
            acc[coeff] = add_mod(
                acc[coeff],
                mul_mod(value, batching_scalars[scalar_base + coeff], header.order),
                header.order,
            );
        }
    }

    let result_base = point_idx * header.extension_degree;
    for (var coeff = 0u; coeff < header.extension_degree; coeff++) {
        results[result_base + coeff] = acc[coeff];
    }
}
"#;

pub(super) fn execute(payload: &PackedMainGpuPayload<'_>) -> Result<Vec<u32>, String> {
    let runtime = runtime()?;
    let device = &runtime.device;
    let queue = &runtime.queue;
    let program_buffers = static_program_buffers(runtime, payload);
    let program_buffers = program_buffers
        .as_ref()
        .expect("wgpu program buffers should be initialized");

    let header_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("whirlaway-air-wgpu-header"),
        contents: cast_slice(std::slice::from_ref(&payload.header)),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let batching_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("whirlaway-air-wgpu-batching"),
        contents: cast_slice(&payload.batching_scalars),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let flat_inputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("whirlaway-air-wgpu-flat-inputs"),
        contents: cast_slice(&payload.flat_inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let results_size = (payload.result_len_u32s() * std::mem::size_of::<u32>()) as u64;
    let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("whirlaway-air-wgpu-results"),
        size: results_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("whirlaway-air-wgpu-staging"),
        size: results_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("whirlaway-air-wgpu-bind-group"),
        layout: &runtime.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: header_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: program_buffers.instruction_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: program_buffers.constants_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: program_buffers.outputs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: batching_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: flat_inputs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: results_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("whirlaway-air-wgpu-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("whirlaway-air-wgpu-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&runtime.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = payload.header.point_count.div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&results_buffer, 0, &staging_buffer, 0, results_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(
        wgpu::MapMode::Read,
        move |result: Result<(), wgpu::BufferAsyncError>| {
            let _ = sender.send(result);
        },
    );
    let _ = device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|err| format!("map_async channel failed: {err}"))?
        .map_err(|err| format!("map_async failed: {err}"))?;

    let mapped = slice.get_mapped_range();
    let result = cast_slice::<u8, u32>(&mapped).to_vec();
    drop(mapped);
    staging_buffer.unmap();
    Ok(result)
}
