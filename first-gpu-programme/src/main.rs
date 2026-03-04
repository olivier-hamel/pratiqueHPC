use cubecl::cube;
use cubecl::{CubeCount, CubeDim, CubeElement, Runtime};
use cubecl::prelude::*;

#[cube(launch)]
fn kernel_double_numbers(input: &Array<u32>, scale: u32, output: &mut Array<u32>) {
    let i = ABSOLUTE_POS;
    output[i] = input[i] * scale;
}

fn main() {
    let device = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    let input_data = (1..11).collect::<Vec<u32>>();
    println!("Input: {:?}", input_data);
    let num_elements = input_data.len();
    let zeros = vec![0u32; num_elements];

    let input_data_gpu = client.create_from_slice(u32::as_bytes(&input_data));
    let output_data_gpu = client.create_from_slice(u32::as_bytes(&zeros));
    
    unsafe {
        kernel_double_numbers::launch::<cubecl::wgpu::WgpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(num_elements as u32),
            ArrayArg::from_raw_parts::<u32>(&input_data_gpu, num_elements, 1),
            ScalarArg::new(3),
            ArrayArg::from_raw_parts::<u32>(&output_data_gpu, num_elements, 1),
        ).expect("kernel launch failed");
    }

    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Output: {:?}", output);
}