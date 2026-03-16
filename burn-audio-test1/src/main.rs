use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

// Permet de faire un gain + empêche le son de monter plus haut 
// Sur des samples simuler et sur un backend NdArray (pas un cubecl ou quelque chose comme ça)

type B = NdArray;

fn main() {
    let device = B::Device::default();

    let samples: Vec<f32> = vec![-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9, 1.2];
    let x = Tensor::<B, 1>::from_data(TensorData::new(samples, [8]), &device); // Le backend c'est B, 1 le nombre de dimension // sample --> liste de nomre, [8] 1 dimension, 8 de long

    let y = (x * 2.0).tanh();

    println!("Hello, world!");
}
