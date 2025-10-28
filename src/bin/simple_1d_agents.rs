// This is a small example of trying to use automatic differentiation
// on a loss computed on the result of a simulation. This could be used
// to perform gradient-based optimization to find simulation parameters
// to achieve a desired outcome (e.g., a specific distribution of particles).
//
// To show show the expected result, I have manually implemented
// the gradient function (d_loss_manual).
//
// Run the example with: RUSTC="/local1/rust_source/rust/build/host/stage1/bin/rustc" RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme run --bin simple_reproducer_no_opt

#![feature(autodiff)]
use std::autodiff::*;

// This is a mock forward simulation of particles in 1D.
// There are two particles. We apply a "force" with value coefficient
// to particle[1] for each time step.
fn forward(x_coordinate: &mut [f32], coefficient: f32, n_steps: u16) {
    for _step in 0..n_steps{
        x_coordinate[1] += coefficient;
    }
}


// Here we have a simple loss that is the squared difference
// between the final distance between the two particles and
// a target distance.
// That is, L = (T - n * C)^2
// where T is the target distance and C is the coefficient.
// I only want to compute the derivative w.r.t. coefficient.
#[autodiff_reverse(d_loss_automatic, Duplicated, Active, Const, Const, Active)]
fn loss(
    x_coordinate: &mut [f32],
    coefficient: f32,
    n_steps: u16,
    target_distance: f32
) -> f32 {
    forward(x_coordinate, coefficient, n_steps);

    // compute the loss
    let distance: f32 = x_coordinate[1] - x_coordinate[0];
    (target_distance - distance) * (target_distance - distance)
}

// Derivative of the loss with respect to the coefficient
// The loss function L = (T - n * C)^2
// where T is the target distance and C is the coefficient.
// Thus, the derivative dL/dC = 2 (- T * n + n^2 * C)
fn d_loss_manual(
    x_coordinate: &mut [f32],
    coefficient: f32,
    n_steps: u16,
    target_distance: f32
) -> (f32, f32) {

    // Compute the forward and the loss
    let loss_value = loss(
        x_coordinate,
        coefficient,
        n_steps,
        target_distance
    );

    // manually calculate the derivative of the loss w.r.t coefficient
    let d_loss_value = (2.0 * n_steps as f32 * n_steps as f32 * coefficient) - (2.0 * target_distance * n_steps as f32);

    (loss_value, d_loss_value)
}


fn main() {

    let initial_coordinates: [f32; 2] = [0.0, 0.0];

    // The initial coefficient guess
    let coefficient: f32 = 1.0;

    // numerical parameters
    let n_forward_steps: u16 = 5;

    // Set the target distance
    let target_distance: f32 = 3.0;

    // Do compute the loss and its derivative using
    // the manual implementation
    let mut coordinates = initial_coordinates;
    let (loss_value_manual, d_loss_value_manual) = d_loss_manual(
        &mut coordinates,
        coefficient,
        n_forward_steps,
        target_distance
    );

    // Compute the loss and its derivative using
    // automatic differentiation
    let mut coordinates_auto = initial_coordinates;
    let mut coordinates_gradients: [f32; 2] = [0.0, 0.0];
    let (loss_value_auto, d_loss_value_auto) = d_loss_automatic(
        &mut coordinates_auto,
        &mut coordinates_gradients,
        coefficient,
        n_forward_steps,
        target_distance,
        1.0
    );

    println!("Expected loss value: 4");
    println!("    Manual Loss value: {:?}", loss_value_manual);
    println!("    Auto Diff Loss value: {:?}", loss_value_auto);
    println!("Expected d_Loss value: 20");
    println!("    Manual d_Loss value: {:?}", d_loss_value_manual);
    println!("    Auto Diff d_Loss value: {:?}", d_loss_value_auto);
    println!("Coordinates gradients: {:?}", coordinates_gradients);

}