// This is a small example of trying to use automatic differentiation
// on a loss computed on the result of a simulation. This could be used
// to perform gradient-based optimization to find simulation parameters
// to achieve a desired outcome (e.g., a specific distribution of particles).
//
// Run the example with: RUSTC="/local1/rust_source/rust/build/host/stage1/bin/rustc" RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme run --bin agents_1d

#![feature(autodiff)]
use std::autodiff::*;

// This is a mock forward simulation of particles in 1D.
// There are two particles. We apply an attraction and repulsion force
// to each particle
fn forward(
    x_coordinate: &mut [f32],
    attraction_coefficient: f32,
    repulsion_coefficient: f32,
    time_step: f32,
    n_steps: u16
) {
    for _step in 0..n_steps{
        let dx = x_coordinate[1] - x_coordinate[0];
        let distance = (dx * dx).sqrt();
        let attraction_force_magnitude = attraction_coefficient / distance;
        let repulsion_force_magnitude = repulsion_coefficient / (distance * distance);
        x_coordinate[0] += time_step * (attraction_force_magnitude - repulsion_force_magnitude);
        x_coordinate[1] -= time_step * (attraction_force_magnitude - repulsion_force_magnitude);
    }
}


// Here we have a simple loss that is the squared difference
// between the final distance between the two particles and
// a target distance.
#[autodiff_reverse(d_loss_automatic, Duplicated, Const, Active, Const, Const, Const, Active)]
fn loss(
    x_coordinate: &mut [f32],
    attraction_coefficient: f32,
    repulsion_coefficient: f32,
    time_step: f32,
    n_steps: u16,
    target_distance: f32
) -> f32 {
    forward(x_coordinate, attraction_coefficient, repulsion_coefficient, time_step, n_steps);

    // compute the loss
    let distance: f32 = x_coordinate[1] - x_coordinate[0];
    (target_distance - distance) * (target_distance - distance)
}

fn main() {

    let initial_coordinates: [f32; 2] = [-0.25, 0.25];

    // The initial coefficient guess
    let mut repulsion_coefficient: f32 = 1.0;
    let attraction_coefficient: f32 = 1.0;

    // numerical parameters
    let n_forward_steps: u16 = 10000;
    let time_step: f32 = 0.001;
    let n_optimizer_steps: u16 = 500;
    let learning_rate: f32 = 0.1;

    // Set the target distance
    let target_distance: f32 = 2.0;

    for optimizer_step in 0..n_optimizer_steps {

        // Compute the loss and its derivative using
        // automatic differentiation
        let mut coordinates_auto = initial_coordinates;
        let mut coordinates_gradients: [f32; 2] = [0.0, 0.0];
        let (loss_value_auto, d_loss_value_auto) = d_loss_automatic(
            &mut coordinates_auto,
            &mut coordinates_gradients,
            attraction_coefficient,
            repulsion_coefficient,
            time_step,
            n_forward_steps,
            target_distance,
            1.0
        );

        println!("Optimizer step: {}", optimizer_step);
        println!("    Coefficient: {}", repulsion_coefficient);
        println!("    Loss: {}, d_Loss: {}", loss_value_auto, d_loss_value_auto);
        println!("    Coordinates: {:?}", coordinates_auto);

        // update the coefficient using the conjugate gradient method
        repulsion_coefficient -= learning_rate * d_loss_value_auto;
    }
}