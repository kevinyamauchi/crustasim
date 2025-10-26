// This is a small example of trying to use gradient-based optimization
// on a simulation to find parameters. The "simulation" is a
// mock 1D particle simulation in which a constance "force"
// is applied to one of the particles. The optimization task
// is to find the force value such that the particles are
// the separated by the target distance at the end of the
// simulation.
//
// To show how I intend this to work, I have manually implemented
// the gradient function (d_loss_manual).

// This is a mock forward simulation of particles in 1D.
// There are two particles. We apply a "force" with value coefficient
// to particle[1] for each time step.

#![feature(autodiff)]
use std::autodiff::*;

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
#[autodiff_reverse(d_loss_automatic, Const, Active, Const, Const, Active)]
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
    let mut coefficient: f32 = 1.0;

    // numerical parameters
    let n_forward_steps: u16 = 5;
    let n_optimizer_steps: u16 = 250;
    let learning_rate: f32 = 0.001;

    // Set the target distance
    let target_distance: f32 = 3.0;

    // Use the conjugate gradient method to find the best
    // coefficient such that the particles end up
    // target_distance apart.
    // We expect the coefficient to be:
    // target_distance / n_forward_steps = 3/5 = 0.6
    for optimizer_step in 0..n_optimizer_steps {
        // make a mutable copy of the particle positions
        // (not sure if this is the best way...)
        let mut coordinates = initial_coordinates;

        // Do compute the loss and its derivative using
        // the manual implementation
        // let (loss_value, d_loss_value) = d_loss_manual(
        //     &mut coordinates,
        //     coefficient,
        //     n_forward_steps,
        //     target_distance
        // );

        // Compute the loss and its derivative using
        // automatic differentiation
        let (loss_value, d_loss_value) = d_loss_automatic(
            &mut coordinates,
            coefficient,
            n_forward_steps,
            target_distance,
            1.0
        );

        println!("Optimizer step {}", optimizer_step);
        println!("    Coeff: {}, Loss: {},  d_Loss: {}", coefficient, loss_value, d_loss_value);
        println!("    Final coordinates: {:?}", coordinates);

        // update the coefficient using the conjugate gradient
        coefficient += -d_loss_value * learning_rate;
    }


}