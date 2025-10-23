// to run: RUSTC="/local1/rust_source/rust/build/host/stage1/bin/rustc" RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme run --bin derivative

#![feature(autodiff)]
use std::autodiff::*;


fn quadratic(x: f32, coefficient: f32) -> f32 {
    return coefficient * x * x
}

// Define the loss function and its derivative using reverse-mode autodiff
// The expected coefficient is 3.0.
// This treats the first argument (x) as a constant
// and the second argument (coefficient) as active
#[autodiff_reverse(d_loss, Const, Active, Active)]
fn loss(x: f32, coefficient: f32) -> f32 {
    let y = quadratic(x, coefficient);

    let expected_value = 3.0 * x * x;

    // Using Mean Squared Error loss
    let loss = ((y - expected_value) * (y - expected_value)) / expected_value;

    return loss;
}


fn main() {

    let mut coefficient_guess: f32 = 1.0;
    let learning_rate: f32 = 0.1;
    let n_iterations: i32 = 100;

    for _iteration in 0..n_iterations {

        // make a forward pass and get the loss and its derivative
        let (loss_value, d_loss_value) = d_loss(3.0, coefficient_guess, 0.0);

        // Update the coefficient guess using gradient descent
        coefficient_guess -= learning_rate * d_loss_value;

        println!("Loss: {}, Coefficient Guess: {}", loss_value, coefficient_guess);
    }

    println!("Final Coefficient Guess: {}", coefficient_guess);
}