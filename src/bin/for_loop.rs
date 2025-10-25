// Attempt to find the bug with performing autodiff through a loop
#![feature(autodiff)]
use std::autodiff::*;

fn forward(x: f32, value_to_add: f32) -> f32{
    let mut accumulated_sum: f32 = x;
    accumulated_sum += value_to_add;
    accumulated_sum += value_to_add;
    accumulated_sum += value_to_add;

    return accumulated_sum;
}

fn forward_loop(x: f32, value_to_add: f32) -> f32{
    let mut accumulated_sum: f32 = x;
    for _i in 0..3 {
        accumulated_sum += value_to_add;
    }

    return accumulated_sum;
}

#[autodiff_reverse(d_loss, Const, Active, Const, Active)]
fn loss(x: f32, value_to_add: f32, expected_value_to_add: f32) -> f32 {
    let y = forward(x, value_to_add);
    let expected_value: f32 = x + 3.0 * expected_value_to_add;

    let loss = (y - expected_value) * (y - expected_value);

    return loss;
}

#[autodiff_reverse(d_loss_loop, Const, Active, Const, Active)]
fn loss_loop(x: f32, value_to_add: f32, expected_value_to_add: f32) -> f32 {
    let y = forward_loop(x, value_to_add);
    let expected_value: f32 = x + 3.0 * expected_value_to_add;

    let loss = (y - expected_value) * (y - expected_value);

    return loss;
}


fn main() {

    let input_value: f32 = 10.0;
    let value_guess: f32 = 3.0;
    let expected_value: f32 = 5.0;

    let result = forward(input_value, value_guess);
    let result_loop = forward_loop(input_value, value_guess);

    assert_eq!(result, result_loop);

    let (loss_value, d_loss_value) = d_loss(input_value, value_guess, expected_value, 1.0);
    let (loss_value_loop, d_loss_loop_value) = d_loss_loop(input_value, value_guess, expected_value, 1.0);
    println!("Result: {}", result);
    println!("Result Loop: {}", result_loop);
    println!("Loss Value: {}", loss_value);
    println!("Loss Value Loop: {}", loss_value_loop);
    println!("Derivative of Loss w.r.t. value_to_add: {}", d_loss_value);
    println!("Derivative of Loss Loop w.r.t. value_to_add: {}", d_loss_loop_value);

    assert_eq!(-36.0, d_loss_value);

}