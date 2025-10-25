#![feature(autodiff)]
use std::autodiff::*;

fn step(x_coordinates: &[f32; 2], y_coordinates: &[f32; 2], coefficient: f32) -> ([f32; 2], [f32; 2]) {
    let new_x_coordinates: [f32; 2] = std::array::from_fn( |i| {
            x_coordinates[i]
        }
    );
    let new_y_coordinates: [f32; 2] = std::array::from_fn( |i| {
            y_coordinates[i] + i as f32 * coefficient
        }
    );
    return (new_x_coordinates, new_y_coordinates)
}


fn forward(coefficient: f32, n_forward_iterations: u16) -> ([f32; 2], [f32; 2]) {
    let mut x_coordinates: [f32; 2] = [0.0, 0.0];
    let mut y_coordinates: [f32; 2] = [-0.5, 0.5];

    for _ in 0..n_forward_iterations {
        (x_coordinates, y_coordinates) = step(&x_coordinates, &y_coordinates, coefficient);
    }

    return (x_coordinates, y_coordinates);
}

#[autodiff_reverse(d_loss, Active, Const, Active)]
fn loss(coefficient: f32, target_value: f32) -> f32 {
    let n_forward_iterations: u16 = 10;
    let (_final_x, final_y) = forward(coefficient, n_forward_iterations);

    let dy = final_y[1] - final_y[0];
    let distance = dy.abs();

    let loss = ((distance - target_value) * (distance - target_value)) / target_value;

    return loss;
}

fn main(){
    let coefficient: f32 = 0.05;
    let n_forward_iterations: u16 = 10;

    // let (final_x, final_y) = forward(coefficient, n_forward_iterations);
    let (loss_value, d_loss_value) = d_loss(coefficient, 2.0, 1.0);

    println!("Loss value: {:?}", loss_value);
    println!("Derivative of Loss w.r.t. coefficient: {:?}", d_loss_value);

}