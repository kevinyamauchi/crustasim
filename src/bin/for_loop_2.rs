// Attempt to find the bug with performing autodiff through a loop
// This one uses the loop to mutate values in a vector
#![feature(autodiff)]
use std::autodiff::*;

struct Agent {
    position: [f32; 2],
    force: [f32; 2],
}


fn compute_attraction_force(agents: &mut [Agent], coefficient: f32) {
    // Get the distance between the two agents
    let dx: f32 = agents[1].position[0] - agents[0].position[0];
    let dy: f32 = agents[1].position[1] - agents[0].position[1];
    let distance: f32 = (dx * dx + dy * dy).sqrt();
    let force_magnitude = coefficient / distance;

    // Set the forces
    agents[0].force[0] += force_magnitude * (dx / distance);
    agents[0].force[1] += force_magnitude * (dy / distance);

    agents[1].force[0] -= force_magnitude * (dx / distance);
    agents[1].force[1] -= force_magnitude * (dy / distance);

}


fn compute_repulsion_force(agents: &mut [Agent], coefficient: f32){
    // Compute the distance between the two agents
    let dx: f32 = agents[1].position[0] - agents[0].position[0];
    let dy: f32 = agents[1].position[1] - agents[0].position[1];
    let distance: f32 = (dx * dx + dy * dy).sqrt();
    let force_magnitude: f32 = coefficient / (distance * distance);

    // set the forces
    agents[0].force[0] -= force_magnitude * (dx / distance);
    agents[0].force[1] -= force_magnitude * (dy / distance);

    agents[1].force[0] += force_magnitude * (dx / distance);
    agents[1].force[1] += force_magnitude * (dy / distance);

}


fn forward(agents: &mut [Agent], attraction_coefficient: f32, repulsion_coefficient: f32, time_step: f32, n_steps: i16) {
    for _step in 0..n_steps {
        compute_attraction_force(agents, attraction_coefficient);
        compute_repulsion_force(agents, repulsion_coefficient);

        // update positions based on forces
        agents[0].position[0] += agents[0].force[0] * time_step;
        agents[0].position[1] += agents[0].force[1] * time_step;
        agents[1].position[0] += agents[1].force[0] * time_step;
        agents[1].position[1] += agents[1].force[1] * time_step;

        // reset the forces
        agents[0].force[0] = 0.0;
        agents[0].force[1] = 0.0;
        agents[1].force[0] = 0.0;
        agents[1].force[1] = 0.0;

    }

}

#[autodiff_reverse(d_loss, Const, Active, Const, Const, Const, Active)]
fn loss(
    attraction_coefficient: f32,
    repulsion_coefficient: f32,
    time_step: f32,
    n_steps: i16,
    target_distance: f32
) -> f32 {

    let mut agents = [
        Agent {
            position: [0.0, -0.5],
            force: [0.0, 0.0],
        },
        Agent {
            position: [0.0, 0.5],
            force: [0.0, 0.0],
        },
    ];

    // run the forward simulation
    forward(&mut agents, attraction_coefficient, repulsion_coefficient, time_step, n_steps);

    // compute the distance between the two agents
    let dx = agents[1].position[0] - agents[0].position[0];
    let dy = agents[1].position[1] - agents[0].position[1];
    let distance = (dx * dx + dy * dy).sqrt();

    // compute the loss as the squared difference from the target distance
    let loss_value = ((distance - target_distance) * (distance - target_distance)) / target_distance;

    return loss_value;
}

fn main() {
    // let x = vec![1.0, 2.0];
    // let coefficient_guess: f32 = 3.0;
    // let expected_value = vec![5.0, 6.0];
    //
    // let (loss_value, d_loss_value) = d_loss(x, coefficient_guess, expected_value, 1.0);
    //
    // println!("Loss Value: {}", loss_value);
    // println!("d Loss Value: {}", d_loss_value);

    let mut agents = [
        Agent {
            position: [0.0, -0.5],
            force: [0.0, 0.0],
        },
        Agent {
            position: [0.0, 0.5],
            force: [0.0, 0.0],
        },
    ];

    // Set the coefficients
    let attraction_coefficient: f32 = 1.0;
    let mut repulsion_coefficient: f32 = 1.0;

    // set the numerical parameters
    let target_distance: f32 = 2.0;
    let time_step: f32 = 0.1;
    let n_forward_steps: i16 = 200;
    let n_optimizer_steps: i16 = 200;
    let learning_rate: f32 = 0.1;

    for _optimizer_iteration in 0..n_optimizer_steps {
        let (loss_value, d_loss_value) = d_loss(attraction_coefficient, repulsion_coefficient, time_step, n_forward_steps, target_distance, 1.0);

        // Update the coefficient guess using gradient descent
        repulsion_coefficient -= learning_rate * d_loss_value;

        println!("Loss Value: {:?}", loss_value);
        println!("Repulsion Coefficient Guess: {:?}", repulsion_coefficient);
    }

    println!("Final Repulsion Coefficient Guess: {:?}", repulsion_coefficient);


    // let coefficient_guess: f32 = 3.0;
    // let (forward_value, d_forward_value) = d_forward(&agents, coefficient_guess, 1.0);
    //
    // println!("Forward Value: {:?}", forward_value);
    // println!("d Forward Value: {:?}", d_forward_value);

}