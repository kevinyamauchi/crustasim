#![feature(autodiff)]
use std::autodiff::*;


fn compute_attraction_force(coordinates: &Vec<(f32, f32)>, coefficent: f32) -> Vec<(f32, f32)> {
    // Get the number of agents
    let n_agents = coordinates.len();

    // Initialize a vector to hold the attraction forces
    // Shape (n_agents, 2)
    let mut attraction_forces = vec![(0.0f32, 0.0f32); n_agents];

    // Compute the attraction forces
    for i in 0..n_agents {
        for j in 0..n_agents {
            if i != j {
                let dx = coordinates[j].0 - coordinates[i].0;
                let dy = coordinates[j].1 - coordinates[i].1;
                let distance = (dx * dx + dy * dy).sqrt();
                if distance > 0.0 {
                    let force_magnitude = coefficent / distance;
                    attraction_forces[i].0 += force_magnitude * (dx / distance);
                    attraction_forces[i].1 += force_magnitude * (dy / distance);
                }
            }
        }
    }

    return attraction_forces;
}

fn compute_repulsion_force(coordinates: &Vec<(f32, f32)>, coefficent: f32) -> Vec<(f32, f32)> {
    // Get the number of agents
    let n_agents = coordinates.len();

    // Initialize a vector to hold the repulsion forces
    // Shape (n_agents, 2)
    let mut repulsion_forces = vec![(0.0f32, 0.0f32); n_agents];

    // Compute the repulsion forces
    for i in 0..n_agents {
        for j in 0..n_agents {
            if i != j {
                let dx = coordinates[j].0 - coordinates[i].0;
                let dy = coordinates[j].1 - coordinates[i].1;
                let distance = (dx * dx + dy * dy).sqrt();
                if distance > 0.0 {
                    let force_magnitude = coefficent / (distance * distance);
                    repulsion_forces[i].0 -= force_magnitude * (dx / distance);
                    repulsion_forces[i].1 -= force_magnitude * (dy / distance);
                }
            }
        }
    }

    return repulsion_forces;
}


fn forward(coordinates: &Vec<(f32, f32)>, attraction_coefficient: f32, repulsion_coefficient: f32, time_step: f32, n_steps: i16) -> Vec<(f32, f32)> {
    let mut updated_coordinates = coordinates.clone();

    for _step in 0..n_steps {
        let attraction_forces = compute_attraction_force(&updated_coordinates, attraction_coefficient);
        let repulsion_forces = compute_repulsion_force(&updated_coordinates, repulsion_coefficient);

        // Update the positions of the agents
        for i in 0..updated_coordinates.len() {
            let total_force_x = attraction_forces[i].0 + repulsion_forces[i].0;
            let total_force_y = attraction_forces[i].1 + repulsion_forces[i].1;

            updated_coordinates[i].0 += total_force_x * time_step;
            updated_coordinates[i].1 += total_force_y * time_step;
        }
    }
    let final_coordinates = updated_coordinates;
    return final_coordinates;
}


// loss function that computes the mean squared distance between agents
#[autodiff_reverse(d_loss, Const, Active, Const, Const, Const, Active)]
fn loss(coordinates: &Vec<(f32, f32)>, attraction_coefficient: f32, repulsion_coefficient: f32, time_step: f32, n_steps: i16) -> f32 {
    let final_coordinates = forward(coordinates, attraction_coefficient, repulsion_coefficient, time_step, n_steps);

    let expected_distance: f32 = 2.0;
    let dx = final_coordinates[0].0 - final_coordinates[1].0;
    let dy = final_coordinates[0].1 - final_coordinates[1].1;
    let distance = (dx * dx + dy * dy).sqrt();
    let loss = ((distance - expected_distance) * (distance - expected_distance)) / expected_distance;

    return loss
}


fn main() {
    // Make the agent coordinates
    let coordinates = vec![(1.0, -0.8), (1.0, 0.8)];

    // Set the coefficients
    let attraction_coefficient: f32 = 10.0;
    let repulsion_coefficient: f32 = 10.0;

    // set the numerical parameters
    let time_step: f32 = 0.01;
    let n_steps: i16 = 100;

    // Run the forward simulation
    let (loss_value, d_loss_value) = d_loss(&coordinates, attraction_coefficient, repulsion_coefficient, time_step, n_steps, 1.0);

    println!("{:?}", loss_value);
    println!("{:?}", d_loss_value);
}
