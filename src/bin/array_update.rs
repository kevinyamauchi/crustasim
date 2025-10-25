
#![feature(autodiff)]
use std::autodiff::*;

const N: usize = 2; // Number of agents


struct Agent {
    position_x: f32,
    position_y: f32,
}

struct WorldState {
    agents: [Agent; N],
    iteration_index: usize,
}

/// Intermediate structure for accumulated forces
pub struct AgentForces {
    pub fx: f32,
    pub fy: f32,
}


fn compute_forces(
    world_state: &WorldState,
    attraction_coefficient: f32,
    repulsion_coefficient: f32,
) -> [AgentForces; N]  {

    std::array::from_fn( |i| {
        AgentForces {
            fx: 0.0,
            fy: 0.0
        }
    }
    )

    //
    // for i in 0..N {
    //     for j in 0..N {
    //         if i != j {
    //             let dx = world_state.agents[j].position_x - world_state.agents[i].position_x;
    //             let dy = world_state.agents[j].position_y - world_state.agents[i].position_y;
    //             let distance = (dx * dx + dy * dy).sqrt();
    //             if distance > 0.0 {
    //
    //                 let mut force_x: f32 = 0.0;
    //                 let mut force_y: f32 = 0.0;
    //
    //                 // Attraction force
    //                 let attraction_magnitude = attraction_coefficient / distance;
    //                 force_x += attraction_magnitude * (dx / distance);
    //                 force_y += attraction_magnitude * (dy / distance);
    //
    //                 // Repulsion force
    //                 let repulsion_magnitude = repulsion_coefficient / (distance * distance);
    //                 force_x -= repulsion_magnitude * (dx / distance);
    //                 force_y -= repulsion_magnitude * (dy / distance);
    //
    //                 all_forces[i].fx += force_x;
    //                 all_forces[i].fy += force_y;
    //             }
    //         }
    //     }
    // }
    //
    // return all_forces

}

fn integrate_agents(
    world_state: &WorldState,
    forces: &[AgentForces; N],
    time_step: f32,
) -> [Agent; N] {
    let mut new_agents = [
        Agent {
            position_x: 0.0,
            position_y: -0.5,
        },
        Agent {
            position_x: 0.0,
            position_y: 0.5,
        },
    ];


    for i in 0..N {
        new_agents[i].position_x += forces[i].fx * time_step;
        new_agents[i].position_y += forces[i].fy * time_step;
    }

    return new_agents;
}


fn step(initial_world_state: &WorldState, attraction_coefficient: f32, repulsion_coefficient: f32, time_step: f32,) -> WorldState {
    // Compute the forces
    let forces = compute_forces(
        initial_world_state,
        attraction_coefficient,
        repulsion_coefficient,
    );

    // Update positions based on forces
    // let new_agents = integrate_agents(initial_world_state, &forces, time_step);

    let new_agents = [
        Agent {
            position_x: 0.0,
            position_y: -0.5,
        },
        Agent {
            position_x: 0.0,
            position_y: -0.5,
        },
    ];

    WorldState {
        agents: new_agents,
        iteration_index: initial_world_state.iteration_index + 1,
    }

}

fn forward(attraction_coefficient: f32, repulsion_coefficient: f32, time_step: f32, n_steps: u16) -> WorldState{

    let initial_agents = [
        Agent {
            position_x: 0.0,
            position_y: -0.5,
        },
        Agent {
            position_x: 0.0,
            position_y: 0.5,
        },
    ];

    let mut state = WorldState{
        agents: initial_agents,
        iteration_index: 0
    };

    for _ in 0..n_steps {
        state = step(&state, attraction_coefficient, repulsion_coefficient, time_step);
    }

    return state;

}

#[autodiff_reverse(d_loss, Const, Active, Const, Const, Const, Active)]
fn loss(attraction_coefficient: f32, repulsion_coefficient: f32, time_step: f32, n_steps: u16, target_distance: f32) -> f32 {
    // Perform the forward simulation
    let final_state = forward(attraction_coefficient, repulsion_coefficient, time_step, n_steps);

    // Compute the distance between the two agents
    let dx = final_state.agents[1].position_x - final_state.agents[0].position_x;
    let dy = final_state.agents[1].position_y - final_state.agents[0].position_y;
    let distance = (dx * dx + dy * dy).sqrt();

    // Compute the loss as the squared difference from the target distance
    let loss_value = ((distance - target_distance) * (distance - target_distance)) / target_distance;

    return loss_value;

}



fn main() {

    // Set the coefficients
    let attraction_coefficient: f32 = 1.0;
    let mut repulsion_coefficient: f32 = 1.0;

    // set the numerical parameters
    let target_distance: f32 = 2.0;
    let time_step: f32 = 0.1;
    let n_forward_steps: u16 = 200;
    let n_optimizer_steps: u16 = 200;
    let learning_rate: f32 = 0.1;

    let (loss_value, d_loss_value) = d_loss(attraction_coefficient, repulsion_coefficient, time_step, n_forward_steps, target_distance, 1.0);

    println!("Loss value: {}", loss_value);
    println!("d Loss value: {}", d_loss_value);

}