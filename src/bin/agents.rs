use plotters::prelude::*;

fn plot(coordinates: Vec<(f32, f32)>) {
    let root_area = BitMapBackend::new("plot.png", (600, 400))
        .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Line Plot Demo", ("sans-serif", 40))
        .build_cartesian_2d(-10f32..10f32, 0f32..100f32)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    // let coords = (-10..=10).map(|x| (x, x* x));

    ctx.draw_series(
        LineSeries::new(coordinates.iter().copied(), &GREEN)
    ).unwrap();
}

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

fn main() {
    // Make the agent coordinates
    let mut coordinates = vec![(1.0, -0.8), (1.0, 0.8)];

    // Set the coefficients
    let attraction_coefficient: f32 = 10.0;
    let repulsion_coefficient: f32 = 20.0;

    // set the numerical parameters
    let time_step: f32 = 0.01;
    let n_steps: i16 = 100;

    // use the forward euler method to update the positions of the agents
    for _step in 0..n_steps {
        let attraction_forces = compute_attraction_force(&coordinates, attraction_coefficient);
        let repulsion_forces = compute_repulsion_force(&coordinates, repulsion_coefficient);

        // Update the positions of the agents
        for i in 0..coordinates.len() {
            let total_force_x = attraction_forces[i].0 + repulsion_forces[i].0;
            let total_force_y = attraction_forces[i].1 + repulsion_forces[i].1;

            coordinates[i].0 += total_force_x * time_step;
            coordinates[i].1 += total_force_y * time_step;
        }
    }

    println!("{:?}", coordinates);

    // let coordinates = vec_2d;
    // let coordinates = vec![(0.0, 0.0), (5.0, 5.0), (8.0, 7.0)];
    // plot(coordinates);
}
