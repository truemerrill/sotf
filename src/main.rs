use rusqlite::{Connection, Error};
use sotf::{setup, write_generation, write_simulation, write_strategy, Simulation};

fn main() -> Result<(), Error> {
    let conn = Connection::open("output.db")?;
    setup(&conn)?;

    let mut sim = Simulation::default();
    sim.num_generations = 100;
    sim.num_games = 200;
    sim.num_population = 500;
    sim.selection_rate = 1.0;
    sim.mutation_rate = 0.05;

    let idx_best = 0;
    let idx_worst = sim.num_population - 1;
    let idx_median = idx_worst / 2;

    let sim_id = write_simulation(&conn, &sim)?;

    for gen in sim.random() {
        let gen_id = write_generation(&conn, &sim, sim_id, &gen)?;

        // Write the best, worst, and median, strategies
        write_strategy(&conn, &gen, gen_id, idx_best)?;
        write_strategy(&conn, &gen, gen_id, idx_median)?;
        write_strategy(&conn, &gen, gen_id, idx_worst)?;

        println!("Generation {:?}", gen_id);
    }
    Ok(())
}
