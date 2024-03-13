use rusqlite::{Connection, Error};
use sotf::{setup, write_generation, write_generation_data, write_simulation, Simulation};

fn main() -> Result<(), Error> {
    let conn = Connection::open("output.db")?;
    setup(&conn)?;

    let mut sim = Simulation::default();
    sim.num_generations = 100;
    sim.num_games = 200;
    sim.num_population = 500;
    sim.selection_rate = 1.0;
    sim.mutation_rate = 0.05;
    let sim_id = write_simulation(&conn, &sim)?;

    for gen in sim.random().into_iter() {
        let gen_id = write_generation(&conn, &gen, sim_id)?;
        write_generation_data(&conn, &sim, &gen, gen_id)?;
        println!("Generation {:?}", gen_id);
    }
    Ok(())
}
