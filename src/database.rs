use crate::genetic::{Generation, GeneticStrategy, Simulation};
use crate::stats::{max, mean, min, stdev};
use rusqlite::{params, Connection, Error};

static SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS simulation (
    id INTEGER PRIMARY KEY,
    num_population INTEGER,
    num_games INTEGER,
    num_generations INTEGER
);

CREATE TABLE IF NOT EXISTS generation (
    id INTEGER PRIMARY KEY,
    number INTEGER,
    simulation_id INTEGER,
    FOREIGN KEY (simulation_id) REFERENCES simulation(id)
);

CREATE TABLE IF NOT EXISTS strategy (
    id INTEGER PRIMARY KEY,
    prior BLOB,
    strategy BLOB,
    generation_id INTEGER,
    FOREIGN KEY (generation_id) REFERENCES generation(id)
);

CREATE TABLE IF NOT EXISTS strategy_data (
    id INTEGER PRIMARY KEY,
    strategy_id INTEGER,
    score NUMERIC,
    score_avg NUMERIC,
    score_all_cooperate_avg NUMERIC,
    score_all_defect_avg NUMERIC,
    score_tit_for_tat_avg NUMERIC,
    score_numerov_avg NUMERIC,
    FOREIGN KEY (strategy_id) REFERENCES strategy(id)
);

CREATE TABLE IF NOT EXISTS generation_data (
    id INTEGER PRIMARY KEY,
    generation_id INTEGER,
    score_avg NUMERIC,
    score_std NUMERIC,
    score_max NUMERIC,
    score_min NUMERIC,
    FOREIGN KEY (generation_id) REFERENCES generation(id)
);
";

pub fn setup(conn: &Connection) -> Result<(), Error> {
    conn.execute_batch(SCHEMA)
}

pub fn write_strategies(
    conn: &Connection,
    strategies: &Vec<GeneticStrategy>,
    generation_id: i64,
) -> Result<(), Error> {
    let mut stmnt = conn.prepare_cached(
        "INSERT INTO strategy (prior, strategy, generation_id)
         VALUES (?1, ?2, ?3)",
    )?;

    for strat in strategies.iter() {
        stmnt.execute(params![strat.prior, strat.strategy, generation_id])?;
    }
    Ok(())
}

pub fn write_generation(
    conn: &Connection,
    generation: &Generation,
    simulation_id: i64,
) -> Result<i64, Error> {
    conn.execute(
        "INSERT INTO generation (number, simulation_id) VALUES (?1, ?2)",
        (generation.number, simulation_id),
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn write_generation_data(
    conn: &Connection,
    simulation: &Simulation,
    generation: &Generation,
    generation_id: i64,
) -> Result<i64, Error> {
    let num_matches = simulation.num_population - 1;
    let norm = (simulation.num_games * num_matches) as f64;
    let score_avg = mean(&generation.scores);
    let score_std = stdev(&generation.scores, score_avg);
    let score_max = max(&generation.scores);
    let score_min = min(&generation.scores);

    conn.execute(
        "INSERT INTO generation_data (
            generation_id,
            score_avg,
            score_std,
            score_max,
            score_min
        ) VALUES (?1, ?2, ?3, ?4, ?5)",
        (
            generation_id,
            score_avg / norm,
            score_std / norm,
            score_max / norm,
            score_min / norm,
        ),
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn write_simulation(conn: &Connection, simulation: &Simulation) -> Result<i64, Error> {
    conn.execute(
        "INSERT INTO simulation (num_population, num_games, num_generations)
         VALUES (?1, ?2, ?3)",
        (
            simulation.num_population,
            simulation.num_games,
            simulation.num_generations,
        ),
    )?;
    Ok(conn.last_insert_rowid())
}

#[cfg(test)]
mod tests {
    use crate::genetic::Simulation;

    use super::*;
    use rusqlite::Result;

    #[test]
    fn test_setup() -> Result<()> {
        let conn = Connection::open_in_memory()?;
        setup(&conn)
    }

    #[test]
    fn test_write_generation() -> Result<(), Error> {
        let conn = Connection::open("test.db")?;
        setup(&conn)?;

        let mut sim = Simulation::default();
        sim.num_generations = 3;
        sim.num_population = 10;
        let sim_id = write_simulation(&conn, &sim)?;

        for gen in sim.random().into_iter() {
            let gen_id = write_generation(&conn, &gen, sim_id)?;
            write_strategies(&conn, &gen.strategies, gen_id)?;
        }

        Ok(())
    }
}
