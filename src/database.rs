use crate::genetic::{Generation, Simulation};
use crate::stats::{fraction_betray, fraction_defect, fraction_retaliate, max, mean, min, stdev};
use rusqlite::{Connection, Error};

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
    rank INTEGER,
    score NUMERIC,
    fraction_win NUMERIC,
    fraction_defect NUMERIC,
    fraction_betray NUMERIC,
    fraction_retaliate NUMERIC,
    FOREIGN KEY (strategy_id) REFERENCES strategy(id)
);

CREATE TABLE IF NOT EXISTS generation_data (
    id INTEGER PRIMARY KEY,
    generation_id INTEGER,
    score_avg NUMERIC,
    score_std NUMERIC,
    score_max NUMERIC,
    score_min NUMERIC,
    fraction_betray_avg NUMERIC,
    fraction_retaliate_avg NUMERIC,
    FOREIGN KEY (generation_id) REFERENCES generation(id)
);
";

pub fn setup(conn: &Connection) -> Result<(), Error> {
    conn.execute_batch(SCHEMA)
}

// pub fn write_strategies(
//     conn: &Connection,
//     strategies: &Vec<GeneticStrategy>,
//     generation_id: i64,
// ) -> Result<(), Error> {
//     let mut stmnt = conn.prepare_cached(
//         "INSERT INTO strategy (prior, strategy, generation_id)
//          VALUES (?1, ?2, ?3)",
//     )?;

//     for strat in strategies.iter() {
//         stmnt.execute(params![strat.prior, strat.strategy, generation_id])?;
//     }
//     Ok(())
// }

pub fn write_strategy(
    conn: &Connection,
    generation: &Generation,
    generation_id: i64,
    rank: usize,
) -> Result<i64, Error> {
    let strategy = &generation.strategies[rank];
    conn.execute(
        "INSERT INTO strategy (prior, strategy, generation_id)
         VALUES (?1, ?2, ?3)",
        (strategy.prior, strategy.strategy, generation_id),
    )?;
    let id = conn.last_insert_rowid();

    write_strategy_data(conn, generation, rank, id)?;
    Ok(id)
}

fn write_strategy_data(
    conn: &Connection,
    generation: &Generation,
    rank: usize,
    strategy_id: i64,
) -> Result<i64, Error> {
    let strategy = &generation.strategies[rank];
    let score = generation.scores[rank];
    let wins = generation.wins[rank];

    let num_rounds = (generation.strategies.len() - 1) as f64;
    let fraction_win = (wins as f64) / num_rounds;

    conn.execute(
        "INSERT INTO strategy_data (
            strategy_id,
            rank,
            score,
            fraction_win,
            fraction_defect,
            fraction_betray,
            fraction_retaliate
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        (
            strategy_id,
            rank,
            score,
            fraction_win,
            fraction_defect(strategy),
            fraction_betray(strategy),
            fraction_retaliate(strategy),
        ),
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn write_generation(
    conn: &Connection,
    simulation: &Simulation,
    simulation_id: i64,
    generation: &Generation,
) -> Result<i64, Error> {
    conn.execute(
        "INSERT INTO generation (number, simulation_id) VALUES (?1, ?2)",
        (generation.number, simulation_id),
    )?;
    let id = conn.last_insert_rowid();

    write_generation_data(conn, simulation, generation, id)?;
    Ok(id)
}

fn write_generation_data(
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

    let fraction_betray_avg = mean(
        &generation
            .strategies
            .iter()
            .map(|s| fraction_betray(s))
            .collect(),
    );

    let fraction_retaliate_avg = mean(
        &generation
            .strategies
            .iter()
            .map(|s| fraction_retaliate(s))
            .collect(),
    );

    conn.execute(
        "INSERT INTO generation_data (
            generation_id,
            score_avg,
            score_std,
            score_max,
            score_min,
            fraction_betray_avg,
            fraction_retaliate_avg
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        (
            generation_id,
            score_avg / norm,
            score_std / norm,
            score_max / norm,
            score_min / norm,
            fraction_betray_avg,
            fraction_retaliate_avg,
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
            let gen_id = write_generation(&conn, &sim, sim_id, &gen)?;
            //write_strategies(&conn, &gen.strategies, gen_id)?;
        }

        Ok(())
    }
}
