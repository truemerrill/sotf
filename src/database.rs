use crate::genetic::{Generation, GeneticStrategy};
use rusqlite::{params, Connection, Error};

static SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS generation (
    id INTEGER PRIMARY KEY,
    number INTEGER
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
    metric TEXT,
    value NUMERIC,
    FOREIGN KEY (strategy_id) REFERENCES strategy(id)
);

CREATE TABLE IF NOT EXISTS generation_data (
    id INTEGER PRIMARY KEY,
    generation_id INTEGER,
    metric TEXT,
    value NUMERIC,
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

pub fn write_generation(conn: &Connection, generation: &Generation) -> Result<(), Error> {
    conn.execute(
        "INSERT INTO generation (number) VALUES (?1)",
        (generation.number,),
    )?;
    let generation_id = conn.last_insert_rowid();
    write_strategies(conn, &generation.strategies, generation_id)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::game::Payoff;
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

        let sim = Simulation::default();
        // let generation = Generation::new(&sim);
        // write_generation(&conn, &generation)?;
        Ok(())
    }
}
