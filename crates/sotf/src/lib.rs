mod database;
mod game;
mod genetic;
mod stats;

pub use crate::database::{setup, write_generation, write_simulation, write_strategy};
pub use crate::game::{
    game, round, tournament, AllCooperate, AllDefect, Choice, Pavlov, Payoff, Strategy, TitForTat,
};
pub use crate::genetic::{Generation, GeneticStrategy, Simulation};
// pub use crate::stats::{fraction_betray, fraction_defect, fraction_retaliate, fraction_win};
