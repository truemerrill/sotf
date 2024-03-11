mod database;
mod game;
mod genetic;

mod scratch;

pub use crate::database::{setup, write_generation, write_strategies};
pub use crate::game::{
    game, round, tournament, AllCooperate, AllDefect, Choice, Pavlov, Payoff, Strategy, TitForTat,
};
pub use crate::genetic::{Generation, GeneticStrategy, Simulation};
