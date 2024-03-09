//! Functions and Structs implementing a Prisoner's Dilemma strategy that
//! can be trained by a Genetic Algorithm.

use crate::game::{Choice, Payoff, Strategy};
use rand::RngCore;

/// A Prisoner's Dilemma strategy that can be trained by GA.
///
/// GeneticStrategies are deterministic strategies that encode their ruleset
/// as bit array, called the strategy array.  Additionally, GeneticStrategies
/// track the previous choices of both players in a bitfield, called the
/// history.  The history is also used as an index used to lookup the next
/// player move from the strategy array.  For example, if the history array
/// stores the numeric value 183, then the value of of the 183th bit (zero
/// indexed) in the strategy array encodes the next move, with 0 corresponding
/// to `Choice::Cooperate` and 1 corresponding to `Choice::Defect`.
///
/// In this implementation, the history is 8 bits long, which means the
/// strategy tracks the last four moves of both players.  There are 2^8 = 256
/// possible histories, so the strategy array is 256 bits or 32 bytes long.
///
/// At the start of a round, the history bitfield is initialized to the prior.
/// The prior can be thought of some incoming bias the strategy starts with
/// at the beginning of each round.
///
/// The prior and strategy array together form the *chromosome* for the
/// genetic algorithm.
///
#[derive(Debug, Clone, Copy)]
pub struct GeneticStrategy {
    prior: u8,
    strategy: [u8; 32],
    history: u8,
}

impl GeneticStrategy {
    /// Create a new GeneticStrategy with a prior and strategy array
    pub fn new(prior: u8, strategy: [u8; 32]) -> GeneticStrategy {
        GeneticStrategy {
            prior: prior,
            strategy: strategy,
            history: prior,
        }
    }

    /// Create a new GeneticStrategy with a random prior and strategy array
    pub fn random() -> GeneticStrategy {
        let prior: u8 = rand::random();
        let mut strategy = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut strategy);
        GeneticStrategy::new(prior, strategy)
    }
}

impl Strategy for GeneticStrategy {
    fn choose(&self) -> Choice {
        match lookup(self.history, &self.strategy) {
            0x00 => Choice::Cooperate,
            _ => Choice::Defect,
        }
    }

    fn update(&mut self, player_choice: &Choice, opponent_choice: &Choice) -> () {
        self.history = append(self.history, player_choice);
        self.history = append(self.history, opponent_choice);
    }

    fn reset(&mut self) -> () {
        self.history = self.prior;
    }
}

/// Append choice to the end of history
///
/// The oldest choice in the history will be dropped from the bitfield.
fn append(history: u8, choice: &Choice) -> u8 {
    let x: u8 = match choice {
        Choice::Cooperate => 0x00,
        Choice::Defect => 0x01,
    };

    (history << 1) + x
}

/// Lookup the value of the `index`'th bit from the data array.
///
/// Arguments:
///
/// * index (u8): The bit index
/// * data (&[u8; 32]): The data array
///
/// Returns:
///
/// u8: An unsigned integer encoding the value of the bit.  A value of 0x00
///     corresponds to the bit value 0.  A value of 0x80 corresponds to the
///     bit value 1.  No other outputs should be possible.
///
fn lookup(index: u8, data: &[u8; 32]) -> u8 {
    let x = (index / 8) as usize;
    let byte = data[x];
    (byte << (index % 8)) & 0x80
}

/// Simulation parameters.  These are static over the course of a run.
#[derive(Debug)]
pub struct Parameters {
    payoff: Payoff,
    population_size: usize,
    mutation_rate: f64,
    num_rounds: usize,
}

/// A single generation of the simulation.
#[derive(Debug)]
pub struct Generation<'a> {
    number: usize,
    strategies: Vec<GeneticStrategy>,
    parameters: &'a Parameters,
}

// impl Generation<'a> {
//     pub fn select(&self) -> {

//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{game, round, Payoff};

    #[test]
    fn test_append() {
        let h1 = 0x01;

        let h2 = append(h1, &Choice::Cooperate);
        assert_eq!(h2, 0x02);

        let h3 = append(h1, &Choice::Defect);
        assert_eq!(h3, 0x03);

        assert_eq!(append(0xFF, &Choice::Cooperate), 0xFE);
    }

    #[test]
    fn test_choice() {
        let mut s: [u8; 32] = [0; 32];
        s[0] = 0b1010_0000;
        s[19] = 0b0000_1011;

        let mut c = GeneticStrategy::new(0x00, s);

        c.history = 0;
        assert_eq!(c.choose(), Choice::Defect);
        c.history = 1;
        assert_eq!(c.choose(), Choice::Cooperate);
        c.history = 2;
        assert_eq!(c.choose(), Choice::Defect);

        c.history = 8 * 19 + 4;
        assert_eq!(c.choose(), Choice::Defect);
        c.history = 8 * 19 + 5;
        assert_eq!(c.choose(), Choice::Cooperate);
        c.history = 8 * 19 + 6;
        assert_eq!(c.choose(), Choice::Defect);
        c.history = 8 * 19 + 7;
        assert_eq!(c.choose(), Choice::Defect);
        c.history = 8 * 20;
        assert_eq!(c.choose(), Choice::Cooperate);
    }

    #[test]
    fn test_random() {
        let _c = GeneticStrategy::random();
        // println!("{:?}", c);
    }

    #[test]
    fn test_game() {
        let mut s1: [u8; 32] = [0; 32];
        s1[0] = 0b1000_0000;
        let s2 = [0u8; 32];

        // Player one should defect and player 2 should cooperate.
        let mut p1 = GeneticStrategy::new(0x00, s1);
        let mut p2 = GeneticStrategy::new(0x00, s2);

        let payoff = Payoff::default();
        let result = game(&mut p1, &mut p2, &payoff);

        assert_eq!(result.0, payoff.temptation);
        assert_eq!(result.1, payoff.sucker);
    }

    #[test]
    fn test_round() {
        let mut all_cooperate = GeneticStrategy::new(0x00, [0u8; 32]);
        let mut all_defect = GeneticStrategy::new(0x00, [0xFFu8; 32]);
        let payoff = Payoff::default();

        let result = round(&mut all_cooperate, &mut all_defect, &payoff, 100);

        assert_eq!(result.0, 100.0 * payoff.sucker);
        assert_eq!(result.1, 100.0 * payoff.temptation);
    }
}
