//! Functions and Structs implementing a Prisoner's Dilemma strategy that
//! can be trained by a Genetic Algorithm.

use crate::game::{tournament, Choice, Payoff, Strategy};
use rand::RngCore;
use std::rc::Rc;

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
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct GeneticStrategy {
    pub(crate) prior: u8,
    pub(crate) strategy: [u8; 32],
}

impl GeneticStrategy {
    /// Create a new GeneticStrategy with a prior and strategy array
    pub fn new(prior: u8, strategy: [u8; 32]) -> GeneticStrategy {
        GeneticStrategy {
            prior: prior,
            strategy: strategy,
        }
    }

    /// Create a new GeneticStrategy with a random prior and strategy array
    pub fn random() -> GeneticStrategy {
        let prior: u8 = rand::random();
        let mut strategy = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut strategy);
        GeneticStrategy::new(prior, strategy)
    }

    /// Get the chromosome as a byte array
    pub fn chromosome(&self) -> [u8; 33] {
        let mut ch = [0u8; 33];
        ch[0] = self.prior;
        for i in 0..32 {
            ch[i + 1] = self.strategy[i];
        }
        ch
    }
}

impl Strategy for GeneticStrategy {
    type History = u8;

    fn choose(&self, history: &Self::History) -> Choice {
        match lookup(*history, &self.strategy) {
            0x00 => Choice::Cooperate,
            _ => Choice::Defect,
        }
    }

    fn update(history: &mut Self::History, player_choice: &Choice, opponent_choice: &Choice) -> () {
        let new_history = append(append(*history, player_choice), opponent_choice);
        *history = new_history;
    }

    fn history(&self) -> Self::History {
        self.prior
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

fn mutate(byte: u8) -> u8 {
    let choice: u8 = rand::random();
    let mask = match choice % 8 {
        0 => 0b0000_0001,
        1 => 0b0000_0010,
        2 => 0b0000_0100,
        3 => 0b0000_1000,
        4 => 0b0001_0000,
        5 => 0b0010_0000,
        6 => 0b0100_0000,
        7 => 0b1000_0000,
        _ => 0,
    };

    byte ^ mask
}

fn might_mutate(byte: u8, mutation_rate: f64) -> u8 {
    if rand::random::<f64>() < mutation_rate {
        mutate(byte)
    } else {
        byte
    }
}

/// A type that can rank strategies and randomly select them by their rank
///
/// The RankSelector uses the rank selection algorithm to select a random strategy
/// with a probability that increases with their rank (the highest scoring strategies
/// are selected with highest probability).
///
#[derive(Debug)]
struct RankSelector {
    wheel: Vec<f64>,
}

impl RankSelector {
    /// Create a new RankSelector
    ///
    /// Arguments:
    ///
    /// - scores (&Vec<f64>): the scores of each strategy
    /// - selection_rate (f64): the selection rate parameter. This must be a number
    ///     between zero and one.
    ///
    pub fn new(scores: &Vec<f64>, selection_rate: f64) -> RankSelector {
        let sp = 1.0 + selection_rate;
        let n = scores.len() as f64;
        let wheel = (0..scores.len())
            .map(|idx| {
                let j = idx as f64;
                (1.0 / n) * (sp - (2.0 * sp - 2.0) * (j / (n - 1.0)))
            })
            .scan(0.0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        RankSelector { wheel }
    }

    /// Select a random strategy for reproduction.
    pub fn select(&self) -> usize {
        let x: f64 = rand::random();
        let loc = self.wheel.binary_search_by(|&y| {
            if y <= x {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        match loc {
            Ok(i) => i,
            Err(i) => i,
        }
    }
}

/// Simulation parameters.  These are static over the course of a run.
///
/// Fields:
///
/// - payoff (Payoff): the payoff structure for the game
/// - mutation_rate (f64): the probability that a given byte will be mutated
/// - selection_rate (f64): a parameter between 0.0 and 1.0 controlling the
///         strength of the selection process.
/// - logging_rate (usize): the rate at which outputs are logged.
/// - num_population (usize): the size of the population
/// - num_games (usize): the number of games in a single round
/// - num_generations (usize): the number of generations to simulate
///
#[derive(Debug, Clone, Copy)]
pub struct Simulation {
    pub payoff: Payoff,
    pub mutation_rate: f64,
    pub selection_rate: f64,
    pub logging_rate: usize,
    pub num_population: usize,
    pub num_games: usize,
    pub num_generations: usize,
}

impl Simulation {
    /// Initialize a simulation with the default parameters
    pub fn default() -> Simulation {
        Simulation {
            payoff: Payoff::default(),
            num_population: 1000,
            mutation_rate: 0.1,
            selection_rate: 2.0,
            logging_rate: 10,
            num_games: 100,
            num_generations: 1000,
        }
    }

    /// Construct a simulation state given a population of strategies
    pub fn state(&self, strategies: &Vec<GeneticStrategy>) -> SimulationState {
        SimulationState::new(self, strategies)
    }

    /// Construct a simulation state using random strategies
    pub fn random(&self) -> SimulationState {
        let strategies = (0..self.num_population)
            .map(|_| GeneticStrategy::random())
            .collect();

        SimulationState::new(self, &strategies)
    }
}

/// The simulation state
///
/// The state contains all mutable state in the simulation
///
/// Fields:
///
/// - simulation (Simulation): the static simulation parameters
/// - number (usize): the current generation number
/// - strategies (Rc<Vec<GeneticStrategy>>): a vector of the strategies
///         currently in the population
///
pub struct SimulationState {
    simulation: Simulation,
    pub number: usize,
    pub strategies: Rc<Vec<GeneticStrategy>>,
    pub scores: Rc<Vec<f64>>,
}

impl SimulationState {
    pub fn new(simulation: &Simulation, strategies: &Vec<GeneticStrategy>) -> SimulationState {
        let mut strats = strategies.clone();
        let mut scores = score_strategies(&strats, simulation);
        sort_strategies(&mut strats, &mut scores);

        SimulationState {
            simulation: simulation.clone(),
            number: 0,
            strategies: Rc::new(strats),
            scores: Rc::new(scores),
        }
    }
}

fn sort_strategies(strategies: &mut Vec<GeneticStrategy>, scores: &mut Vec<f64>) {
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());

    let sorted_strategies = indices.iter().map(|&i| strategies[i]).collect();
    let sorted_scores = indices.iter().map(|&i| scores[i]).collect();
    *strategies = sorted_strategies;
    *scores = sorted_scores;
}

fn score_strategies(strategies: &Vec<GeneticStrategy>, simulation: &Simulation) -> Vec<f64> {
    tournament(strategies, &simulation.payoff, simulation.num_games)
        .expect("failed to compute scores")
}

fn breed_strategies(
    selector: &RankSelector,
    strategies: &Vec<GeneticStrategy>,
    simulation: &Simulation,
) -> Vec<GeneticStrategy> {
    let mut children: Vec<GeneticStrategy> = Vec::new();

    for _ in 0..simulation.num_population {
        let index = selector.select();
        let parent = strategies[index];
        let child = GeneticStrategy::new(
            might_mutate(parent.prior, simulation.mutation_rate),
            parent
                .strategy
                .clone()
                .map(|x| might_mutate(x, simulation.mutation_rate)),
        );

        children.push(child);
    }

    children
}

impl Iterator for SimulationState {
    type Item = Generation;

    fn next(&mut self) -> Option<Self::Item> {
        if self.number < self.simulation.num_generations {
            let selector = RankSelector::new(&self.scores, self.simulation.selection_rate);
            let mut strats = breed_strategies(&selector, &self.strategies, &self.simulation);
            let mut scores = score_strategies(&strats, &self.simulation);
            sort_strategies(&mut strats, &mut scores);

            self.strategies = Rc::new(strats);
            self.scores = Rc::new(scores);
            self.number += 1;

            Some(Generation {
                number: self.number,
                strategies: Rc::clone(&self.strategies),
                scores: Rc::clone(&self.scores),
            })
        } else {
            None
        }
    }
}

/// A single generation of the simulation.
///
/// Fields:
///
/// * number (usize): The generation number.  Successive generations increase
///     the generation number by one.
/// * strategies (Vec<GeneticStrategy>): A vector storing all of the strategies
///     in the generation.
/// * parameters (&'a Parameters): The simulation parameters.
///
#[derive(Debug)]
pub struct Generation {
    pub number: usize,
    pub strategies: Rc<Vec<GeneticStrategy>>,
    pub scores: Rc<Vec<f64>>,
}

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

        let c = GeneticStrategy::new(0x00, s);

        assert_eq!(c.choose(&0u8), Choice::Defect);
        assert_eq!(c.choose(&1u8), Choice::Cooperate);
        assert_eq!(c.choose(&2u8), Choice::Defect);

        let mut history = 8 * 19 + 4;
        assert_eq!(c.choose(&history), Choice::Defect);
        history = 8 * 19 + 5;
        assert_eq!(c.choose(&history), Choice::Cooperate);
        history = 8 * 19 + 6;
        assert_eq!(c.choose(&history), Choice::Defect);
        history = 8 * 19 + 7;
        assert_eq!(c.choose(&history), Choice::Defect);
        history = 8 * 20;
        assert_eq!(c.choose(&history), Choice::Cooperate);
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
        let p1 = GeneticStrategy::new(0x00, s1);
        let p2 = GeneticStrategy::new(0x00, s2);

        let payoff = Payoff::default();
        let result = game(&p1, &mut p1.history(), &p2, &mut p2.history(), &payoff);

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

    #[test]
    fn test_generation_run() {
        let mut sim = Simulation::default();
        sim.num_generations = 3;
        sim.num_population = 20;
        sim.selection_rate = 1.0;

        for gen in sim.random() {
            assert!(gen.scores.windows(2).all(|pair| pair[0] >= pair[1]));
        }
    }
}
