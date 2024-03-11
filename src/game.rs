//! Functions and Traits for playing Prisoner's Dilemma

/// Enum representing the choices in a prisoners dilemma
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Choice {
    Cooperate,
    Defect,
}

pub trait Strategy: Copy {
    type History;

    /// Make a choice in the game
    fn choose(&self, history: &Self::History) -> Choice;

    /// Update the strategy with the opponent's prior choice
    fn update(history: &mut Self::History, player_choice: &Choice, opponent_choice: &Choice) -> ();

    /// Create a new history to prepare for a new match
    fn history(&self) -> Self::History;
}

/// Prisoners dilemma payoffs
///
/// # Fields
///
/// - reward (f64): The cooperation payoff
/// - temptation (f64): The temptation payoff
/// - sucker (f64): The sucker payoff
/// - punishment (f64): The punishment payoff
///
#[derive(Debug, Clone, Copy)]
pub struct Payoff {
    pub reward: f64,
    pub temptation: f64,
    pub sucker: f64,
    pub punishment: f64,
}

impl Payoff {
    /// The classic prisoners dilemma payoff structure
    pub fn default() -> Payoff {
        Payoff {
            reward: 3.0,
            temptation: 5.0,
            sucker: 0.0,
            punishment: 1.0,
        }
    }
}

/// Play a single prisoners dilemma game between two strategies
///
/// This function updates both strategies with each players respective choices.
///
/// Arguments:
///
/// * first (&mut F: Strategy): The first strategy
/// * second (&mut S: Strategy): The second strategy
/// * payoff (&Payoff): The payoff structure
///
/// Returns:
///
/// (f64, 64): The payoff to the first and second players.
///
pub fn game<F: Strategy, S: Strategy>(
    first: &F,
    first_history: &mut F::History,
    second: &S,
    second_history: &mut S::History,
    payoff: &Payoff,
) -> (f64, f64) {
    let first_choice = first.choose(first_history);
    let second_choice = second.choose(second_history);

    F::update(first_history, &first_choice, &second_choice);
    S::update(second_history, &second_choice, &first_choice);

    match (first_choice, second_choice) {
        (Choice::Cooperate, Choice::Cooperate) => (payoff.reward, payoff.reward),
        (Choice::Cooperate, Choice::Defect) => (payoff.sucker, payoff.temptation),
        (Choice::Defect, Choice::Cooperate) => (payoff.temptation, payoff.sucker),
        (Choice::Defect, Choice::Defect) => (payoff.punishment, payoff.punishment),
    }
}

/// Play a round of Prisoners Dilemma games
///
/// Arguments:
///
/// * first (&mut F: Strategy): The first strategy
/// * second (&mut S: Strategy): The second strategy
/// * payoff (&Payoff): The payoff structure
/// * num_games (usize): The number of games in the round
///
/// Returns:
///
/// (f64, f64): The cumulative payoff to the first and second players.
pub fn round<F: Strategy, S: Strategy>(
    first: &F,
    second: &S,
    payoff: &Payoff,
    num_games: usize,
) -> (f64, f64) {
    let mut first_points = 0.0;
    let mut second_points = 0.0;
    let mut first_history = first.history();
    let mut second_history = second.history();

    for _ in 0..num_games {
        let points = game(
            first,
            &mut first_history,
            second,
            &mut second_history,
            payoff,
        );
        first_points += points.0;
        second_points += points.1;
    }

    (first_points, second_points)
}

/// Play a round-robin tournament between strategies
///
/// Arguments:
///
/// * strategies (&mut Vec<S: Strategy>): The strategies
/// * payoff (&Payoff): The payoff structure
/// * num_games (usize): The number of games in each round
///
/// Returns:
///
/// Vec<f64>: The cumulative payoff for each player in the tournament
pub fn tournament<'a, S: Strategy>(
    strategies: &'a Vec<S>,
    payoff: &'a Payoff,
    num_games: usize,
) -> Result<Vec<f64>, &'static str> {
    let n = strategies.len();
    if n < 2 {
        return Err("must have at least two strategies");
    }

    let mut scores = vec![0.0; n];
    for i in 0..n {
        for j in i + 1..n {
            let si = strategies[i];
            let sj = strategies[j];
            let (score_i, score_j) = round(&si, &sj, payoff, num_games);

            scores[i] += score_i;
            scores[j] += score_j;
        }
    }

    Ok(scores)
}

// Some classic strategies ...

#[derive(Debug, Clone, Copy)]
pub struct TitForTat();

impl Strategy for TitForTat {
    type History = Choice;

    fn choose(&self, history: &Self::History) -> Choice {
        *history
    }

    fn update(history: &mut Self::History, player_choice: &Choice, opponent_choice: &Choice) -> () {
        *history = *opponent_choice;
    }

    fn history(&self) -> Self::History {
        Choice::Cooperate
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AllCooperate;

impl Strategy for AllCooperate {
    type History = ();

    fn choose(&self, _history: &Self::History) -> Choice {
        Choice::Cooperate
    }

    fn update(
        _history: &mut Self::History,
        _player_choice: &Choice,
        _opponent_choice: &Choice,
    ) -> () {
    }

    fn history(&self) -> Self::History {}
}

#[derive(Debug, Clone, Copy)]
pub struct AllDefect;

impl Strategy for AllDefect {
    type History = ();

    fn choose(&self, _history: &Self::History) -> Choice {
        Choice::Defect
    }

    fn update(
        _history: &mut Self::History,
        _player_choice: &Choice,
        _opponent_choice: &Choice,
    ) -> () {
    }

    fn history(&self) -> Self::History {}
}

#[derive(Debug, Clone, Copy)]
pub struct Pavlov {
    prior: Choice,
}

impl Strategy for Pavlov {
    type History = Choice;

    fn choose(&self, history: &Self::History) -> Choice {
        *history
    }

    fn update(history: &mut Self::History, player_choice: &Choice, opponent_choice: &Choice) -> () {
        *history = match (player_choice, opponent_choice) {
            (Choice::Cooperate, Choice::Defect) => Choice::Defect,
            (Choice::Defect, Choice::Defect) => Choice::Cooperate,
            _ => *history,
        };
    }

    fn history(&self) -> Self::History {
        self.prior
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooperate() {
        let mut p1 = TitForTat {};
        let mut p2 = TitForTat {};
        let payoff = Payoff::default();

        let (s1, s2) = game(&p1, &mut p1.history(), &p2, &mut p2.history(), &payoff);
        assert_eq!(s1, payoff.reward);
        assert_eq!(s2, payoff.reward);
    }

    // #[test]
    // fn test_defect() {
    //     let mut p1 = AllCooperate {};
    //     let mut p2 = AllDefect {};
    //     let payoff = Payoff::default();

    //     let (s1, s2) = game(&mut p1, &mut p2, &payoff);
    //     assert_eq!(s1, payoff.sucker);
    //     assert_eq!(s2, payoff.temptation);
    // }

    // #[test]
    // fn test_both_defect() {
    //     let mut p1 = AllDefect {};
    //     let mut p2 = AllDefect {};
    //     let payoff = Payoff::default();

    //     let (s1, s2) = game(&mut p1, &mut p2, &payoff);
    //     assert_eq!(s1, payoff.punishment);
    //     assert_eq!(s2, payoff.punishment);
    // }

    // #[test]
    // fn test_round() {
    //     let mut p1 = AllDefect {};
    //     let mut p2 = AllCooperate {};
    //     let payoff = Payoff::default();

    //     let (s1, s2) = round(&mut p1, &mut p2, &payoff, 100);
    //     assert_eq!(s1, 100.0 * payoff.temptation);
    //     assert_eq!(s2, 100.0 * payoff.sucker);
    // }

    // #[test]
    // fn test_tournament() -> Result<(), &'static str> {
    //     let mut strats = vec![TitForTat::new(), TitForTat::new(), TitForTat::new()];
    //     let payoff = Payoff::default();

    //     let results = tournament(&mut strats, &payoff, 100)?;
    //     assert_eq!(results[0], 2.0 * 100.0 * payoff.reward);
    //     assert_eq!(results[1], 2.0 * 100.0 * payoff.reward);
    //     assert_eq!(results[2], 2.0 * 100.0 * payoff.reward);
    //     Ok(())
    // }
}
