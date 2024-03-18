use crate::game::{Choice, Strategy};
use crate::genetic::GeneticStrategy;

use std::f64::INFINITY;
use std::ops::RangeInclusive;

pub fn mean(data: &Vec<f64>) -> f64 {
    let sum: f64 = data.iter().sum();
    let count = data.len() as f64;
    sum / count
}

pub fn variance(data: &Vec<f64>, mean: f64) -> f64 {
    let sum_of_squares: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
    let count = data.len() as f64;
    sum_of_squares / count
}

pub fn stdev(data: &Vec<f64>, mean: f64) -> f64 {
    variance(data, mean).sqrt()
}

pub fn max(data: &Vec<f64>) -> f64 {
    data.iter()
        .fold(-INFINITY, |acc, &x| if x > acc { x } else { acc })
}

pub fn min(data: &Vec<f64>) -> f64 {
    data.iter()
        .fold(INFINITY, |acc, &x| if x < acc { x } else { acc })
}

/// Iterate over all histories
fn histories() -> RangeInclusive<u8> {
    0u8..=255u8
}

/// Fraction of defections averaged over all histories
///
/// Arguments:
///
/// * strategy (&GeneticStrategy): the strategy to examine
///
/// Returns:
///
/// f64: the fraction of defections
pub fn fraction_defect(strategy: &GeneticStrategy) -> f64 {
    let num_defect: f64 = histories()
        .map(|h| match strategy.choose(&h) {
            Choice::Cooperate => 0.0,
            Choice::Defect => 1.0,
        })
        .sum();
    num_defect / 256.0
}

static MASK_OPPONENT_DEFECT: u8 = 0x02;

/// Fraction of defections after an opponent defection
///
/// A retaliation is a defection choice which immediately follows an opponents
/// defection.  This metric is averaged over all histories where the opponent's
/// prior move was defect.
///
/// Arguments:
///
/// * strategy (&GeneticStrategy): the strategy to examine
///
/// Returns:
///
/// f64: the fraction of retaliations
pub fn fraction_retaliate(strategy: &GeneticStrategy) -> f64 {
    let num_retaliate: f64 = histories()
        .filter(|&h| match h & MASK_OPPONENT_DEFECT {
            0x00 => false,
            _ => true,
        })
        .map(|h| match strategy.choose(&h) {
            Choice::Cooperate => 0.0,
            Choice::Defect => 1.0,
        })
        .sum();
    num_retaliate / 128.0
}

/// Fraction of defections after an opponent cooperation
///
/// A betrayal is a defection choice which immediately follows an opponents
/// cooperation.  This metric is averaged over all histories where the
/// opponent's prior move was cooperate.
///
/// Arguments:
///
/// * strategy (&GeneticStrategy): the strategy to examine
///
/// Returns:
///
/// f64: the fraction of betrayals
pub fn fraction_betray(strategy: &GeneticStrategy) -> f64 {
    let num_betray: f64 = histories()
        .filter(|&h| match h & MASK_OPPONENT_DEFECT {
            0x00 => true,
            _ => false,
        })
        .map(|h| match strategy.choose(&h) {
            Choice::Cooperate => 0.0,
            Choice::Defect => 1.0,
        })
        .sum();
    num_betray / 128.0
}
