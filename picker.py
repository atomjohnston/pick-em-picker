#!/usr/bin/env python
"""Pro Football Pick 'Em Picker


Usage:
  pick-em get-results <date-range> [--output CSV]
  pick-em simple-ranking <records-file> <pick-week> [<date-range>]
  pick-em make-picks <records-file> <pick-week>
                     [<date-range>] [--output CSV] [--spread HTML]

Arguments:
  pick-week     year:week i.e. 2017:07
  date-range    date range like 2017:01-2017:07 (inclusive)
  records-file  CSV containing game results
  CSV           file to write CSV output (default: STDOUT)
  HTML          HTML file containing OddsShark NFL spread page

Options:
  -o --output CSV    output to CSV file, instead of default STDOUT
  -s --spread HTML   HTML file of OddsShark odds page
  -h --help          display this help


"""

import csv
import operator
import statistics
import sys

import requests

from collections import namedtuple
from concurrent import futures
from functools import reduce
from itertools import takewhile, repeat

from docopt import docopt
from lxml import html
from toolz import compose, curry, flip, interleave, pipe
from toolz.curried import map


Game = namedtuple('Game', 'key year week away_team away_points home_team home_points')
Record = namedtuple('Record', 'team opponents games wins losses ties points_for points_against')
Stats = namedtuple('Stats', 'team win_per points_per against_per point_delta margin_of_victory strength_of_schedule simple_ranking pythagorean')
Summary = namedtuple('Summary', 'team record stats')
Pick = namedtuple('Pick', 'delta winner rank')
Picks = namedtuple('Picks', 'point_spread simple_ranking pythagorean margin_of_victory wins')

NICKNAMES = frozenset([
    '49ers', 'bears', 'bengals', 'bills', 'broncos', 'browns', 'buccaneers',
    'cardinals', 'chargers', 'chiefs', 'colts', 'cowboys', 'dolphins',
    'eagles', 'falcons', 'giants', 'jaguars', 'jets', 'lions', 'packers',
    'panthers', 'patriots', 'raiders', 'rams', 'ravens', 'redskins', 'saints',
    'seahawks', 'steelers', 'texans', 'titans', 'vikings'
])


round2 = flip(round, 2)  # pylint: disable=E1120
round3 = flip(round, 3)  # pylint: disable=E1120

class TeamRecord(Record):
    def __add__(self, other):
        if not self.team == other.team:
            raise ValueError('cannot add {} to {}'.format(self.team, other.team))
        return TeamRecord(self.team, self.opponents + (other.opponents,),
            *tuple(map(operator.add, self[2:], other[2:])))


def win_loss_tie(pf, pa):
    if pf == pa:
        return (0, 0, 1)
    return ((1, 0, 0) if pf > pa else
            (0, 1, 0))


def add_stats(stats, team, opponent, pf, pa):
    return stats + TeamRecord(team, opponent, 1, *win_loss_tie(pf, pa), pf, pa)


def append_game(table, game):
    table[game.away_team] = add_stats(table[game.away_team], game.away_team, game.home_team, game.away_points, game.home_points)
    table[game.home_team] = add_stats(table[game.home_team], game.home_team, game.away_team, game.home_points, game.away_points)
    return table


def dict2game(src):
    src.update({k: int(v) for (k, v) in src.items()
                          if not k.endswith('team')})
    src['key'] = src['year'] * 100 + src['week']
    return Game(**src)


def calc(record):
    delta = record.points_for - record.points_against
    mov = round2(delta/record.games)
    return Stats(
        record.team,
        round3(record.wins/record.games),
        round2(record.points_for/record.games),
        round2(record.points_against/record.games),
        delta,
        mov,
        0,
        mov,
        round3((record.points_for**2.37)/((record.points_for**2.37)+(record.points_against**2.37)))
    )


def simplify(summary):
    calculate = compose(round2, statistics.mean)

    def calc_sos(team_smry, get_opponent_srs):
        return calculate([get_opponent_srs(name) for name in team_smry.record.opponents])

    def adjust(summaries, t_sum):
        sos = calc_sos(t_sum, lambda s: summaries[s].stats.simple_ranking)
        return t_sum._replace(
            stats=t_sum.stats._replace(
                strength_of_schedule=sos,
                simple_ranking=round2(t_sum.stats.margin_of_victory + sos)))
    
    calc_drift = compose(round2, curry(max))
    def calculate_all(previous):
        adjustments = {name: adjust(previous, smry) for (name, smry) in previous.items()}
        drift = calc_drift([
            abs(current.stats.strength_of_schedule - previous.stats.strength_of_schedule)
            for (previous, current) in zip(previous.values(), adjustments.values())])
        return (adjustments if drift <= 0.01 else
                calculate_all(adjustments))

    return calculate_all(summary)


def write_summary(file_, team_summary):
    outcsv = csv.DictWriter(file_, fieldnames=['team', 'w', 'l', 't', 'w%', 'pf', 'pf/g', 'pa', 'pa/g', 'pd', 'mov', 'sos', 'srs', 'pyth'])
    outcsv.writeheader()
    for summary in sorted(team_summary.values(), key=lambda x: x.stats.win_per, reverse=True):
        outcsv.writerow({
            'team': summary.record.team,
            'w': summary.record.wins,
            'l': summary.record.losses,
            't': summary.record.ties,
            'w%': summary.stats.win_per,
            'pf': summary.record.points_for,
            'pf/g': summary.stats.points_per, 
            'pa': summary.record.points_against,
            'pa/g': summary.stats.against_per,
            'pd': summary.stats.point_delta,
            'mov': summary.stats.margin_of_victory,
            'sos': summary.stats.strength_of_schedule,
            'srs': summary.stats.simple_ranking,
            'pyth': summary.stats.pythagorean,
        })


def read_records(year, records):
    return takewhile(lambda row: row['year'] == year, csv.DictReader(records))


def init_teams():
    return {team: TeamRecord(team, (), *list(repeat(0, 6))) for team in NICKNAMES}


def get_team_stats(season, records):
    read_season_records = curry(read_records)(season)
    games = compose(map(dict2game), read_season_records)(records)
    records = reduce(append_game, games, init_teams())
    return {r.team: Summary(r.team, r, calc(r)) for r in records.values()}


def predict_winners(team_summary, games):
    # pylint: disable=E1120,E1102

    def stats_for(team):
        return team_summary[team].stats

    @curry
    def predict(game, sel, round_to=2):
        delta = sel(stats_for(game.home_team)) - sel(stats_for(game.away_team))
        return Pick(round(delta, round_to), (game.home_team if delta >= 0 else game.away_team), 0)

    def predict_all(game):
        pick = predict(game)
        return Picks(
            point_spread=pick(lambda x: 0),
            simple_ranking=pick(lambda x: x.simple_ranking),
            pythagorean=pick(lambda x: x.pythagorean, 3),
            margin_of_victory=pick(lambda x: x.margin_of_victory),
            wins=pick(lambda x: x.win_per, 3),
        )

    def rank(key, predictions):
        # x._replace(simple_ranking=x.simple_ranking._replace(rank=n))
        return [key(x)._replace(rank=n) for (n, x) in enumerate(sorted(predictions, key=compose(abs, lambda x: x.delta, key)), start=1)]

    rank = curry(rank)
    return pipe([predict_all(game) for game in games],
                    rank(lambda x: x.simple_ranking),
                    rank(lambda x: x.point_spread),
                    rank(lambda x: x.pythagorean),
                    rank(lambda x: x.margin_of_victory),
                    rank(lambda x: x.wins),
    )


def score_scrape(yr, wk_from, wk_to=None):
    URL = 'https://www.pro-football-reference.com/years/{}/week_{}.htm'

    def select_games(week_xml):
        return week_xml.xpath(
            '//div[starts-with(@class,"game_summary")]/table[@class="teams"]/tbody')

    def parse_game(game_xml):
        teams = [x.text.split()[-1].lower() for x in game_xml.xpath('tr/td/a[starts-with(@href, "/teams/")]')]
        scores = [x.text if x.text else -1 for x in game_xml.xpath('tr/td[@class="right"]')][0:2]
        return zip(teams, [int(scores[0]), int(scores[1])])

    def scrape_week(yr, week):
        parser = compose(map(parse_game), select_games)
        game_info = parser(html.fromstring(requests.get(URL.format(yr, week)).content))
        for (away, home) in game_info:
            yield Game(yr * 100 + week, yr, week, *away, *home)

    def scrape_weeks(yr, wk_from, wk_to):
        step = 1 if wk_from < wk_to else -1
        ex = futures.ThreadPoolExecutor(max_workers=4)
        return sorted(ex.map(scrape_week, repeat(yr), [wk for wk in range(wk_from, wk_to + step, step)]), reverse=True)

    #return scrape_weeks(yr, wk_from, wk_to if wk_to else wk_from)
    return scrape_week(yr, wk_from)


def run(outfile, opts):
    with open(opts['<records-file>'], 'r') as records:
        team_summary = pipe(records, curry(get_team_stats)('2018'),
                                     simplify)
    predictions = predict_winners(team_summary, score_scrape(2018, 9))
    print(predictions)
    write_summary(outfile, team_summary)


def main():
    doc = docopt(__doc__)
    write_fh = open(doc['--output'], 'w') if doc['--output'] else sys.stdout
    try:
        run(write_fh, doc)
    finally:
        write_fh.close()


if __name__ == '__main__':
    main()