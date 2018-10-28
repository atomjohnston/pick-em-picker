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
import sys

from collections import namedtuple
from docopt import docopt
from functools import reduce
from itertools import takewhile, repeat
from toolz import compose, curry
from toolz.curried import map

Game = namedtuple('Game', 'key year week away_team away_points home_team home_points')
Record = namedtuple('Record', 'team games wins losses ties points_for points_against')
Stats = namedtuple('Stats', 'team win_per points_per against_per point_delta margin_of_victory simple_ranking')
Summary = namedtuple('Summary', 'team record stats')

NICKNAMES = frozenset([
    '49ers', 'bears', 'bengals', 'bills', 'broncos', 'browns', 'buccaneers',
    'cardinals', 'chargers', 'chiefs', 'colts', 'cowboys', 'dolphins',
    'eagles', 'falcons', 'giants', 'jaguars', 'jets', 'lions', 'packers',
    'panthers', 'patriots', 'raiders', 'rams', 'ravens', 'redskins', 'saints',
    'seahawks', 'steelers', 'texans', 'titans', 'vikings'
])


class TeamRecord(Record):
    def __add__(self, other):
        if not self.team == other.team:
            raise ValueError('cannot add {} to {}'.format(self.team, other.team))
        return TeamRecord(self.team, *tuple(map(operator.add, self[1:], other[1:])))


def win_loss_tie(pf, pa):
    if pf == pa:
        return (0, 0, 1)
    return ((1, 0, 0) if pf > pa else
            (0, 1, 0))


def add_stats(stats, team, pf, pa):
    return stats + TeamRecord(team, 1, *win_loss_tie(pf, pa), pf, pa)


def append_game(table, game):
    table[game.away_team] = add_stats(table[game.away_team], game.away_team, game.away_points, game.home_points)
    table[game.home_team] = add_stats(table[game.home_team], game.home_team, game.home_points, game.away_points)
    return table


def dict2game(src):
    src.update({k: int(v) for (k, v) in src.items()
                          if not k.endswith('team')})
    src['key'] = src['year'] * 100 + src['week']
    return Game(**src)


def calc(record):
    delta = record.points_for - record.points_against
    return Stats(
        record.team,
        round(record.wins/record.games, 3),
        round(record.points_for/record.games, 1),
        round(record.points_against/record.games, 1),
        delta,
        round(delta/record.games, 1),
        0
    )


def format(record, stats):
    return {
        'team': record.team,
        'w': record.wins,
        'l': record.losses,
        't': record.ties,
        'w%': stats.win_per,
        'pf': record.points_for,
        'pf/g': stats.points_per, 
        'pa': record.points_against,
        'pa/g': stats.against_per,
        'pd': stats.point_delta,
        'mov': stats.margin_of_victory,
        'srs': stats.simple_ranking
    }


def read_records(year, records):
    return takewhile(lambda row: row['year'] == year, csv.DictReader(records))


def init_teams():
    return {team: TeamRecord(team, *list(repeat(0, 6))) for team in NICKNAMES}


def get_team_stats(season, records):
    read_season_records = curry(read_records)(season)
    games = compose(map(dict2game), read_season_records)(records)
    records = reduce(append_game, games, init_teams())
    return {r.team: Summary(r.team, r, calc(r)) for r in records.values()}


def run(outfile, opts):
    with open(opts['<records-file>'], 'r') as records:
        table = get_team_stats('2018', records)
    outcsv = csv.DictWriter(outfile, fieldnames=['team', 'w', 'l', 't', 'w%', 'pf', 'pf/g', 'pa', 'pa/g', 'pd', 'mov', 'srs'])
    outcsv.writeheader()
    for summary in sorted(table.values(), key=lambda x: x.stats.margin_of_victory, reverse=True):
        outcsv.writerow(format(summary.record, summary.stats))


def main():
    doc = docopt(__doc__)
    write_fh = open(doc['--output'], 'w') if doc['--output'] else sys.stdout
    try:
        run(write_fh, doc)
    finally:
        write_fh.close()


if __name__ == '__main__':
    main()