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
Stats = namedtuple('TeamStats', 'team wins losses ties points_for points_against')


class TeamStats(Stats):
    def __add__(self, other):
        if not self.team == other.team:
            raise ValueError('cannot add {} to {}'.format(self.team, other.team))
        return TeamStats(self.team, *tuple(map(operator.add, self[1:], other[1:])))


def win_loss_tie(pf, pa):
    if pf == pa:
        return (0, 0, 1)
    return ((1, 0, 0) if pf > pa else
            (0, 1, 0))


def update_stats(table, team, pf, pa):
    try:
        stats = table[team]
    except KeyError:
        stats = blank_team(team)
    return stats + TeamStats(team, *win_loss_tie(pf, pa), pf, pa)


def append_game(stat_table, game):
    stat_table[game.away_team] = update_stats(stat_table, game.away_team, game.away_points, game.home_points)
    stat_table[game.home_team] = update_stats(stat_table, game.home_team, game.home_points, game.away_points)
    return stat_table


def blank_team(name):
    return TeamStats(name, *list(repeat(0, 5)))


def dict2game(src):
    src.update({
        k: int(v)
        for (k, v) in src.items()
        if not k.endswith('team')
    })
    src['key'] = src['year'] * 100 + src['week']
    return Game(**src)


def read_records(year, records):
    return takewhile(lambda row: row['year'] == year, csv.DictReader(records))


def get_team_stats(records):
    read_season_records = curry(read_records)('2018')
    return reduce(append_game, compose(map(dict2game), read_season_records)(records), {})


def calc(stats):
    delta = stats.points_for - stats.points_against
    game_count = sum([stats.wins, stats.losses, stats.ties])
    return {
        'team': stats.team,
        'w': stats.wins, 'l': stats.losses, 't': stats.ties,
        'w%': round(stats.wins/game_count, 3),
        'pf': stats.points_for, 'pf/g': round(stats.points_for/game_count, 1), 
        'pa': stats.points_against, 'pa/g': round(stats.points_against/game_count, 1),
        'pd': delta, 'mov': round(delta/game_count, 1),
    }


def run(outfile, opts):
    with open(opts['<records-file>'], 'r') as records:
        table = get_team_stats(records)
    outcsv = csv.DictWriter(outfile, fieldnames=['team', 'w', 'l', 't', 'w%', 'pf', 'pf/g', 'pa', 'pa/g', 'pd', 'mov'])
    outcsv.writeheader()
    for team_stats in table.values():
        outcsv.writerow(calc(team_stats))


def main():
    doc = docopt(__doc__)
    write_fh = open(doc['--output'], 'w') if doc['--output'] else sys.stdout
    try:
        run(write_fh, doc)
    finally:
        write_fh.close()


if __name__ == '__main__':
    main()