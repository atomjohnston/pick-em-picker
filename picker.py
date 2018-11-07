#!/usr/bin/env python
"""Pro Football Pick 'Em Picker


Usage:
  pick-em get-results <start> [<end>] [--output=CSV ]
  pick-em standings <records-file> [<start> <end>] [--output=CSV ]
  pick-em make-picks <records-file> <pick-week>
                     [<start> <end>] [--spread=HTML] [--output=CSV ]

Options:
  -o CSV  --output=CSV    output to CSV file, instead of default STDOUT
  -s HTML --spread=HTML   HTML file of OddsShark odds page
  -h --help                display this help


"""

import csv
import datetime
import json
import operator
import statistics
import sys

import requests

from collections import namedtuple
from concurrent import futures
from functools import reduce
from itertools import islice, repeat, takewhile

from docopt import docopt
from lxml import html
from toolz import compose, concat, curry, flip, interleave, pipe, valmap
from toolz.curried import filter, first, map, partition


Week = namedtuple('Week', 'year week')
Scored = namedtuple('Scored', 'team points')
Game = namedtuple('Game', 'away home year week ')
Record = namedtuple('Record', 'team opponents games wins losses ties points_for points_against')
Stats = namedtuple('Stats', ('team win_rate points_per against_per point_delta margin_of_victory '
                             'strength_of_schedule simple_ranking pythagorean'))
Summary = namedtuple('Summary', 'team record stats')
Pick = namedtuple('Pick', 'winner rank delta')
Picks = namedtuple('Picks', ('id game point_spread simple_ranking simple_ranking_1 '
                             'pythagorean margin_of_victory wins'))

PYTH_EX = 2.37
TEAM_MAP = {
    'ari': 'cardinals', 'atl': 'falcons',  'bal': 'ravens',
    'buf': 'bills',     'car': 'panthers', 'chi': 'bears',
    'cin': 'bengals',   'cle': 'browns',   'dal': 'cowboys',
    'den': 'broncos',   'det': 'lions',    'gb':  'packers',
    'hou': 'texans',    'ind': 'colts',    'jac': 'jaguars',
    'kc':  'chiefs',    'lac': 'chargers', 'lar': 'rams',
    'min': 'vikings',   'ne':  'patriots', 'no':  'saints',
    'nyg': 'giants',    'nyj': 'jets',     'oak': 'raiders',
    'phi': 'eagles',    'pit': 'steelers', 'sea': 'seahawks',
    'sf':  '49ers',     'ten': 'titans',   'was': 'redskins',
    'mia': 'dolphins',  'tb':  'buccaneers'
}


round2 = flip(round, 2)  # pylint: disable=E1120
round3 = flip(round, 3)  # pylint: disable=E1120


class TeamRecord(Record):
    def __add__(self, other):
        if not self.team == other.team:
            raise ValueError(f'cannot add {self.team} to {other.team}')
        return TeamRecord(self.team, self.opponents + (other.opponents,),
                          *tuple(map(operator.add, self[2:], other[2:])))


def win_loss_tie(pf, pa):
    if pf == pa:
        return (0, 0, 1)
    return ((1, 0, 0) if pf > pa else
            (0, 1, 0))


def add_stats(stats, team, opponent):
    return stats + TeamRecord(
        team.team, opponent.team, 1, *win_loss_tie(team.points, opponent.points),
        team.points, opponent.points)


def append_game(table, game):
    table[game.away.team] = add_stats(table[game.away.team], game.away, game.home)
    table[game.home.team] = add_stats(table[game.home.team], game.home, game.away)
    return table


def dict2game(src):
    src.update({k: int(v) for (k, v) in src.items()
                if not k.endswith('team')})
    away = Scored(src.pop('away_team'), src.pop('away_points'))
    home = Scored(src.pop('home_team'), src.pop('home_points'))
    return Game(away, home, **src)


def calc(record):
    delta = record.points_for - record.points_against
    mov = round2(delta / record.games)
    return Stats(
        record.team,
        round3(record.wins / record.games),
        round2(record.points_for / record.games),
        round2(record.points_against / record.games), delta, mov, 0, mov,
        round3(calc_pythagorean(record.points_for, record.points_against))
    )


def calc_pythagorean(pf, pa):
    return (pf ** PYTH_EX) / ((pf ** PYTH_EX) + (pa ** PYTH_EX))


def simplify(summary):
    def update_sum_stats(smry, sos, srs):
        return smry._replace(
            stats=smry.stats._replace(strength_of_schedule=sos, simple_ranking=srs))

    def adjust(summaries, t_sum):
        sos = statistics.mean([summaries[name].stats.simple_ranking for name in t_sum.record.opponents])
        return update_sum_stats(t_sum, sos, t_sum.stats.margin_of_victory + sos)

    def correct(summaries):
        mean = statistics.mean([x.stats.simple_ranking for x in summaries.values()])
        return valmap(lambda s: update_sum_stats(s, s.stats.strength_of_schedule - mean, s.stats.simple_ranking - mean), summaries)

    def round_all(summaries):
        return valmap(lambda s: update_sum_stats(s, round2(s.stats.strength_of_schedule), round2(s.stats.simple_ranking)), summaries)

    def calculate_all(previous):
        adjustments = correct({name: adjust(previous, smry) for (name, smry) in previous.items()})
        drift = max([
            abs(current.stats.strength_of_schedule - previous.stats.strength_of_schedule)
            for (previous, current) in zip(previous.values(), adjustments.values())])
        return (round_all(adjustments) if drift <= 0.001 else
                calculate_all(adjustments))

    return calculate_all(summary)


def write_summary(file_, team_summary):
    outcsv = csv.writer(file_)
    outcsv.writerow(['team', 'g', 'w', 'l', 't', 'pf', 'pa', 'w%',
                     'pf/g', 'pa/g', 'pd', 'mov', 'sos', 'srs', 'pyth'])
    for summary in sorted(team_summary.values(), key=lambda x: x.stats.win_rate, reverse=True):
        record = summary.record._asdict()
        stats = summary.stats._asdict()
        del record['opponents']
        del stats['team']
        outcsv.writerow(concat([record.values(), stats.values()]))


def write_game_results(file_, results):
    outcsv = csv.writer(file_)
    outcsv.writerow(['year', 'week', 'away_team', 'away_points', 'home_team', 'home_points'])
    for game in results:
        outcsv.writerow([
            game.year, f'{game.week:02}',
            game.away.team, game.away.points,
            game.home.team, game.home.points
        ])


def write_predictions(file_, predictions):
    outcsv = csv.writer(file_)
    outcsv.writerow([
        '#', 'away', 'home',
        'spread', 'spread_r', 'spread_d',
        'srs', 'srs_r', 'srs_d',
        'srs1', 'srs1_r', 'srs1_d',
        'pyth', 'pyth_r', 'pyth_d',
        'mov', 'mov_r', 'mov_d',
        'w%', 'w%_r', 'w%_d',
    ])
    for picks in sorted(predictions, key=lambda x: x.simple_ranking.rank, reverse=True):
        outcsv.writerow(concat([
            [picks.id],
            [picks.game.away.team, picks.game.home.team],
            picks.point_spread,
            picks.simple_ranking,
            picks.simple_ranking_1,
            picks.pythagorean,
            picks.margin_of_victory,
            picks.wins,
        ]))


def read_records(years, records):
    return takewhile(lambda row: int(row['year']) in years, csv.DictReader(records))


def init_teams():
    return {team: TeamRecord(team, (), *list(repeat(0, 6))) for team in TEAM_MAP.values()}


def get_team_stats(season, records):
    read_season_records = curry(read_records)(season)
    games = compose(map(dict2game), read_season_records)(records)
    records = reduce(append_game, games, init_teams())
    return {r.team: Summary(r.team, r, calc(r)) for r in records.values()}


def predict_winners(team_summary, games, spreads):
    # pylint: disable=E1120,E1102

    def stats_for(team):
        return team_summary[team.team].stats

    @curry
    def predict(away, home, n_away, n_home, round_f=round2):
        delta = n_home - n_away
        return Pick((home if delta >= 0 else away), 0, round_f(delta))

    def predict_all(id, game, a_stats, h_stats):
        pick = predict(a_stats.team, h_stats.team)
        point_spread = spreads[(a_stats.team, h_stats.team)]
        return dict(
            id=id,
            game=game,
            point_spread=pick(point_spread.away.points, point_spread.home.points),
            simple_ranking=pick(a_stats.simple_ranking, h_stats.simple_ranking),
            simple_ranking_1=pick(a_stats.simple_ranking, h_stats.simple_ranking + 2),
            pythagorean=pick(a_stats.pythagorean, h_stats.pythagorean, round3),
            margin_of_victory=pick(a_stats.margin_of_victory, h_stats.margin_of_victory),
            wins=pick(a_stats.win_rate, h_stats.win_rate, round3),
        )

    @curry
    def rank(key, predictions):
        for (n, pick) in enumerate(sorted(predictions, key=lambda x: abs(x[key].delta)), start=1):
            pick[key] = pick[key]._replace(rank=n)
        return predictions

    ranked = pipe(
        [predict_all(n + 1, game, stats_for(game.away), stats_for(game.home))
         for (n, game) in enumerate(games)],
        rank('simple_ranking'),
        rank('simple_ranking_1'),
        rank('point_spread'),
        rank('pythagorean'),
        rank('margin_of_victory'),
        rank('wins'),
    )
    return [Picks(**picks) for picks in ranked]


get_html_from_url = compose(html.fromstring, lambda x: x.content, requests.get)


def score_scrape(yr, wk_from, wk_to=None):
    def select_games(week_xml):
        return week_xml.xpath(
            '//div[starts-with(@class,"game_summary")]/table[@class="teams"]/tbody')

    def parse_game(game_xml):
        teams = [x.text.split()[-1].lower()
                 for x in game_xml.xpath('tr/td/a[starts-with(@href, "/teams/")]')]
        scores = [x.text if x.text else -1 for x in game_xml.xpath('tr/td[@class="right"]')][0:2]
        return zip(teams, [int(scores[0]), int(scores[1])])

    def scrape_week(yr, week):
        parse = compose(map(parse_game), select_games, get_html_from_url)
        game_info = parse(f'https://www.pro-football-reference.com/years/{yr}/week_{week}.htm')
        return [Game(Scored(*away), Scored(*home), int(yr), int(week))
                for (away, home) in game_info]

    def scrape_weeks(yr, wk_from, wk_to):
        start, end = int(wk_from), int(wk_to)
        step = 1 if start < end else -1
        ex = futures.ThreadPoolExecutor(max_workers=4)
        return sorted(ex.map(scrape_week, repeat(yr), [wk for wk in range(start, end + step, step)]), reverse=True)

    return scrape_weeks(yr, wk_from, wk_to if wk_to else wk_from)


def spread_scrape(yr, wk, odds=None):
    def get_odds_html(odds_file):
        if not odds_file:
            return get_html_from_url('https://www.oddsshark.com/nfl/odds')

        with open(odds_file, 'r') as f:
            return html.fromstring(f.read())

    def parse_spreads(g):
        return [json.loads(x.attrib['data-op-info'])['fullgame']
                for x in g.xpath('div/div[starts-with(@class, "op-item op-spread")]')]

    def convert_spreads(diffs):
        return [0 if n == 'Ev' else float(n) for n in diffs if not n == '']

    lookup_nickname = map(lambda x: (TEAM_MAP[x[0]], TEAM_MAP[x[1]]))
    matchups = compose(lookup_nickname, partition(2))
    away_spreads = map(compose(list, lambda s: islice(s, 0, None, 2)))
    get_spreads = map(compose(convert_spreads, parse_spreads))
    calc_spreads = compose(list, map(statistics.mean), away_spreads,
                           filter(lambda x: len(x) > 0), get_spreads)

    def parse_odds_xml(p):
        team_names = [str.lower(json.loads(x.attrib['data-op-name'])['short_name'])
                      for x in p.xpath('//div[starts-with(@class, "op-matchup-team")]')]
        games = p.xpath(
            '//div[@id="op-results"]')[0].xpath('div[starts-with(@class, "op-item-row-wrapper")]')
        return zip(matchups(team_names), calc_spreads(games))

    def spread2score(compare, spread):
        return round2(abs(spread) if compare(spread, 0) else 0)

    result = compose(list, parse_odds_xml, get_odds_html)(odds)
    return {
        teams: Game(Scored(teams[0], spread2score(operator.lt, spread)),
                    Scored(teams[1], spread2score(operator.ge, spread)), int(yr), int(wk))
        for (teams, spread) in result
    }


def run(opts, write_fh):
    start_date = parse_date(opts['<start>'])
    end_date = (parse_date(opts['<end>']) if opts['<end>'] else
                start_date)

    if opts['get-results']:
        write_game_results(write_fh, first(score_scrape(
            start_date.year, start_date.week, end_date.week)))
        return

    standings = compose(simplify, curry(get_team_stats)((start_date.year, end_date.year)))

    with open(opts['<records-file>'], 'r') as records:
        if opts['standings']:
            write_summary(write_fh, standings(records))
            return

        pick_week = parse_date(opts['<pick-week>'])
        predictions = predict_winners(
            standings(records), first(score_scrape(*pick_week)), spread_scrape(*pick_week))
        write_predictions(write_fh, predictions)


def parse_date(date):
    try:
        x = ''.join([c for c in date if c.isdigit()])
        return Week(int(x[:4]), int(x[4:]))
    except:
        return Week(datetime.datetime.now().year, 0)


def main():
    opts = docopt(__doc__)
    write_fh = open(opts['--output'], 'w') if opts['--output'] else sys.stdout
    try:
        run(opts, write_fh)
    finally:
        if not write_fh == sys.stdout:
            write_fh.close()


if __name__ == '__main__':
    main()
