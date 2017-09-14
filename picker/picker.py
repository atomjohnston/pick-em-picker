#!/usr/bin/env python


"""Pick Picker


Usage:
  picker.py get-results <year> <start-week> [<end-week>] [--output CSV]
  picker.py make-picks <records-file> <year> <week>
            [--output CSV] [--spread HTML]


Options:
  -o --output CSV    output to CSV file, instead of default STDOUT
  -s --spread HTML   HTML file of OddsShark odds page
  -h --help          display this help


"""


import json
import operator
import requests
import sys

from collections import namedtuple
from concurrent import futures
from docopt import docopt
from itertools import repeat, islice
from lxml import html
from toolz import interleave


Game = namedtuple('Game', 'sort_key year week away away_pts home home_pts')
Team = namedtuple('Team', 'name win_p pyth point_diff_avg point_avg record')
Record = namedtuple('Record', 'name points record')
Pick = namedtuple('Pick', 'won lost delta')
Guess = namedtuple('Guess', ('away home pyth points '
                             'wins spread pyth_spread'))


def new_game(year, week, away, ascore, home, hscore):
    return Game(int(str(year) + '{0:0>2}'.format(week)), int(year),
                int(week), away, float(ascore), home, float(hscore))


def team_calculator(games):

    GAME_COUNT = 16
    EXP = 2.37

    def add_records(r1, r2):
        if sum(r1.record) == GAME_COUNT:
            return r1
        else:
            return Record(
                name=r1.name,
                points=sum_tuples(r1.points, r2.points),
                record=sum_tuples(r1.record, r2.record))

    def sum_tuples(t1, t2):
        return tuple(map(operator.add, t1, t2))

    def get_record(name, records):
        return records[name] if name in records else \
               Record(name=name, points=(0, 0), record=(0, 0))

    def eval_game(game, a_rec, h_rec):
        winner, loser = (a_rec, h_rec) if game.away_pts > game.home_pts else \
                        (h_rec, a_rec)
        score = (game.away_pts, game.home_pts)
        return (
            add_records(
                winner, Record(winner.name, (max(score), min(score)), (1, 0))),
            add_records(
                loser, Record(loser.name, (min(score), max(score)), (0, 1))))

    def calc_team(name, rec):
        return Team(name, rec.record[0]/GAME_COUNT,
                    rec.points[0]**EXP/(rec.points[0]**EXP+rec.points[1]**EXP),
                    operator.sub(*rec.points)/GAME_COUNT,
                    rec.points[0]/GAME_COUNT, rec)
    records = {}

    for game in games:
        records[game.away] = get_record(game.away, records)
        records[game.home] = get_record(game.home, records)
        winner, loser = eval_game(
            game, records[game.away], records[game.home])
        records[winner.name] = winner
        records[loser.name] = loser

    return {team: calc_team(team, record) for team, record in records.items()}


def picker(file_name, year, week, spreads_html):

    def csv2game(line):
        return new_game(*line.strip().split(','))

    def get_games(file_name):
        with open(file_name, 'r') as fh:
            return sorted([csv2game(line) for line in fh.readlines()],
                          key=lambda g: g.sort_key, reverse=True)

    def pick(away, a_val, home, h_val):
        if a_val > h_val:
            return Pick(away.name, home.name, a_val - h_val)
        else:
            return Pick(home.name, away.name, h_val - a_val)

    def guess(stats, game, projected_winners):
        away, home = (stats[game.away][0], stats[game.home][0])
        p_away, p_home = (stats[game.away][1], stats[game.home][1])
        p_game = projected_winners[(game.away, game.home)]
        return Guess(
            away=away, home=home,
            pyth=pick(away, away.pyth, home, home.pyth),
            points=pick(away, away.point_diff_avg, home, home.point_diff_avg),
            wins=pick(away, away.win_p, home, home.win_p),
            spread=pick(p_away, p_game.away_pts, p_home, p_game.home_pts),
            pyth_spread=pick(p_away, p_away.pyth, p_home, p_home.pyth))

    played_games = get_games(file_name)
    teams = team_calculator(played_games)

    projected_games = spread_scrape(year, week, spreads_html)
    projected_teams = team_calculator(
        list(projected_games.values()) + played_games)

    team_stats = {name: (stats, projected_teams[name])
                  for name, stats in teams.items()}

    guessing_games = score_scrape(year, week, -1).split('\n')
    return [guess(team_stats, game, projected_games) for game in
            [csv2game(line) for line in guessing_games]]


def write_predictions(file_, guesses):

    def format_pick(idx, pick):
        return ','.join([pick.won, str(pick.delta), str(idx)])

    def format_guesses(pick_map, guesses, sel_fn):
        i = len(guesses)
        for x in sorted(guesses, key=lambda x: sel_fn(x).delta, reverse=True):
            pick = sel_fn(x)
            pick_map[(x.away.name, x.home.name)].append(
                [pick.won, str(pick.delta), str(i)])
            i -= 1

    pick_map = {(g.away.name, g.home.name): [] for g in guesses}

    format_guesses(pick_map, guesses, lambda x: x.pyth)
    format_guesses(pick_map, guesses, lambda x: x.spread)
    format_guesses(pick_map, guesses, lambda x: x.points)
    format_guesses(pick_map, guesses, lambda x: x.wins)
    format_guesses(pick_map, guesses, lambda x: x.pyth_spread)

    print('away,home,'
          'pyth,pyth_r,spread,spread_r,points,points_r,'
          'wins,wins_r,p_spr,p_spr_r,'
          'pyth\u0394,spread\u0394,points\u0394,wins\u0394,p_spr\u0394,'
          'pyth_act,spread_act,points_act,wins_act,p_spr_act', file=file_)

    for key, val in pick_map.items():
        picks, deltas, ranks = zip(*val)
        print('{},{},{},0,0,0,0,0'.format(
            ','.join(key),
            ','.join(interleave([picks, ranks])),
            ','.join(deltas)),
            file=file_)


# def spread_scrape(year, week, t_stats, file_):
def spread_scrape(year, week, file_):
    URL = 'https://www.oddsshark.com/nfl/odds'
    team_map = {'ari': 'cardinals', 'atl': 'falcons', 'bal': 'ravens',
                'buf': 'bills', 'car': 'panthers', 'chi': 'bears',
                'cin': 'bengals', 'cle': 'browns', 'dal': 'cowboys',
                'den': 'broncos', 'det': 'lions', 'gb': 'packers',
                'hou': 'texans', 'ind': 'colts', 'jac': 'jaguars',
                'kc': 'chiefs', 'lac': 'chargers', 'lar': 'rams',
                'min': 'vikings', 'ne': 'patriots', 'no': 'saints',
                'nyg': 'giants', 'nyj': 'jets', 'oak': 'raiders',
                'phi': 'eagles', 'pit': 'steelers', 'sea': 'seahawks',
                'sf': '49ers', 'ten': 'titans', 'was': 'redskins',
                'mia': 'dolphins', 'tb': 'buccaneers'}

    def parse_odds_xml(odds_file):
        if odds_file:
            with open(odds_file, 'r') as f:
                p = html.fromstring(f.read())
        else:
            p = html.fromstring(requests.get(URL).content)

        teams = [str.lower(json.loads(x.attrib['data-op-name'])['short_name'])
                 for x in
                 p.xpath('//div[starts-with(@class, "op-matchup-team")]')]
        games = p.xpath('//div[@id="op-results"]')[0] \
                 .xpath('div[starts-with(@class, "op-item-row-wrapper")]')
        return (teams, games)

    def parse_spreads(g):
        return [json.loads(x.attrib['data-op-info'])['fullgame'] for x in
                g.xpath('div/div[starts-with(@class, "op-item op-spread")]')]

    teams, games = parse_odds_xml(file_)
    matchups = [(team_map[teams[n]], team_map[teams[n+1]])
                for n in range(0, len(teams) - 1, 2)]

    predictions = {}
    n = 0
    for g in games:
        diffs = parse_spreads(g)
        spreads = [0 if n == 'Ev' else float(n) for n in diffs if n != '']
        if len(spreads) == 0:
            continue
        avg_spread = \
            sum([n for n in islice(spreads, 0, None, 2)]) / (len(spreads) / 2)
        scores = (abs(avg_spread), 0) if avg_spread < 0 else \
                 (0, abs(avg_spread))
        game = new_game(
            year, week, matchups[n][0], scores[0], matchups[n][1], scores[1])
        predictions[(game.away, game.home)] = game
        n += 1

    return predictions


def score_scrape(yr, wk_from, wk_to):

    URL = 'https://www.pro-football-reference.com/years/{}/week_{}.htm'

    def select_games(xml):
        return xml.xpath('//div[starts-with(@class,"game_summary")]'
                         '/table[@class="teams"]'
                         '/tbody')

    def select_teams(game_xml):
        return [x.text.split()[-1]
                for x in
                game_xml.xpath('tr/td/a[starts-with(@href, "/teams/")]')]

    def select_scores(game_xml):
        score = [x.text for x in game_xml.xpath('tr/td[@class="right"]')][0:2]
        if not score[0] or not score[1]:
            return ['-1', '-1']
        else:
            return score

    def fetch_page_html(yr, wk):
        return html.fromstring(requests.get(URL.format(yr, wk)).content)

    def game_csv(yr, wk, gm):
        return ','.join(
            [str(yr), '{0:0>2}'.format(wk)] +
            list(interleave([select_teams(gm), select_scores(gm)])))

    def scrape_single(yr, week):
        return '\n'.join([str.lower(game_csv(yr, week, game))
                         for game in select_games(fetch_page_html(yr, week))])

    def scrape_weeks(yr, wk_from, wk_to):
        step = 1 if wk_from < wk_to else -1
        ex = futures.ThreadPoolExecutor(max_workers=4)
        return '\n'.join(sorted(ex.map(
            scrape_single, repeat(yr),
            [wk for wk in range(wk_from, wk_to + step, step)]), reverse=True))

    return \
        scrape_single(yr, wk_from) if wk_to == -1 else \
        scrape_weeks(yr, wk_from, wk_to)


def main(write_fh, doc):
    if doc['get-results']:
        csv = score_scrape(
            int(doc['<year>']), int(doc['<start-week>']),
            -1 if not doc['<end-week>'] else int(doc['<end-week>']))
        write_fh.write(csv)
    elif doc['make-picks']:
        predictions = picker(
            doc['<records-file>'], int(doc['<year>']),
            int(doc['<week>']), doc['--spread'])
        write_predictions(write_fh, predictions)
    elif doc['spread-scrape']:
        spread_scrape(doc['<year>'], doc['<week>'], doc['<file>'])


if __name__ == '__main__':
    doc = docopt(__doc__)
    write_fh = open(doc['--output'], 'w') if doc['--output'] else sys.stdout
    try:
        main(write_fh, doc)
    finally:
        write_fh.close()
