#!/usr/bin/env python


"""Pick Picker


Usage:
  picker.py get-results <year> <start-week> [<end-week>] [--output CSV]
  picker.py make-picks <records-file> <year> <week> [--output CSV]


Options:
  -o --output CSV    output to CSV file, instead of default STDOUT
  -h --help          display this help


"""


import operator
import requests
import sys

from collections import namedtuple
from concurrent import futures
from docopt import docopt
from itertools import repeat
from lxml import html
from toolz import interleave


Game = namedtuple('Game', 'year week away away_pts home home_pts')
Team = namedtuple('Team', 'name win_pct pyth_pct pts_per_game record')
Record = namedtuple('Record', 'name points record')
Prediction = namedtuple('Prediction', 'winner loser pyth_delta')


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
                    operator.sub(*rec.points)/GAME_COUNT, rec)
    records = {}

    for game in games:
        records[game.away] = get_record(game.away, records)
        records[game.home] = get_record(game.home, records)
        winner, loser = eval_game(
            game, records[game.away], records[game.home])
        records[winner.name] = winner
        records[loser.name] = loser

    return {team: calc_team(team, record) for team, record in records.items()}


def picker(file_name, year, week):

    def new_game(year, week, away, ascore, home, hscore):
        return Game(int(year), int(week), away, int(ascore), home, int(hscore))

    def csv2game(line):
        return new_game(*line.strip().split(','))

    def get_games(file_name):
        with open(file_name, 'r') as fh:
            return [csv2game(line) for line in fh.readlines()]

    def predict(away, home):
        if away.pyth_pct > home.pyth_pct:
            return Prediction(away, home, away.pyth_pct - home.pyth_pct)
        else:
            return Prediction(home, away, home.pyth_pct - away.pyth_pct)

    teams = team_calculator(get_games(file_name))

    predict_week = [csv2game(line) 
                    for line in score_scrape(year, week, -1).split('\n')]

    # Prediction = namedtupel('Prediction', 'winner loser pyth_delta')
    predictions = [predict(teams[game.away], teams[game.home]) for game in
                   predict_week]

    for x in sorted(predictions, key=lambda t: t.pyth_delta, reverse=True):
        print(x)
    
    for x in sorted(predictions, key=lambda t: t.pyth_delta, reverse=True):
        print('{},{},{}'.format(x.winner.name, x.loser.name, x.pyth_delta))


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
        return '\n'.join(sorted(ex.map(scrape_single, repeat(yr), [wk for wk in range(wk_from, wk_to + step, step)]), reverse=True))

    return \
        scrape_single(yr, wk_from) if wk_to == -1 else \
        scrape_weeks(yr, wk_from, wk_to)


if __name__ == '__main__':
    doc = docopt(__doc__)
    if doc['get-results']:
        csv = score_scrape(
            int(doc['<year>']), int(doc['<start-week>']),
            -1 if not doc['<end-week>'] else int(doc['<end-week>']))
        if doc['--output']:
            with open(doc['--output'], 'w') as fh:
                fh.write(csv)
        else:
            print(csv)
    elif doc['make-picks']:
        picker(doc['<records-file>'], int(doc['<year>']), int(doc['<week>']))
