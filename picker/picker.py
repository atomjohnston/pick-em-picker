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

from collections import namedtuple, defaultdict
from concurrent import futures
from docopt import docopt
from itertools import repeat, islice
from lxml import html  # type: ignore
from toolz import interleave, frequencies, concat
from typing import NamedTuple, Tuple, TextIO, Any, List, Dict, Callable


PickMap = Dict[Tuple[str, str], List[List[Any]]]
IntPair = Tuple[Any, ...]

Game = NamedTuple('Game', [('sort_key', int), ('year', int), ('week', int),
                           ('away', str), ('away_pts', float),
                           ('home', str), ('home_pts', float)])

Record = NamedTuple('Record', [('name', str), ('points', IntPair),
                               ('record', IntPair)])

Team = NamedTuple('Team', [('name', str), ('win_p', float), ('pyth', float),
                           ('point_diff_avg', float), ('point_avg', float),
                           ('record', Record)])

Pick = NamedTuple('Pick', [('won', str), ('lost', str), ('delta', float)])

Guess = NamedTuple('Guess', [('away', Team), ('home', Team), ('pyth', Pick),
                             ('points', Pick), ('wins', Pick),
                             ('spread', Pick), ('pyth_spread', Pick)])


def new_game(year: str, week: str,
             away: str, ascore: str,
             home: str, hscore: str) -> Game:
    return Game(int(str(year) + '{0:0>2}'.format(week)), int(year),
                int(week), away, float(ascore), home, float(hscore))


def team_calculator(games: List[Game]) -> Dict[str, Team]:

    GAME_COUNT = 16
    EXP = 2.37

    def add_records(r1: Record, r2: Record) -> Record:
        if sum(r1.record) == GAME_COUNT:
            return r1
        else:
            return Record(
                name=r1.name,
                points=sum_tuples(r1.points, r2.points),
                record=sum_tuples(r1.record, r2.record))

    def sum_tuples(t1: IntPair, t2: IntPair) -> IntPair:
        return tuple(map(operator.add, t1, t2))

    def get_record(name: str, records: Dict[str, Record]) -> Record:
        return records[name] if name in records else \
               Record(name=name, points=(0, 0), record=(0, 0))

    def eval_game(game: Game, a_rec: Record,
                  h_rec: Record) -> Tuple[Record, Record]:
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

    records = {}  # type: Dict[str, Record]

    for game in games:
        records[game.away] = get_record(game.away, records)
        records[game.home] = get_record(game.home, records)
        winner, loser = eval_game(
            game, records[game.away], records[game.home])
        records[winner.name] = winner
        records[loser.name] = loser

    return {team: calc_team(team, record) for team, record in records.items()}


def picker(file_name: str, spreads_html: str,
           year: int, week: int) -> List[Guess]:

    def csv2game(line: str) -> Game:
        return new_game(*line.strip().split(','))

    def get_games(file_name: str) -> List[Game]:
        with open(file_name, 'r') as fh:
            return sorted([csv2game(line) for line in fh.readlines()],
                          key=lambda g: g.sort_key, reverse=True)

    def pick(away: Team, a_val: float, home: Team, h_val: float) -> Pick:
        if a_val > h_val:
            return Pick(away.name, home.name, a_val - h_val)
        else:
            return Pick(home.name, away.name, h_val - a_val)

    def guess(stats: Dict[str, Tuple[Team, Team]],
              game: Game,
              projected_winners: Dict[Tuple[str, str], Game]) -> Guess:
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

    projected_games = spread_scrape(str(year), str(week), spreads_html)
    projected_teams = team_calculator(
        list(projected_games.values()) + played_games)

    team_stats = {name: (stats, projected_teams[name])
                  for name, stats in teams.items()}

    guessing_games = score_scrape(year, week, -1).split('\n')
    return [guess(team_stats, game, projected_games) for game in
            [csv2game(line) for line in guessing_games]]


def rank_predictions(guesses: List[Guess]) -> Tuple(PickMap):

    def rank(pick_map: PickMap, guesses: List[Guess],
             sel_fn: Callable[[Guess], Pick]) -> None:
        i = len(guesses)
        for x in sorted(guesses, key=lambda x: sel_fn(x).delta, reverse=True):
            pick = sel_fn(x)
            pick_map[(x.away.name, x.home.name)].append(
                [pick.won, str(pick.delta), str(i)])
            i -= 1

    def average_rank(matchup, values):
        victors, deltas, ranks = zip(*values)
        away, home = matchup
        avg = defaultdict(lambda: 0)
        for n, victor in enumerate(victors):
            avg[victor] = avg[victor] + int(ranks[n])
        consensus = frequencies(victors)
        winner, loser = (away, home) if away in consensus and consensus[away] >= 2 else \
                        (home, away) 
        winner_avg = (avg[winner] - avg[loser]) / len(victors)
        avg_winner = winner if winner_avg >=0 else loser
        return [avg_winner, abs(winner_avg)]

    picks = defaultdict(list)

    rank(picks, guesses, lambda x: x.pyth)
    rank(picks, guesses, lambda x: x.spread)
    rank(picks, guesses, lambda x: x.wins)

    average_winners = [average_rank(k, v) for k, v in picks.items()]

    rank(picks, guesses, lambda x: x.points)
    rank(picks, guesses, lambda x: x.pyth_spread)

    return (picks, [x for x in sorted(average_winners, key=lambda k: k[1])])


def write_predictions(file_: TextIO,
                      pick_map: PickMap,
                      averages: List[Tuple[str, int]]) -> None:

    print('away,home,'
          'pyth,pyth_r,spread,spread_r,wins,wins_r,'
          'avg,avg_r,',
          'points,points_r,p_spr,p_spr_r,'
          'pyth\u0394,spread\u0394,wins\u0394,avg\u0394,points\u0394,p_spr\u0394,'
          'pyth_act,spread_act,wins_act,points_act,p_spr_act', file=file_)


    for key, val in pick_map.items():
        picks, deltas, ranks = zip(*val)
        print('{},{},{},{},0,0,0,0,0'.format(
            ','.join(key),
            ','.join(interleave([picks, ranks])),
            ','.join(deltas)),
            file=file_)


def spread_scrape(yr: str, wk: str, odds: str) -> Dict[Tuple[str, str], Game]:
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

    def parse_odds_xml(odds_file: str) -> Tuple[List[str], List[Any]]:
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

    def parse_spreads(g: Any) -> List[str]:
        return [json.loads(x.attrib['data-op-info'])['fullgame'] for x in
                g.xpath('div/div[starts-with(@class, "op-item op-spread")]')]

    teams, games = parse_odds_xml(odds)
    matchups = [(team_map[teams[n]], team_map[teams[n+1]])
                for n in range(0, len(teams) - 1, 2)]

    predictions = {}  # type: Dict[Tuple[str, str], Game]
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
        game = new_game(yr, wk,
                        matchups[n][0], str(scores[0]),
                        matchups[n][1], str(scores[1]))
        predictions[(game.away, game.home)] = game
        n += 1

    return predictions


def score_scrape(yr: int, wk_from: int, wk_to: int) -> str:

    URL = 'https://www.pro-football-reference.com/years/{}/week_{}.htm'

    def select_games(xml: Any) -> Any:
        return xml.xpath('//div[starts-with(@class,"game_summary")]'
                         '/table[@class="teams"]'
                         '/tbody')

    def select_teams(game_xml: Any) -> List[str]:
        return [x.text.split()[-1]
                for x in
                game_xml.xpath('tr/td/a[starts-with(@href, "/teams/")]')]

    def select_scores(game_xml: Any) -> List[str]:
        score = [x.text for x in game_xml.xpath('tr/td[@class="right"]')][0:2]
        if not score[0] or not score[1]:
            return ['-1', '-1']
        else:
            return score

    def fetch_page_html(yr: int, wk: int) -> Any:
        return html.fromstring(requests.get(URL.format(yr, wk)).content)

    def game_csv(yr: int, wk: int, gm: str) -> str:
        return ','.join(
            [str(yr), '{0:0>2}'.format(wk)] +
            list(interleave([select_teams(gm), select_scores(gm)])))

    def scrape_single(yr: int, week: int) -> str:
        return '\n'.join([str.lower(game_csv(yr, week, game))
                         for game in select_games(fetch_page_html(yr, week))])

    def scrape_weeks(yr: int, wk_from: int, wk_to: int) -> str:
        step = 1 if wk_from < wk_to else -1
        ex = futures.ThreadPoolExecutor(max_workers=4)
        return '\n'.join(sorted(ex.map(
            scrape_single, repeat(yr),
            [wk for wk in range(wk_from, wk_to + step, step)]), reverse=True))

    return \
        scrape_single(yr, wk_from) if wk_to == -1 else \
        scrape_weeks(yr, wk_from, wk_to)


def main(write_fh: TextIO, doc: Any) -> None:
    if doc['get-results']:
        csv = score_scrape(
            int(doc['<year>']), int(doc['<start-week>']),
            -1 if not doc['<end-week>'] else int(doc['<end-week>']))
        write_fh.write(csv)
    elif doc['make-picks']:
        predictions = picker(
            doc['<records-file>'], doc['--spread'],
            int(doc['<year>']), int(doc['<week>']))
        ranks, averages = rank_predictions(predictions)
        write_predictions(write_fh, ranks, averages)
    elif doc['spread-scrape']:
        spread_scrape(doc['<year>'], doc['<week>'], doc['<file>'])


if __name__ == '__main__':
    doc = docopt(__doc__)
    write_fh = open(doc['--output'], 'w') if doc['--output'] else sys.stdout
    try:
        main(write_fh, doc)
    finally:
        write_fh.close()
