#!/usr/bin/env python


"""Pro Football Pick 'Em Picker


Usage:
  pick-em get-results <year> <start-week> [<end-week>] [--output CSV]
  pick-em simple-ranking <records-file> <year> <start-week>
  pick-em make-picks <records-file> <year> <week>
            [--output CSV] [--spread HTML] [--season]


Options:
  -o --output CSV    output to CSV file, instead of default STDOUT
  -s --spread HTML   HTML file of OddsShark odds page
  --season           just run numbers for this season
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
from toolz import interleave, frequencies, concat, pipe
from typing import (NamedTuple, Tuple, TextIO, Any, List, Dict, Callable,
                    Sequence, Union)
from pprint import pprint


IntPair = Tuple[Any, ...]

Opponents = Tuple[str, ...]

Game = NamedTuple('Game', [('sort_key', int), ('year', int), ('week', int),
                           ('away', str), ('away_pts', float),
                           ('home', str), ('home_pts', float)])

Record = NamedTuple('Record', [('name', str), ('points', IntPair),
                               ('record', IntPair), ('opponents', Opponents)])

Team = NamedTuple('Team', [('name', str), ('win_p', float), ('pyth', float),
                           ('mov', float), ('record', Record), ('srs', float)])

Pick = NamedTuple('Pick', [('won', str), ('lost', str), ('delta', float)])

Ranked = NamedTuple('Ranked', [('victor', str), ('delta', float),
                               ('rank', int)])

Guess = NamedTuple('Guess', [('away', Team), ('home', Team), ('pyth', Pick),
                             ('points', Pick), ('wins', Pick),
                             ('spread', Pick), ('srs', Pick)])

Matchup = Tuple[str, str]

PickMap = Dict[Matchup, List[Ranked]]

EXP = 2.37
SRS_X = 1000


def new_game(year: str, week: str, away: str, ascore: str,
             home: str, hscore: str) -> Game:
    return Game(int(str(year) + '{0:0>2}'.format(week)), int(year),
                int(week), away, float(ascore), home, float(hscore))


def calculate_team_stats(games: List[Game]) -> Dict[str, Team]:

    def add_records(r1: Record, r2: Record) -> Record:
        if sum(r1.record) >= 16:
            return r1
        else:
            return Record(
                name=r1.name,
                points=sum_tuples(r1.points, r2.points),
                record=sum_tuples(r1.record, r2.record),
                opponents=r1.opponents + r2.opponents)

    def sum_tuples(t1: IntPair, t2: IntPair) -> IntPair:
        return tuple(map(operator.add, t1, t2))

    def get_record(name: str, records: Dict[str, Record]) -> Record:
        return records[name] if name in records else \
            Record(name=name, points=(0, 0), record=(0, 0), opponents=())

    def eval_game(game: Game, a_rec: Record,
                  h_rec: Record) -> Tuple[Record, Record]:
        winner, loser = (a_rec, h_rec) if game.away_pts > game.home_pts else \
                        (h_rec, a_rec)
        score = (game.away_pts, game.home_pts)
        return (add_records(winner, Record(winner.name, (max(score), min(score)), (1, 0), (loser.name,))),
                add_records(loser, Record(loser.name, (min(score), max(score)), (0, 1), (winner.name,))))

    def calc_team(name: str, rec: Record) -> Team:
        return Team(name, rec.record[0] / sum(rec.record),
                    rec.points[0]**EXP /
                    (rec.points[0]**EXP + rec.points[1]**EXP),
                    operator.sub(*rec.points) / sum(rec.record), rec, 0.0)

    records = {}  # type: Dict[str, Record]

    for game in games:
        records[game.away] = get_record(game.away, records)
        records[game.home] = get_record(game.home, records)
        winner, loser = eval_game(
            game, records[game.away], records[game.home])
        records[winner.name] = winner
        records[loser.name] = loser

    return {team: calc_team(team, record) for team, record in records.items()}


def csv2game(line: str) -> Game:
    return new_game(*line.strip().split(','))


def get_played_games(scores_file: str) -> List[Game]:
    with open(scores_file, 'r') as fh:
        return sorted([csv2game(line) for line in fh.readlines()],
                      key=lambda g: g.sort_key, reverse=True)


def picker(team_stats: Dict[str, Team], spreads_html: str,
           year: int, week: int) -> List[Guess]:

    def pick(away: Team, a_val: float, home: Team, h_val: float) -> Pick:
        if a_val > h_val:
            return Pick(away.name, home.name, abs(a_val - h_val))
        else:
            return Pick(home.name, away.name, abs(h_val - a_val))

    def guess(stats: Dict[str, Team], game: Game,
              projected_winners: Dict[Matchup, Game]) -> Guess:
        away, home = (stats[game.away], stats[game.home])
        p_game = projected_winners[(game.away, game.home)]
        return Guess(
            away=away, home=home,
            pyth=pick(away, away.pyth, home, home.pyth),
            points=pick(away, away.mov, home, home.mov),
            wins=pick(away, away.win_p, home, home.win_p),
            srs=pick(away, away.srs, home, home.srs + (2 * SRS_X)),
            spread=pick(away, p_game.away_pts, home, p_game.home_pts))

    projected_games = spread_scrape(str(year), str(week), spreads_html)
    guessing_games = score_scrape(year, week, -1).split('\n')

    return [guess(team_stats, game, projected_games) for game in
            [csv2game(line) for line in guessing_games]]


def simple_ranking(stats: Dict[str, Team]) -> Dict[str, Team]:

    def adjust(srs: Dict[str, int], sos: Dict[str, int]
               ) -> Tuple[Dict[str, int], Dict[str, int], int]:
        new_srs, delta = {}, 0
        for team, t_stat in stats.items():
            prev_sos = sos[team]
            sos[team] = int(sum([srs[t] for t in t_stat.record.opponents]) /
                            sum(t_stat.record.record))
            delta = max(delta, abs(sos[team] - prev_sos))
            new_srs[team] = true_mov[team] + sos[team]
        return new_srs, sos, delta

    def set_srs(team: Team, srs: int) -> Team:
        team_values = [*team]
        team_values[-1] = srs
        return Team(*team_values)  # type: ignore

    true_mov = {team: int(t_stats.mov * SRS_X)
                for team, t_stats in stats.items()}

    srs, sos, delta = adjust(true_mov, defaultdict(lambda: 0))

    delta_ = 1
    while delta > 0 and delta != delta_:
        delta_ = delta
        srs, sos, delta = adjust(srs, sos)

    avg = int((sum(srs.values()) / len(srs)) + 0.5)
    return {_: set_srs(tm, srs[tm.name] - avg) for _, tm in stats.items()}


def rank_predictions(guesses: List[Guess]) -> Tuple[PickMap, PickMap]:

    def rank(pick_map: PickMap, guesses: List[Guess],
             sel_fn: Callable[[Guess], Pick]) -> None:
        i = len(guesses)
        for x in sorted(guesses, key=lambda x: sel_fn(x).delta, reverse=True):
            pick = sel_fn(x)
            pick_map[(x.away.name, x.home.name)].append(
                Ranked(pick.won, pick.delta, i))
            i -= 1

    def average_rank(matchup: Matchup,
                     values: List[Ranked]) -> Tuple[Matchup, str, float]:
        victors, deltas, ranks = zip(*values)
        away, home = matchup
        avg = defaultdict(lambda: 0)  # type: Dict[str, float]
        for n, victor in enumerate(victors):
            avg[victor] = avg[victor] + int(ranks[n])
        consensus = frequencies(victors)
        if away in consensus and consensus[away] >= 2:
            winner, loser = (away, home)
        else:
            winner, loser = (home, away)
        winner_avg = (avg[winner] - avg[loser]) / len(victors)
        avg_winner = winner if winner_avg >= 0 else loser
        return (matchup, avg_winner, abs(winner_avg))

    picks = defaultdict(list)  # type: PickMap

    rank(picks, guesses, lambda x: x.pyth)
    rank(picks, guesses, lambda x: x.spread)
    rank(picks, guesses, lambda x: x.wins)

    average_winners = [average_rank(k, v) for k, v in picks.items()]

    rank(picks, guesses, lambda x: x.points)
    rank(picks, guesses, lambda x: x.srs)

    return (picks, {x[0]: [Ranked(x[1], x[2], n + 1)]
                    for n, x in
                    enumerate(sorted(average_winners, key=lambda k: k[-1]))})


def write_predictions(file_: TextIO,
                      pick_map: PickMap,
                      averages: PickMap) -> None:

    def strings(list_: Sequence[Union[str, int]]) -> List[str]:
        return [str(x) for x in list_]

    print('away,home,winner,me,me_r,'
          'avg,avg_r,pyth,pyth_r,spread,spread_r,wins,wins_r,'
          'points,points_r,srs,srs_r,'
          'avg\u0394,pyth\u0394,spread\u0394,wins\u0394,'
          'points\u0394,srs\u0394,'
          'me_act,,pyth_act,,spread_act,,'
          'wins_act,,points_act,,srs_act,,', file=file_)

    for key, val in pick_map.items():
        picks, deltas, ranks = zip(*val)
        a_pix, a_delts, a_ranx = zip(*averages[key])
        print('{},,x,0,{},{},0,,0,,0,,0,,0,,0,'.format(
            ','.join(key),
            ','.join(interleave([strings(a_pix + picks),
                                 strings(a_ranx + ranks)])),
            ','.join(strings(a_delts + deltas))),
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
    matchups = [(team_map[teams[n]], team_map[teams[n + 1]])
                for n in range(0, len(teams) - 1, 2)]

    predictions = {}  # type: Dict[Tuple[str, str], Game]
    n = 0
    for g in games:
        diffs = parse_spreads(g)
        spreads = [0 if n == 'Ev' or n == '' else float(n) for n in diffs]
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


def run(write_fh: TextIO, doc: Any) -> None:
    if doc['get-results']:
        csv = score_scrape(
            int(doc['<year>']), int(doc['<start-week>']),
            -1 if not doc['<end-week>'] else int(doc['<end-week>']))
        write_fh.write(csv)
    else:
        teams = pipe(doc['<records-file>'], get_played_games,
                     calculate_team_stats, simple_ranking)
        if doc['make-picks']:
            predictions = picker(
                teams, doc['--spread'],
                int(doc['<year>']), int(doc['<week>']))
            ranks, averages = rank_predictions(predictions)
            write_predictions(write_fh, ranks, averages)
        elif doc['simple-ranking']:
            for t in sorted(teams.values(), key=lambda x: x.srs, reverse=True):
                # print('{0:12s} {1:6.2f}'.format(t.name, t.srs/SRS_X))
                print('{0:12s} {1:6}'.format(t.name, t.srs))


def main():
    doc = docopt(__doc__)
    write_fh = open(doc['--output'], 'w') if doc['--output'] else sys.stdout
    try:
        run(write_fh, doc)
    finally:
        write_fh.close()


if __name__ == '__main__':
    main()
