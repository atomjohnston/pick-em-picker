#!/usr/bin/env python

import csv
import json
import operator
import statistics
import sys

import click
import numpy as np
import requests

from collections import namedtuple
from concurrent import futures
from datetime import datetime
from functools import reduce
from itertools import islice, repeat, takewhile

from lxml import html
from toolz import compose, concat, curry, flip, interleave, pipe, valmap
from toolz.curried import filter, first, map, partition


Week = namedtuple("Week", "year week")
Scored = namedtuple("Scored", "team points")
Game = namedtuple("Game", "away home year week ")
Record = namedtuple("Record", "team opponents games wins losses ties points_for points_against")
Stats = namedtuple(
    "Stats",
    (
        "team win_rate points_per against_per point_delta margin_of_victory "
        "strength_of_schedule simple_ranking pythagorean"
    ),
)
Summary = namedtuple("Summary", "team record stats")
Pick = namedtuple("Pick", "winner rank delta")
Picks = namedtuple(
    "Picks",
    (
        "id game point_spread point_spread_1 simple_ranking simple_ranking_1 "
        "pythagorean margin_of_victory wins"
    ),
)

PYTH_EX = 2.37
TEAM_MAP = {
    "ari": "cardinals",
    "atl": "falcons",
    "bal": "ravens",
    "buf": "bills",
    "car": "panthers",
    "chi": "bears",
    "cin": "bengals",
    "cle": "browns",
    "dal": "cowboys",
    "den": "broncos",
    "det": "lions",
    "gb": "packers",
    "hou": "texans",
    "ind": "colts",
    "jac": "jaguars",
    "kc": "chiefs",
    "lac": "chargers",
    "lar": "rams",
    "lv": "raiders",
    "min": "vikings",
    "ne": "patriots",
    "no": "saints",
    "nyg": "giants",
    "nyj": "jets",
    "phi": "eagles",
    "pit": "steelers",
    "sea": "seahawks",
    "sf": "49ers",
    "ten": "titans",
    "was": "team",
    "mia": "dolphins",
    "tb": "buccaneers",
}


round2 = flip(round, 2)  # pylint: disable=E1120
round3 = flip(round, 3)  # pylint: disable=E1120


class TeamRecord(Record):
    def __add__(self, other):
        if not self.team == other.team:
            raise ValueError(f"cannot add {self.team} to {other.team}")
        return TeamRecord(
            self.team,
            self.opponents + (other.opponents,),
            *tuple(map(operator.add, self[2:], other[2:])),
        )


def win_loss_tie(pf, pa):
    if pf == pa:
        return (0, 0, 1)
    return (1, 0, 0) if pf > pa else (0, 1, 0)


def add_stats(stats, team, opponent):
    return stats + TeamRecord(
        team.team,
        opponent.team,
        1,
        *win_loss_tie(team.points, opponent.points),
        team.points,
        opponent.points,
    )


def append_game(table, game):
    table[game.away.team] = add_stats(table[game.away.team], game.away, game.home)
    table[game.home.team] = add_stats(table[game.home.team], game.home, game.away)
    return table


def dict2game(src):
    src.update({k: int(v) for (k, v) in src.items() if not k.endswith("team")})
    away = Scored(src.pop("away_team"), src.pop("away_points"))
    home = Scored(src.pop("home_team"), src.pop("home_points"))
    return Game(away, home, **src)


def calc_home_field(games):
    a_h = [(g.away.points, g.home.points) for g in games]
    points = tuple(reduce(map(operator.add), a_h, (0, 0)))
    return (points[1] / len(a_h)) - (points[0] / len(a_h))


def calc(record):
    delta = record.points_for - record.points_against
    mov = round2(delta / record.games)
    return Stats(
        record.team,  # team
        round3(record.wins / record.games),  # win_rate
        round2(record.points_for / record.games),  # points_per
        round2(record.points_against / record.games),  # against_per
        delta,  # point_delta
        mov,  # margin_of_victory
        0,  # strength_of_schedule
        mov,  # simple_ranking
        round3(calc_pythagorean(record.points_for, record.points_against)),  # pythagorean
    )


def calc_pythagorean(pf, pa):
    return (pf ** PYTH_EX) / ((pf ** PYTH_EX) + (pa ** PYTH_EX))


def simplify(summary):
    def update_sum_stats(from_, smry, sos, srs):
        return smry._replace(
            stats=smry.stats._replace(strength_of_schedule=sos, simple_ranking=srs)
        )

    def adjust(summaries, t_sum):
        opps = [summaries[name].stats.simple_ranking for name in t_sum.record.opponents]
        sos = statistics.mean(opps)
        # if t_sum.team == 'patriots':
        #     print(opps, 'sos:', sos)
        return update_sum_stats('adjust', t_sum, sos, t_sum.stats.margin_of_victory + sos)

    def correct(summaries):
        mean = statistics.mean([x.stats.simple_ranking for x in summaries.values()])
        #print('corrected mean', mean)
        return valmap(
            lambda s: update_sum_stats(
                'correct', s, s.stats.strength_of_schedule - mean, s.stats.simple_ranking - mean
            ),
            summaries,
        )

    def round_all(summaries):
        return valmap(
            lambda s: update_sum_stats(
                'round_all', s, round2(s.stats.strength_of_schedule), round2(s.stats.simple_ranking)
            ),
            summaries,
        )

    def calculate_all(previous):
        p_drift = None
        for n in range(9999):
            adjustments = correct(
                {name: adjust(previous, smry) for (name, smry) in previous.items()}
            )
            drift = max(
                [
                    abs(current.stats.strength_of_schedule - previous.stats.strength_of_schedule)
                    for (previous, current) in zip(previous.values(), adjustments.values())
                ]
            )
            print(n, drift)
            if drift <= 0.0001:
                return round_all(adjustments)
            previous = adjustments
        return round_all(adjustments)

    def r_calculate_all(i, previous):
        #print(i, 'previous', previous['patriots'].stats.strength_of_schedule, previous['patriots'].stats.simple_ranking, previous['patriots'].stats.margin_of_victory)
        adjustments = correct({name: adjust(previous, smry) for (name, smry) in previous.items()})
        #print(i, 'adjustments', adjustments['patriots'].stats.strength_of_schedule, adjustments['patriots'].stats.simple_ranking, adjustments['patriots'].stats.margin_of_victory)
        drift = max(
            [
                abs(current.stats.strength_of_schedule - previous.stats.strength_of_schedule)
                for (previous, current) in zip(previous.values(), adjustments.values())
            ]
        )
        #return round_all(adjustments)
        #print(i, '; '.join([f"{k}|{v.stats.simple_ranking}|{round2(v.stats.strength_of_schedule)}" for k, v in previous.items()]))
        print(i, '; '.join([f"{k}|{round2(v.stats.simple_ranking)}|{round2(v.stats.strength_of_schedule)}" for k, v in adjustments.items()]))
        try:
            return (round_all(adjustments) if drift <= 0.001 else
                    r_calculate_all(i + 1, adjustments))
        except RecursionError:
            return round_all(adjustments)

    #return r_calculate_all(1, summary)
    return calculate_all(summary)


def fancy_srs(teams):
    #first matrix with the coefficients of each of the variables
    terms = []

    #second matrix with the constant term (-average spread)
    solutions = []
    
    for name in sorted(teams.keys()):
        #add in a row for each team
        team = teams[name]
        row = []
        
        #rating = average spread + average opponent rating
        #-> -average spread = -rating + average opponent rating
        #-> -average spread = -rating + (number of opponents/1) * (opponent 1 rating+opponent 2 rating...)
        #each row of the matrix describes right side equation
        for o_name in sorted(teams.keys()):
            opp = teams[o_name]
            if opp.team == team.team:
                row.append(1)
            elif opp.team in team.record.opponents:
                row.append(-1.0/len(team.record.opponents))
            else:
                row.append(0)
        terms.append(row)
        
        #each row of this matrix describes the left side of the above equation
        solutions.append(team.stats.margin_of_victory)
    
    #solve the simultaneous equations using numpy
    for l in terms:
        print(l)
    print(solutions)
    solutions = np.linalg.solve(np.array(terms), np.array(solutions))
    sys.exit(0)

def simple_ranking(somery, correct = True, debug = False):
    tptr = {
        k: {
            'point_spread': v.stats.margin_of_victory,
            'played': v.record.opponents,
        }
        for k, v in somery.items()
    }
    for k in tptr:
        #tptr[k]['mov'] = tptr[k]['point_spread']/float(len(tptr[k]['played']))
        tptr[k]['mov'] = tptr[k]['point_spread']
        tptr[k]['srs'] = tptr[k]['mov']
        tptr[k]['sos'] = 0.0
    delta = 10.0
    while delta > 0.001:
        delta = 0.0
        print(tptr['patriots'])
        for k in tptr:
            sos = statistics.mean([tptr[g]['srs'] for g in tptr[k]['played']])
            tptr[k]['srs'] = tptr[k]['mov'] + sos
            newdelta = abs(sos - tptr[k]['sos'])
            tptr[k]['sos'] = sos
            delta = max(delta, newdelta)
        print(tptr['patriots'])
    #if correct:
    #    srs_correction( tptr )
    #if debug:
    #print("iters = {0:d}".format(iters)) 
    #print(tptr)
    sys.exit(0)
    return True


def write_summary(file_, team_summary):
    outcsv = csv.writer(file_)
    outcsv.writerow(
        [
            "team",
            "g",
            "w",
            "l",
            "t",
            "pf",
            "pa",
            "w%",
            "pf/g",
            "pa/g",
            "pd",
            "mov",
            "sos",
            "srs",
            "pyth",
        ]
    )
    for summary in sorted(team_summary.values(), key=lambda x: x.stats.win_rate, reverse=True):
        record = summary.record._asdict()
        stats = summary.stats._asdict()
        del record["opponents"]
        del stats["team"]
        outcsv.writerow(concat([record.values(), stats.values()]))


def write_game_results(file_, weeks):
    outcsv = csv.writer(file_)
    outcsv.writerow(["year", "week", "away_team", "away_points", "home_team", "home_points"])
    for results in weeks:
        for game in results:
            outcsv.writerow(
                [
                    game.year,
                    f"{game.week:02}",
                    game.away.team,
                    game.away.points,
                    game.home.team,
                    game.home.points,
                ]
            )


def write_predictions(file_, predictions):
    outcsv = csv.writer(file_)
    outcsv.writerow(
        [
            "#",
            "away",
            "home",
            "spread",
            "spread_r",
            "spread_d",
            "spread_m",
            "spread_m_r",
            "spread_m_d",
            "srs",
            "srs_r",
            "srs_d",
            "srs1",
            "srs1_r",
            "srs1_d",
            "pyth",
            "pyth_r",
            "pyth_d",
            "mov",
            "mov_r",
            "mov_d",
            "w%",
            "w%_r",
            "w%_d",
        ]
    )
    for picks in sorted(predictions, key=lambda x: x.simple_ranking.rank, reverse=True):
        outcsv.writerow(
            concat(
                [
                    [picks.id],
                    [picks.game.away.team, picks.game.home.team],
                    picks.point_spread,
                    picks.point_spread_1,
                    picks.simple_ranking,
                    picks.simple_ranking_1,
                    picks.pythagorean,
                    picks.margin_of_victory,
                    picks.wins,
                ]
            )
        )


def read_records(weeks, records):
    if len(weeks) > 0:
        predicate = lambda row: Week(int(row["year"]), int(row["week"])) in weeks
    else:
        predicate = lambda row: int(row["year"]) == datetime.now().year
    return takewhile(predicate, csv.DictReader(records))


def init_teams():
    return {team: TeamRecord(team, (), *list(repeat(0, 6))) for team in TEAM_MAP.values()}


def get_games(weeks, records_file):
    read_season_records = curry(read_records)(weeks)
    return compose(map(dict2game), read_season_records)(records_file)


def get_team_stats(games):
    records = reduce(append_game, games, init_teams())
    return {r.team: Summary(r.team, r, calc(r)) for r in records.values()}


def predict_winners(team_summary, games, spreads_avg, spreads_med, home_field):
    # pylint: disable=E1120,E1102
    print("home field advantage is", home_field, "points")

    def stats_for(team):
        return team_summary[team.team].stats

    @curry
    def predict(away, home, n_away, n_home, round_f=round2):
        delta = n_home - n_away
        return Pick((home if delta >= 0 else away), 0, round_f(delta))

    def predict_all(id, game, a_stats, h_stats):
        pick = predict(a_stats.team, h_stats.team)
        point_spread = spreads_avg[(a_stats.team, h_stats.team)]
        point_spread_m = spreads_med[(a_stats.team, h_stats.team)]
        return dict(
            id=id,
            game=game,
            point_spread=pick(point_spread.away.points, point_spread.home.points),
            point_spread_1=pick(point_spread_m.away.points, point_spread_m.home.points),
            simple_ranking=pick(a_stats.simple_ranking, h_stats.simple_ranking),
            simple_ranking_1=pick(a_stats.simple_ranking, h_stats.simple_ranking + home_field),
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
        [
            predict_all(n + 1, game, stats_for(game.away), stats_for(game.home))
            for (n, game) in enumerate(games)
        ],
        rank("simple_ranking"),
        rank("simple_ranking_1"),
        rank("point_spread"),
        rank("point_spread_1"),
        rank("pythagorean"),
        rank("margin_of_victory"),
        rank("wins"),
    )
    return [Picks(**picks) for picks in ranked]


get_html_from_url = compose(html.fromstring, lambda x: x.content, requests.get)


def score_scrape(weeks):
    def select_games(week_xml):
        return week_xml.xpath(
            '//div[starts-with(@class,"game_summary")]/table[@class="teams"]/tbody'
        )

    def parse_game(game_xml):
        teams = [
            x.text.split()[-1].lower()
            for x in game_xml.xpath('tr/td/a[starts-with(@href, "/teams/")]')
        ]
        scores = [x.text if x.text else -1 for x in game_xml.xpath('tr/td[@class="right"]')][0:2]
        return zip(teams, [int(scores[0]), int(scores[1])])

    def scrape_week(yr, week):
        parse = compose(map(parse_game), select_games, get_html_from_url)
        # print(f'https://www.pro-football-reference.com/years/{yr}/week_{week}.htm')
        game_info = parse(f"https://www.pro-football-reference.com/years/{yr}/week_{week}.htm")
        return [
            Game(Scored(*away), Scored(*home), int(yr), int(week)) for (away, home) in game_info
        ]

    ex = futures.ThreadPoolExecutor(max_workers=4)
    return sorted(
        ex.map(scrape_week, [x.year for x in weeks], [x.week for x in weeks]),
        key=lambda gms: gms[0].year * 100 + gms[0].week,
        reverse=True,
    )


def spread_scrape(yr, wk, odds=None):
    def get_odds_html(odds_file):
        if not odds_file:
            return get_html_from_url("https://www.oddsshark.com/nfl/odds")

        with open(odds_file, "r") as f:
            return html.fromstring(f.read())

    def parse_spreads(g):
        return [
            json.loads(x.attrib["data-op-info"])["fullgame"]
            for x in g.xpath('div/div/div[starts-with(@class, "op-item op-spread")]')
        ]

    def convert_spreads(diffs):
        return [0 if n == "Ev" else float(n) for n in diffs if not n == ""]

    lookup_nickname = map(lambda x: (TEAM_MAP[x[0]], TEAM_MAP[x[1]]))
    matchups = compose(lookup_nickname, partition(2))
    get_spreads = map(compose(convert_spreads, parse_spreads))
    get_away_spreads = compose(
        map(compose(list, lambda s: islice(s, 0, None, 2))),
        filter(lambda x: len(x) > 0),
        get_spreads,
    )
    calc_spreads_avg = compose(list, map(statistics.mean), get_away_spreads)
    calc_spreads_med = compose(list, map(statistics.median), get_away_spreads)

    def parse_odds_xml(p, calc):
        team_names = [
            str.lower(json.loads(x.attrib["data-op-name"])["short_name"])
            for x in p.xpath('//div[contains(@class, "op-matchup-team")]')
            if "data-op-name" in x.attrib
        ]
        games = p.xpath('//div[@id="op-results"]')[0].xpath(
            'div[starts-with(@class, "op-item-row-wrapper")]'
        )
        return zip(matchups(team_names), calc(games))

    html = get_odds_html(odds)
    return (
        spread_dict(yr, wk, list(parse_odds_xml(html, calc_spreads_avg))),
        spread_dict(yr, wk, list(parse_odds_xml(html, calc_spreads_med))),
    )


def spread2score(compare, spread):
    return round2(abs(spread) if compare(spread, 0) else 0)


def spread_dict(yr, wk, spreads):
    return {
        teams: Game(
            Scored(teams[0], spread2score(operator.lt, spread)),
            Scored(teams[1], spread2score(operator.ge, spread)),
            int(yr),
            int(wk),
        )
        for (teams, spread) in spreads
    }


def mock_spreads(yr, wk, games):
    result = spread_dict(yr, wk, [((g.away.team, g.home.team), 0) for g in games])
    return (result, result)


def parse_week(date):
    try:
        x = "".join([c for c in date if c.isdigit()])
        return Week(int(x[:4]), int(x[4:]))
    except:
        return Week(datetime.now().year, 0)


def get_weeks(s_date, e_date=None):
    def _sort(weeks):
        return sorted(weeks, key=lambda w: w.year * 100 + w.week, reverse=True)

    start = parse_week(s_date)
    if not e_date:
        return [start]

    end = parse_week(e_date)

    if end.year == start.year:
        return _sort([Week(end.year, n) for n in range(end.week, start.week + 1)])

    return _sort(
        concat(
            [
                [Week(end.year, n) for n in range(end.week, 18)],
                [
                    Week(m_year, m_week)
                    for m_week in range(1, 18)
                    for m_year in range(end.year + 1, start.year)
                ],
                [Week(start.year, n) for n in range(1, start.week + 1)],
            ]
        )
    )


def calc_standings(games):
    #print(calc_home_field(games))
    #return simple_ranking(get_team_stats(games))
    #return fancy_srs(get_team_stats(games))
    return simplify(get_team_stats(games))


@click.group()
def main():
    """Pro Football Pick 'Em Picker"""
    pass


@main.command()
@click.argument("start", type=str)
@click.option("--end", "-e", type=str, default=None, help="end week (i.e. 201706)")
@click.option("--output", "-o", default="CSV", help="output format (default: CSV)")
def get_results(start, end, output):
    """get game scores for a given season and week (i.e. 201809 or 2018:9)"""
    write_game_results(sys.stdout, score_scrape(get_weeks(start, end)))


@main.command()
@click.argument("records-file", type=click.File("r"))
@click.option("--start", "-s", type=str, default=None, help="start week (i.e. 201706)")
@click.option("--end", "-e", type=str, default=None, help="end week (i.e. 201706)")
@click.option("--output", "-o", default="CSV", help="output format (default: CSV)")
def standings(records_file, start, end, output):
    """calculate the standings for a given range of weeks (default is this season)"""
    write_summary(
        sys.stdout,
        calc_standings(get_games([] if not start else get_weeks(start, end), records_file)),
    )


@main.command()
@click.argument("records-file", type=click.File("r"))
@click.argument("pick-week", type=str)
@click.option("--end", "-e", type=str, default=None, help="end week (i.e. 201706)")
@click.option("--spread-html", default=None, help="HTML with OddsShark spreads")
@click.option("--no-spread", is_flag=True, help="don't rank spreads")
@click.option("--output", "-o", default="CSV", help="output format (default: CSV)")
@click.option("--exclude", "-x", default=None)
def make_picks(records_file, pick_week, end, spread_html, no_spread, output, exclude):
    """predict the winners of the specified weeks games based on numerous criteria"""
    to_pick = parse_week(pick_week)
    games = list(get_games([] if not end else get_weeks(pick_week, end), records_file))
    # make a set to dedup, sometimes there are mistakes on the website
    to_play = set(first(score_scrape([to_pick])))
    if exclude:
        excludes = set([tuple(s.split(':')) for s in exclude.split(',')])
        this_week = [game for game in to_play if (game.away.team, game.home.team) not in excludes]
    else:
        this_week = to_play

    write_predictions(
        sys.stdout,
        predict_winners(
            calc_standings(games),
            this_week,
            *spread_scrape(*to_pick) if not no_spread else mock_spreads(*to_pick, this_week),
            calc_home_field(games),
        ),
    )


@main.command()
@click.argument("pick-week", type=str)
def spreads(pick_week):
    averages, medians = spread_scrape(*parse_week(pick_week))
    result = []
    for k, v in averages.items():
        key = "away" if v.away.points > 0 else "home"
        winner = v._asdict()[key]
        result.append((winner.team, winner.points, medians[k]._asdict()[key].points, (v.away.team, v.home.team)))
    print(f"winner      median  average")
    print(f"---------------------------")
    for v in sorted(result, key=lambda a: (a[2], a[1]), reverse=True):
        winner, avg, median, matchup = v
        print(f"{winner:<12}{median:<7}{avg:5.2f}", matchup)


if __name__ == "__main__":
    main()
