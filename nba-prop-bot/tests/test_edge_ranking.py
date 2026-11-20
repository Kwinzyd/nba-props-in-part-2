from src.models.edge_ranker import rank_edges


def _candidate(player='Player A', market='player_points', side='OVER',
               model_prob=0.60, implied_prob=0.50, odds=2.0,
               proj_minutes=28.0, mean=22.0, line=20.5,
               injury_status='Healthy', variance_scale=1.0,
               steam_flag=False, velocity=0.0, dispersion=0.02, book_role='rec', **extra):
    return dict(
        player_id=player, market=market, side=side,
        model_prob=model_prob, implied_prob=implied_prob, odds=odds,
        projected_minutes=proj_minutes, mean=mean, line=line,
        injury_status=injury_status, variance_scale=variance_scale,
        steam_flag=steam_flag, velocity=velocity, dispersion=dispersion,
        book_role=book_role, book='draftkings', **extra,
    )


def test_returns_sorted_by_risk_adjusted_ev():
    low  = _candidate(model_prob=0.52, player='Low Edge')
    high = _candidate(model_prob=0.65, player='High Edge')
    ranked = rank_edges([low, high])
    assert ranked[0]['player_id'] == 'High Edge', "Higher edge should rank first"


def test_out_player_excluded():
    c = _candidate(injury_status='Out')
    ranked = rank_edges([c])
    assert len(ranked) == 0, "OUT player should be excluded from ranking"


def test_low_minutes_excluded():
    c = _candidate(proj_minutes=10.0)  # below MIN_PROJECTED_MINUTES=15
    ranked = rank_edges([c])
    assert len(ranked) == 0, "Low-minutes player should be filtered out"


def test_doubtful_reduces_edge():
    healthy  = _candidate(injury_status='Healthy',  player='Healthy')
    doubtful = _candidate(injury_status='Doubtful', player='Doubtful')
    ranked = rank_edges([healthy, doubtful])
    h_edge = next(r for r in ranked if r['player_id'] == 'Healthy')['edge']
    d_edge = next(r for r in ranked if r['player_id'] == 'Doubtful')['edge']
    assert d_edge < h_edge, "Doubtful should have smaller edge than Healthy"


def test_empty_candidates_returns_empty():
    assert rank_edges([]) == []


def test_ranked_by_edge_vs_breakeven_not_vig():
    # Same model prob; the high-hold market's devigged implied is lower, so its
    # `edge` looks bigger, but it pays a worse price (lower odds).
    high_hold = _candidate(player='Steals 0.5', market='player_steals', mean=0.9, line=0.5,
                           model_prob=0.60, implied_prob=0.50, odds=1.75)
    fair_line = _candidate(player='Points', model_prob=0.60, implied_prob=0.52, odds=1.91)
    ranked = rank_edges([high_hold, fair_line])
    assert ranked[0]['player_id'] == 'Points'
    assert ranked[1]['edge'] < ranked[0]['edge']      # floors use the price paid
    assert ranked[1]['fair_market_edge'] > ranked[0]['fair_market_edge']


def test_low_mean_props_not_favoured():
    low_mean = _candidate(player='Blocks', market='player_blocks', mean=1.0, line=0.5)
    high_mean = _candidate(player='Points', mean=22.0, model_prob=0.61)
    assert rank_edges([low_mean, high_mean])[0]['player_id'] == 'Points'


def test_sharp_consensus_veto():
    # consensus_prob is the devigged sharp P(over)
    agrees    = _candidate(player='Agrees', side='UNDER', odds=1.91, consensus_prob=0.45)
    disagrees = _candidate(player='Disagrees', side='UNDER', odds=1.91, consensus_prob=0.50)
    ranked = rank_edges([agrees, disagrees])
    assert [r['player_id'] for r in ranked] == ['Agrees']
    assert ranked[0]['sharp_fair_prob'] == 0.55


def test_boosters_removed():
    plain = _candidate(player='Plain')
    boosted = _candidate(player='Boosted', steam_flag=True, book_role='sharp', dispersion=0.05)
    boosted['timestamp_stale'] = True
    r = {x['player_id']: x['edge'] for x in rank_edges([plain, boosted])}
    assert r['Plain'] == r['Boosted']
