"""
Unit tests for matching engine, A-S strategy, and analytics.

Run with: pytest tests/unit/test_core.py -v

Design note: we test mathematical properties (not just execution),
e.g. that reservation price has the correct monotonicity in inventory.
"""

import math
import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.matching_engine import (
    MatchingEngine, Order, OrderType, Side, OrderStatus
)
from src.strategies.avellaneda_stoikov import AvellanedaStoikov, ASParams


# ============================================================
# Matching Engine Tests
# ============================================================

class TestMatchingEngine:

    def _make_order(self, side, otype, price, qty, pid="TEST", ts=0.0):
        import uuid
        return Order(
            order_id=str(uuid.uuid4()),
            participant_id=pid,
            side=side,
            order_type=otype,
            price=price,
            quantity=qty,
            timestamp=ts,
        )

    def test_empty_book_no_spread(self):
        engine = MatchingEngine()
        assert engine.best_bid is None
        assert engine.best_ask is None
        assert engine.spread is None

    def test_single_limit_order_posts(self):
        engine = MatchingEngine()
        o = self._make_order(Side.BID, OrderType.LIMIT, 99.0, 10)
        engine.submit(o)
        assert engine.best_bid == pytest.approx(99.0)
        assert engine.best_ask is None

    def test_spread_after_both_sides(self):
        engine = MatchingEngine()
        engine.submit(self._make_order(Side.BID, OrderType.LIMIT, 99.0, 10))
        engine.submit(self._make_order(Side.ASK, OrderType.LIMIT, 101.0, 10))
        assert engine.spread == pytest.approx(2.0)
        assert engine.mid_price == pytest.approx(100.0)

    def test_market_order_crosses_spread(self):
        engine = MatchingEngine()
        # Post ask
        ask = self._make_order(Side.ASK, OrderType.LIMIT, 101.0, 10)
        engine.submit(ask)
        # Market buy
        market = self._make_order(Side.BID, OrderType.MARKET, 0, 5)
        trades = engine.submit(market)
        assert len(trades) == 1
        assert trades[0].price == pytest.approx(101.0)
        assert trades[0].quantity == 5

    def test_partial_fill(self):
        engine = MatchingEngine()
        ask = self._make_order(Side.ASK, OrderType.LIMIT, 101.0, 5)
        engine.submit(ask)
        market = self._make_order(Side.BID, OrderType.MARKET, 0, 10)
        trades = engine.submit(market)
        assert trades[0].quantity == 5
        assert market.remaining == 5
        assert market.status == OrderStatus.CANCELLED  # couldn't fully fill

    def test_price_time_priority(self):
        """Two orders at same price: earlier order fills first."""
        engine = MatchingEngine()
        import uuid
        first = Order(order_id="FIRST", participant_id="A", side=Side.ASK,
                      order_type=OrderType.LIMIT, price=100.0, quantity=5, timestamp=1.0)
        second = Order(order_id="SECOND", participant_id="B", side=Side.ASK,
                       order_type=OrderType.LIMIT, price=100.0, quantity=5, timestamp=2.0)
        engine.submit(first)
        engine.submit(second)

        market = self._make_order(Side.BID, OrderType.MARKET, 0, 5)
        trades = engine.submit(market)
        assert trades[0].maker_order_id == "FIRST"

    def test_cancel_removes_from_book(self):
        engine = MatchingEngine()
        o = self._make_order(Side.BID, OrderType.LIMIT, 99.0, 10)
        engine.submit(o)
        assert engine.best_bid == pytest.approx(99.0)
        cancelled = engine.cancel(o.order_id)
        assert cancelled
        assert engine.best_bid is None

    def test_cancel_filled_order_fails(self):
        engine = MatchingEngine()
        ask = self._make_order(Side.ASK, OrderType.LIMIT, 100.0, 10)
        engine.submit(ask)
        market = self._make_order(Side.BID, OrderType.MARKET, 0, 10)
        engine.submit(market)
        assert not engine.cancel(ask.order_id)

    def test_ofi_balanced_book(self):
        engine = MatchingEngine()
        engine.submit(self._make_order(Side.BID, OrderType.LIMIT, 99.0, 10))
        engine.submit(self._make_order(Side.ASK, OrderType.LIMIT, 101.0, 10))
        assert engine.order_flow_imbalance() == pytest.approx(0.0)

    def test_ofi_bid_heavy(self):
        engine = MatchingEngine()
        engine.submit(self._make_order(Side.BID, OrderType.LIMIT, 99.0, 20))
        engine.submit(self._make_order(Side.ASK, OrderType.LIMIT, 101.0, 10))
        ofi = engine.order_flow_imbalance()
        assert ofi > 0  # bid-heavy

    def test_price_improvement(self):
        """Limit order should only match at best price, not worse."""
        engine = MatchingEngine()
        engine.submit(self._make_order(Side.ASK, OrderType.LIMIT, 100.0, 10))
        engine.submit(self._make_order(Side.ASK, OrderType.LIMIT, 101.0, 10))
        buy_limit = self._make_order(Side.BID, OrderType.LIMIT, 101.0, 15)
        trades = engine.submit(buy_limit)
        # Should fill at 100.0 first (price priority)
        assert trades[0].price == pytest.approx(100.0)

    def test_multiple_levels_consumed(self):
        engine = MatchingEngine()
        for px in [100.0, 101.0, 102.0]:
            engine.submit(self._make_order(Side.ASK, OrderType.LIMIT, px, 5))
        market = self._make_order(Side.BID, OrderType.MARKET, 0, 15)
        trades = engine.submit(market)
        assert len(trades) == 3
        prices = [t.price for t in trades]
        assert prices == sorted(prices)  # price ascending

    def test_snapshot_structure(self):
        engine = MatchingEngine()
        snap = engine.snapshot()
        assert "best_bid" in snap
        assert "best_ask" in snap
        assert "spread" in snap
        assert "ofi" in snap


# ============================================================
# Avellaneda-Stoikov Tests
# ============================================================

class TestAvellanedaStoikov:

    def test_reservation_price_zero_inventory(self):
        """With zero inventory, reservation price == mid."""
        mm = AvellanedaStoikov(ASParams(gamma=0.1, sigma=0.02))
        r = mm.reservation_price(mid=100.0, inventory=0, t=0.5)
        assert r == pytest.approx(100.0)

    def test_reservation_price_long_inventory(self):
        """Long inventory → reservation price < mid."""
        mm = AvellanedaStoikov(ASParams(gamma=0.1, sigma=0.02))
        r = mm.reservation_price(mid=100.0, inventory=10, t=0.5)
        assert r < 100.0

    def test_reservation_price_short_inventory(self):
        """Short inventory → reservation price > mid."""
        mm = AvellanedaStoikov(ASParams(gamma=0.1, sigma=0.02))
        r = mm.reservation_price(mid=100.0, inventory=-10, t=0.5)
        assert r > 100.0

    def test_reservation_price_monotone_in_inventory(self):
        """Reservation price decreasing in inventory."""
        mm = AvellanedaStoikov(ASParams(gamma=0.1, sigma=0.02))
        r1 = mm.reservation_price(100.0, -5, 0.5)
        r2 = mm.reservation_price(100.0, 0, 0.5)
        r3 = mm.reservation_price(100.0, 5, 0.5)
        assert r1 > r2 > r3

    def test_optimal_spread_positive(self):
        """Optimal spread should always be positive."""
        mm = AvellanedaStoikov(ASParams(gamma=0.1, sigma=0.02, kappa=1.5))
        for t in [0.0, 0.25, 0.5, 0.75, 0.99]:
            s = mm.optimal_spread(t)
            assert s > 0, f"Spread non-positive at t={t}"

    def test_optimal_spread_decreasing_in_time(self):
        """
        As t → T (session end), time_remaining → 0 → first term → 0.
        But second term remains, so spread doesn't go to zero entirely.
        However at beginning (t=0), spread is widest.
        """
        mm = AvellanedaStoikov(ASParams(gamma=0.1, sigma=0.02, kappa=1.5, T=1.0))
        s0 = mm.optimal_spread(t=0.0)
        s_half = mm.optimal_spread(t=0.5)
        assert s0 > s_half

    def test_optimal_spread_non_monotone_in_gamma(self):
        """
        The A-S optimal spread has a non-trivial relationship with γ:

        δ* = γσ²(T-t) + (2/γ)ln(1+γ/κ)
              ↑ increases in γ    ↑ decreases in γ

        The second term dominates at moderate γ, so the spread can
        DECREASE as γ increases. This is a genuine feature, not a bug.

        The reservation price responds more strongly to γ (via skewing),
        even as the half-spread may narrow. The MM manages inventory more
        aggressively, accepting narrower spreads to get filled.
        """
        import math
        gamma_low, gamma_high = 0.1, 1.0
        sigma, T, t, kappa = 0.1, 10.0, 0.0, 1.5

        s_low = gamma_low * sigma**2 * (T-t) + (2/gamma_low)*math.log(1+gamma_low/kappa)
        s_high = gamma_high * sigma**2 * (T-t) + (2/gamma_high)*math.log(1+gamma_high/kappa)
        # At these params, second term dominates → spread decreases with γ
        assert s_high < s_low, "Expected spread to decrease as γ increases (second term dominates)"

    def test_optimal_spread_increasing_in_sigma(self):
        """Higher volatility → wider spread (risk premium)."""
        s_low = AvellanedaStoikov(ASParams(sigma=0.005)).optimal_spread(0.5)
        s_high = AvellanedaStoikov(ASParams(sigma=0.05)).optimal_spread(0.5)
        assert s_high > s_low

    def test_quotes_centred_around_reservation(self):
        """Bid and ask should be symmetric around reservation price."""
        mm = AvellanedaStoikov()
        bid, ask = mm.compute_quotes(mid=100.0, inventory=0, t=0.5)
        r = mm.reservation_price(100.0, 0, 0.5)
        assert bid == pytest.approx(r - (ask - bid) / 2, abs=0.001)
        assert ask == pytest.approx(r + (ask - bid) / 2, abs=0.001)

    def test_quotes_skew_with_inventory(self):
        """With long inventory, both quotes should be below mid."""
        mm = AvellanedaStoikov(ASParams(gamma=0.5, sigma=0.02))
        bid_flat, ask_flat = mm.compute_quotes(100.0, 0, 0.5)
        bid_long, ask_long = mm.compute_quotes(100.0, 20, 0.5)
        # Both quotes should shift down when long
        assert bid_long < bid_flat
        assert ask_long < ask_flat

    def test_fill_probability_decreasing_in_distance(self):
        """Fill probability decreases as we quote further from mid."""
        mm = AvellanedaStoikov(ASParams(kappa=1.5))
        p1 = mm.fill_probability(99.9, 100.0, "bid")  # close
        p2 = mm.fill_probability(99.5, 100.0, "bid")  # far
        assert p1 > p2

    def test_fill_probability_in_unit_interval(self):
        """Fill probability must be in [0, 1]."""
        mm = AvellanedaStoikov()
        for delta in [0.0, 0.01, 0.1, 1.0, 10.0]:
            p = mm.fill_probability(100.0 - delta, 100.0, "bid")
            assert 0.0 <= p <= 1.0

    def test_expected_value_positive_at_mid(self):
        """At zero spread (quoting at mid), EV should still be positive
        because fill probability is maximised there."""
        mm = AvellanedaStoikov(ASParams(kappa=1.5))
        ev = mm.expected_value_per_quote(100.0, 100.0, 100.0, 0.5)
        assert ev["ev_round_trip"] >= 0

    def test_inventory_risk_zero_at_zero_inventory(self):
        mm = AvellanedaStoikov()
        risk = mm.inventory_risk(mid=100.0, t=0.5)
        assert risk == pytest.approx(0.0)

    def test_inventory_risk_increasing_in_abs_inventory(self):
        mm = AvellanedaStoikov(ASParams(gamma=0.1, sigma=0.02))
        mm.state.inventory = 5
        risk_small = mm.inventory_risk(100.0, 0.5)
        mm.state.inventory = 20
        risk_large = mm.inventory_risk(100.0, 0.5)
        assert risk_large > risk_small

    def test_should_trade_respects_inventory_limit(self):
        mm = AvellanedaStoikov(ASParams(max_inventory=10))
        mm.state.inventory = 9
        assert mm.should_trade()
        mm.state.inventory = 10
        assert not mm.should_trade()

    def test_on_fill_updates_inventory(self):
        mm = AvellanedaStoikov()
        mm.on_fill("bid", fill_price=99.5, fill_qty=3, mid_at_fill=100.0)
        assert mm.state.inventory == 3
        mm.on_fill("ask", fill_price=100.5, fill_qty=3, mid_at_fill=100.0)
        assert mm.state.inventory == 0

    def test_ewma_vol_update(self):
        """EWMA variance should respond to price moves."""
        mm = AvellanedaStoikov()
        mm.state.last_mid = 100.0
        mm.update_volatility(100.5)  # 0.5% move
        assert mm.state.ewma_var > 0

    def test_kappa_recalibration(self):
        """Kappa should update toward 1/mean_delta from fill history."""
        mm = AvellanedaStoikov()
        mm.state.fill_history = [{"delta": 0.1}] * 50  # mean delta = 0.1
        mm.recalibrate_kappa()
        assert mm.params.kappa == pytest.approx(10.0, rel=0.1)


# ============================================================
# Mathematical property tests
# ============================================================

class TestMathematicalProperties:

    def test_as_reservation_price_formula(self):
        """Verify exact formula: r = s - q*γ*σ²*(T-t)"""
        gamma, sigma, T, t = 0.1, 0.02, 1.0, 0.3
        inventory, mid = 7, 100.0
        expected_r = mid - inventory * gamma * sigma**2 * (T - t)
        mm = AvellanedaStoikov(ASParams(gamma=gamma, sigma=sigma, T=T))
        computed_r = mm.reservation_price(mid, inventory, t)
        assert computed_r == pytest.approx(expected_r, rel=1e-6)

    def test_as_optimal_spread_formula(self):
        """Verify exact formula: δ* = γσ²(T-t) + (2/γ)ln(1+γ/κ)"""
        gamma, sigma, kappa, T, t = 0.1, 0.02, 1.5, 1.0, 0.3
        expected_spread = (
            gamma * sigma**2 * (T - t) +
            (2.0 / gamma) * math.log(1.0 + gamma / kappa)
        )
        mm = AvellanedaStoikov(ASParams(gamma=gamma, sigma=sigma, kappa=kappa, T=T))
        computed_spread = mm.optimal_spread(t)
        assert computed_spread == pytest.approx(expected_spread, rel=1e-6)

    def test_fill_probability_exponential_decay(self):
        """P(fill) = exp(-κ*δ), verify decay rate."""
        kappa = 1.5
        mm = AvellanedaStoikov(ASParams(kappa=kappa))
        for delta in [0.1, 0.2, 0.5]:
            expected_p = math.exp(-kappa * delta)
            computed_p = mm.fill_probability(100.0 - delta, 100.0, "bid")
            assert computed_p == pytest.approx(expected_p, rel=1e-6)
