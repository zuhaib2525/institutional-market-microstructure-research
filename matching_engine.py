"""
Matching Engine — price-time priority, supports limit/market orders,
partial fills, cancellations, and trade generation.

Design notes (written after a few iterations):
- Using SortedDict from sortedcontainers is ~10x faster than a plain dict +
  sorting on each tick.  Worth the dependency.
- We do NOT use a lock here; the sim is single-threaded. A real engine would
  need a separate thread per side or a lock-free structure.
- fill_probability is tracked per queue slot, not per order, to match how
  real MM systems think about queue position.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

from sortedcontainers import SortedDict


class Side(Enum):
    BID = auto()
    ASK = auto()


class OrderType(Enum):
    LIMIT = auto()
    MARKET = auto()
    CANCEL = auto()


class OrderStatus(Enum):
    PENDING = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()


@dataclass
class Order:
    order_id: str
    participant_id: str
    side: Side
    order_type: OrderType
    price: float          # ignored for market orders
    quantity: int
    timestamp: float      # exchange time in seconds
    remaining: int = field(init=False)
    status: OrderStatus = field(init=False, default=OrderStatus.PENDING)
    queue_position: int = field(init=False, default=0)  # set by LOB on insert

    def __post_init__(self) -> None:
        self.remaining = self.quantity


@dataclass
class Trade:
    trade_id: str
    timestamp: float
    price: float
    quantity: int
    aggressor_side: Side
    maker_order_id: str
    taker_order_id: str
    maker_participant_id: str
    taker_participant_id: str


@dataclass
class FillReport:
    order_id: str
    participant_id: str
    fill_price: float
    fill_qty: int
    remaining_qty: int
    is_maker: bool
    timestamp: float


class PriceLevel:
    """
    Represents a single price level in the order book.
    Orders are stored in FIFO order (price-time priority).
    """

    __slots__ = ("price", "orders", "total_volume")

    def __init__(self, price: float) -> None:
        self.price: float = price
        self.orders: Deque[Order] = deque()
        self.total_volume: int = 0

    def add_order(self, order: Order) -> None:
        order.queue_position = len(self.orders)
        self.orders.append(order)
        self.total_volume += order.remaining

    def remove_order(self, order_id: str) -> Optional[Order]:
        for i, o in enumerate(self.orders):
            if o.order_id == order_id:
                self.total_volume -= o.remaining
                del self.orders[i]   # O(n) but levels are typically small
                return o
        return None

    @property
    def is_empty(self) -> bool:
        return len(self.orders) == 0

    @property
    def best_order(self) -> Optional[Order]:
        return self.orders[0] if self.orders else None


class MatchingEngine:
    """
    Central limit order book matching engine.

    Bid side: descending price (highest bid first)
    Ask side: ascending price (lowest ask first)

    Attributes
    ----------
    bids : SortedDict  price → PriceLevel, sorted ascending (we negate key)
    asks : SortedDict  price → PriceLevel, sorted ascending
    """

    def __init__(self) -> None:
        # bids: we store negative price as key so highest bid comes first
        self._bids: SortedDict = SortedDict()   # -price → PriceLevel
        self._asks: SortedDict = SortedDict()   # +price → PriceLevel
        self._orders: Dict[str, Order] = {}     # order_id → Order
        self.trades: List[Trade] = []
        self.fill_reports: List[FillReport] = []
        self._sequence: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, order: Order) -> List[Trade]:
        """Submit an order; returns list of trades generated."""
        if order.order_type == OrderType.CANCEL:
            self._cancel(order.order_id)
            return []
        self._orders[order.order_id] = order
        if order.order_type == OrderType.MARKET:
            return self._match_market(order)
        else:
            return self._match_limit(order)

    def cancel(self, order_id: str) -> bool:
        return self._cancel(order_id)

    @property
    def best_bid(self) -> Optional[float]:
        if not self._bids:
            return None
        neg_price = self._bids.keys()[0]
        return -neg_price

    @property
    def best_ask(self) -> Optional[float]:
        if not self._asks:
            return None
        return self._asks.keys()[0]

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def bid_depth(self, n_levels: int = 5) -> List[Tuple[float, int]]:
        result = []
        for neg_p, level in self._bids.items():
            result.append((-neg_p, level.total_volume))
            if len(result) >= n_levels:
                break
        return result

    def ask_depth(self, n_levels: int = 5) -> List[Tuple[float, int]]:
        result = []
        for p, level in self._asks.items():
            result.append((p, level.total_volume))
            if len(result) >= n_levels:
                break
        return result

    def order_flow_imbalance(self, n_levels: int = 1) -> float:
        """
        OFI = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        Ranges in [-1, 1].  Positive = more buy pressure.
        """
        bid_vol = sum(v for _, v in self.bid_depth(n_levels))
        ask_vol = sum(v for _, v in self.ask_depth(n_levels))
        denom = bid_vol + ask_vol
        if denom == 0:
            return 0.0
        return (bid_vol - ask_vol) / denom

    def queue_position(self, order_id: str) -> Optional[int]:
        order = self._orders.get(order_id)
        if order is None or order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return None
        return order.queue_position

    def volume_ahead(self, order_id: str) -> Optional[int]:
        """Volume ahead of this order in its queue."""
        order = self._orders.get(order_id)
        if order is None:
            return None
        level = self._get_level(order.side, order.price)
        if level is None:
            return None
        vol = 0
        for o in level.orders:
            if o.order_id == order_id:
                break
            vol += o.remaining
        return vol

    # ------------------------------------------------------------------
    # Internal matching logic
    # ------------------------------------------------------------------

    def _match_market(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        opposite = self._asks if order.side == Side.BID else self._bids
        sign = 1 if order.side == Side.ASK else -1   # sign for key lookup

        while order.remaining > 0 and opposite:
            best_key = opposite.keys()[0]
            best_level = opposite[best_key]
            trades.extend(self._fill_against_level(order, best_level, sign * best_key))
            if best_level.is_empty:
                del opposite[best_key]

        if order.remaining > 0:
            order.status = OrderStatus.CANCELLED  # market order not fully filled

        return trades

    def _match_limit(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        opposite = self._asks if order.side == Side.BID else self._bids
        sign = 1 if order.side == Side.ASK else -1

        while order.remaining > 0 and opposite:
            best_key = opposite.keys()[0]
            best_price = sign * best_key
            # Check price condition
            if order.side == Side.BID and best_price > order.price:
                break
            if order.side == Side.ASK and best_price < order.price:
                break
            best_level = opposite[best_key]
            trades.extend(self._fill_against_level(order, best_level, best_price))
            if best_level.is_empty:
                del opposite[best_key]

        if order.remaining > 0 and order.status != OrderStatus.FILLED:
            self._post_order(order)

        return trades

    def _fill_against_level(
        self, aggressor: Order, level: PriceLevel, price: float
    ) -> List[Trade]:
        price = abs(price)   # always positive; sign was from SortedDict key trick
        trades: List[Trade] = []
        while aggressor.remaining > 0 and not level.is_empty:
            maker = level.best_order
            fill_qty = min(aggressor.remaining, maker.remaining)
            self._sequence += 1
            trade = Trade(
                trade_id=f"T{self._sequence:08d}",
                timestamp=aggressor.timestamp,
                price=price,
                quantity=fill_qty,
                aggressor_side=aggressor.side,
                maker_order_id=maker.order_id,
                taker_order_id=aggressor.order_id,
                maker_participant_id=maker.participant_id,
                taker_participant_id=aggressor.participant_id,
            )
            self.trades.append(trade)
            trades.append(trade)

            # Update quantities
            aggressor.remaining -= fill_qty
            maker.remaining -= fill_qty
            level.total_volume -= fill_qty

            # Fill reports
            self.fill_reports.append(FillReport(
                order_id=aggressor.order_id,
                participant_id=aggressor.participant_id,
                fill_price=price,
                fill_qty=fill_qty,
                remaining_qty=aggressor.remaining,
                is_maker=False,
                timestamp=aggressor.timestamp,
            ))
            self.fill_reports.append(FillReport(
                order_id=maker.order_id,
                participant_id=maker.participant_id,
                fill_price=price,
                fill_qty=fill_qty,
                remaining_qty=maker.remaining,
                is_maker=True,
                timestamp=aggressor.timestamp,
            ))

            if maker.remaining == 0:
                maker.status = OrderStatus.FILLED
                level.orders.popleft()
            else:
                maker.status = OrderStatus.PARTIAL

            aggressor.status = (
                OrderStatus.FILLED if aggressor.remaining == 0 else OrderStatus.PARTIAL
            )

        return trades

    def _post_order(self, order: Order) -> None:
        if order.side == Side.BID:
            key = -order.price
            if key not in self._bids:
                self._bids[key] = PriceLevel(order.price)
            self._bids[key].add_order(order)
        else:
            if order.price not in self._asks:
                self._asks[order.price] = PriceLevel(order.price)
            self._asks[order.price].add_order(order)

    def _cancel(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return False
        level = self._get_level(order.side, order.price)
        if level:
            level.remove_order(order_id)
            if level.is_empty:
                self._remove_level(order.side, order.price)
        order.status = OrderStatus.CANCELLED
        return True

    def _get_level(self, side: Side, price: float) -> Optional[PriceLevel]:
        if side == Side.BID:
            return self._bids.get(-price)
        return self._asks.get(price)

    def _remove_level(self, side: Side, price: float) -> None:
        if side == Side.BID:
            self._bids.pop(-price, None)
        else:
            self._asks.pop(price, None)

    def snapshot(self) -> Dict:
        return {
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "mid_price": self.mid_price,
            "bid_depth": self.bid_depth(10),
            "ask_depth": self.ask_depth(10),
            "ofi": self.order_flow_imbalance(1),
            "total_trades": len(self.trades),
        }
