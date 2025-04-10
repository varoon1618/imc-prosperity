from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import json
import statistics
from typing import Any
import pandas as pd
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()
class KalmanFilter:
    def __init__(self, initial_estimate, initial_error=2, process_noise=0.001, measurement_noise=4):
        self.x = initial_estimate          # Initial estimate
        self.P = initial_error             # Initial estimate covariance
        self.Q = process_noise             # Process (model) noise
        self.R = measurement_noise         # Measurement noise

    def update(self, measurement):
        # Prediction step (nothing changes here as we're assuming constant model)
        self.P += self.Q

        # Kalman Gain
        K = self.P / (self.P + self.R)

        # Update estimate with new measurement
        self.x += K * (measurement - self.x)

        # Update error covariance
        self.P = (1 - K) * self.P

        return self.x  # Return the updated prediction

class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""
        squidKF = KalmanFilter(initial_estimate=1969)
        squidPrices = [1971, 1971, 1971, 1969, 1970, 1971, 1973, 1972, 1973, 1972]

        kelpKF = KalmanFilter(initial_estimate=2000)
        for product in state.order_depths:
            own_trades: List[Trade] = state.own_trades.get(product, [])  # Avoid KeyError
            trades: List[Trade] = state.market_trades.get(product, [])  # Avoid KeyError
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)  # Avoid KeyError
            #acceptable_price = self.getBuyPrice(trades);  # Participant should calculate this value

            buyPrices = list(order_depth.buy_orders.keys())
            best_bid = max(buyPrices)
            sellPrices = list(order_depth.sell_orders.keys())
            buyOrders = order_depth.buy_orders
            sellOrders = order_depth.sell_orders
            best_ask = min(sellPrices)
            buy = best_bid + 1
            sell = best_ask - 1
            if product == 'RAINFOREST_RESIN':
                if position < 40 and position > -40:
                    if best_bid >=10001:
                        orders.append(Order('RAINFOREST_RESIN', best_bid, -buyOrders[best_bid]))
                        orders.append(Order('RAINFOREST_RESIN', best_bid+1, -5))
                    if best_ask <=9999:
                        orders.append(Order('RAINFOREST_RESIN', best_ask, -sellOrders[best_ask]))
                        orders.append(Order('RAINFOREST_RESIN', best_ask-1, 5))
                elif position>=40:
                    orders.append(Order('RAINFOREST_RESIN', sell, -10))
                    orders.append(Order('RAINFOREST_RESIN', sell+1, -10))
                elif position<=-40:
                    orders.append(Order('RAINFOREST_RESIN', buy, 10))
                    orders.append(Order('RAINFOREST_RESIN', buy-1, 10))

            if product == 'SQUID_INK':
                midPrice = self.calcMean(buyPrices, sellPrices)
                zscore,squidPrices = self.zScore(midPrice, squidPrices, 5)
                logger.print(f'squid ink mid price: {midPrice}')
                logger.print(f'squid ink zscore: {zscore}')
                buyorders = order_depth.buy_orders
                sellorders = order_depth.sell_orders
                if zscore < -10 and position <25:
                    orders.append(Order('SQUID_INK', best_ask, -sellorders[best_ask]))
                if zscore > 7 and position >-25:
                    orders.append(Order('SQUID_INK', best_bid, -buyorders[best_bid]))
                elif position >=25 or position <=-25:
                    if position < 0 and -2<zscore<2:
                        if abs(position) < abs(sellorders[best_ask]):
                            orders.append(Order('SQUID_INK',best_ask, abs(position)))
                        if abs(position) > abs(sellorders[best_ask]):
                            orders.append(Order('SQUID_INK', best_ask, -sellorders[best_ask]))
                            orders.append(Order('SQUID_INK', best_ask+1, (abs(position) - abs(sellorders[best_ask]))))
                        else:
                            orders.append(Order('SQUID_INK', best_ask, -sellorders[best_ask]))
                    elif position > 0 and -4<zscore<2:
                        if abs(position) < abs(buyorders[best_bid]):
                            orders.append(Order('SQUID_INK', best_bid, -abs(position)))
                        if abs(position) > abs(buyorders[best_bid]) > 0:
                            orders.append(Order('SQUID_INK', best_bid, -buyorders[best_bid]))
                            orders.append(Order('SQUID_INK', best_bid-1, -(abs(position) - abs(buyorders[best_bid]))))
                        else:
                            orders.append(Order('SQUID_INK', best_bid, -buyorders[best_bid]))
            if product == 'KELP':
                buyorders = order_depth.buy_orders
                sellorders = order_depth.sell_orders
                fairPrice = self.fairPriceKelp(order_depth)
                logger.print(f'kelp fair price: {fairPrice}')
                if fairPrice - best_ask >= 2:
                    orders.append(Order('KELP', best_ask,-sellorders[best_ask]))
                if fairPrice - best_bid <= -2:
                    orders.append(Order('KELP', best_bid,-buyorders[best_bid]))
                if position > 0:
                    orders.append(Order('KELP', fairPrice+1, -5))
                    orders.append(Order('KELP', fairPrice-2, -5))
                if position < 0:
                    orders.append(Order('KELP', fairPrice-1, 5))
                    orders.append(Order('KELP', fairPrice+2, 5))

            result[product] = orders
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    
    def fairPriceKelp(self, order_depth: OrderDepth) -> int:
        buyOrders = order_depth.buy_orders
        bidPrice = max(buyOrders, key=buyOrders.get)
        sellOrders = order_depth.sell_orders
        askPrcie = min(sellOrders, key=sellOrders.get)
        logger.print(f'bid price: {bidPrice}')
        logger.print(f'ask price: {askPrcie}')
        return(int((bidPrice + askPrcie)/2))

    def fillNa(self,list):
        padded_list = (list + [0]*3)[:3]
        return padded_list
    
    def calcMean(self,buyPrices,sellPrices):
        buyPrices.extend(sellPrices)
        mean = sum(buyPrices) / len(buyPrices)
        return int(mean)
    
    def zScore(self, new_price, prices, block_size):
        prices.append(new_price)

    # Don't compute z-score until we have at least one full block + 1 new point
        if len(prices) < block_size + 1:
            return None, prices

    # Determine which block to use (shift every time new group of 3 is complete)
        block_start = ((len(prices) - block_size - 1) // block_size) * block_size
        block = prices[block_start:block_start + block_size]

        mean = sum(block) / block_size
        std = pd.Series(block).std()

        z = (new_price - mean) / std if std != 0 else 0
        return z, prices


    def unrealized_pnl(self,trades: list[Trade], user_id: str, best_bid: int, best_ask: int):
        long_inventory = []   # (price, quantity)
        short_inventory = []  # (price, quantity)

        for trade in trades:
            if trade.buyer == user_id:
                # You bought (long)
                qty = trade.quantity
                # If covering short position
                while qty > 0 and short_inventory:
                    sell_price, sell_qty = short_inventory[0]
                    matched_qty = min(qty, sell_qty)
                    if matched_qty == sell_qty:
                        short_inventory.pop(0)
                    else:
                        short_inventory[0] = (sell_price, sell_qty - matched_qty)
                    qty -= matched_qty
            # Remaining qty goes into long inventory
                if qty > 0:
                    long_inventory.append((trade.price, qty))

            elif trade.seller == user_id:
                # You sold (short)
                qty = trade.quantity
                # If covering long position
                while qty > 0 and long_inventory:
                    buy_price, buy_qty = long_inventory[0]
                    matched_qty = min(qty, buy_qty)
                    if matched_qty == buy_qty:
                        long_inventory.pop(0)
                    else:
                        long_inventory[0] = (buy_price, buy_qty - matched_qty)
                    qty -= matched_qty
            # Remaining qty goes into short inventory
                if qty > 0:
                    short_inventory.append((trade.price, qty))

    # Compute unrealized PnL
        unrealized_long = sum((best_bid - price) * qty for price, qty in long_inventory)
        unrealized_short = sum((price - best_ask) * qty for price, qty in short_inventory)

        return unrealized_long + unrealized_short
 
