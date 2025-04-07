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


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        for product in state.order_depths:

            trades: List[Trade] = state.market_trades.get(product, [])  # Avoid KeyError
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)  # Avoid KeyError
            #acceptable_price = self.getBuyPrice(trades);  # Participant should calculate this value

            buyPrices = list(order_depth.buy_orders.keys())
            best_bid = max(buyPrices)
            sellPrices = list(order_depth.sell_orders.keys())
            best_ask = min(sellPrices)
            buy = best_bid + 1
            sell = best_ask - 1
            kelpBids = [4997]
            kelpAsks = [5003]
            kelpBidVolumes = [26]
            kelpAskVolumes = [26]
            if product == 'RAINFOREST_RESIN':
                if position < 40 and position > -40:
                    if best_bid >=10002:
                        orders.append(Order('RAINFOREST_RESIN', best_bid, -5))
                        orders.append(Order('RAINFOREST_RESIN', best_bid+1, -5))
                    if best_ask <=9998:
                        orders.append(Order('RAINFOREST_RESIN', best_ask, 5))
                        orders.append(Order('RAINFOREST_RESIN', best_ask-1, 5))
                elif position>=40:
                    orders.append(Order('RAINFOREST_RESIN', sell, -5))
                    orders.append(Order('RAINFOREST_RESIN', sell+1, -5))
                elif position<=-40:
                    orders.append(Order('RAINFOREST_RESIN', buy, 10))
                    orders.append(Order('RAINFOREST_RESIN', buy-1, 10))
            if product == 'KELP':
                pred = self.predictKelp(order_depth,kelpBids[-1], kelpBidVolumes[-1], kelpAsks[-1], kelpAskVolumes[-1])
                logger.print("Predicted Kelp Price:", pred)
                kelpBids.append(best_bid)
                kelpBidVolumes.append(order_depth.buy_orders[best_bid])
                kelpAsks.append(best_ask)
                kelpAskVolumes.append(order_depth.sell_orders[best_ask])
                if position < 30 and position > -30:
                    if pred-best_bid<=-2:
                        orders.append(Order('KELP', best_bid, -5))
                    if pred-best_ask >= 2:
                        orders.append(Order('KELP', best_ask, 5))
                elif position<=-30:
                    orders.append(Order('KELP', best_ask, 10))
                elif position>=30:
                    orders.append(Order('KELP', best_bid, -10))
    

            result[product] = orders
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    



    def predictKelp(self,order_depth: OrderDepth,prev_bid1,prev_bid_vol,prev_ask1,prev_ask_vol) -> float:
        bidPrices = list(order_depth.buy_orders.keys())
        bidVolumes = list(order_depth.buy_orders.values())
        askPrices = list(order_depth.sell_orders.keys())
        askVolumes = list(order_depth.sell_orders.values())
        askPrices = self.fillNa(askPrices)
        bidPrices = self.fillNa(bidPrices)
        bidVolumes = self.fillNa(bidVolumes)
        askVolumes = self.fillNa(askVolumes)

        coeff = {'ask_price_1':0.279515,
                 'bid_price_1':0.263735,
                 'prev1_ask_price': 0.235733,
                 'prev1_bid_price':0.218569,
                 'ask_volume_1':-0.023115,
                 'bid_volume_1':0.022885,
                 'prev1_bid_volume':0.019662,
                 'prev1_ask_volume':-0.017221}
        
        pred = coeff['bid_price_1']*max(bidPrices) + coeff['ask_price_1']*min(askPrices) + coeff['prev1_bid_price']*prev_bid1 + coeff['prev1_ask_price']*prev_ask1 + coeff['bid_volume_1']*bidVolumes[0] + coeff['ask_volume_1']*askVolumes[0] + coeff['prev1_bid_volume']*prev_bid_vol + coeff['prev1_ask_volume']*prev_ask_vol
        return pred 
    
    def fillNa(self,list):
        padded_list = (list + [0]*3)[:3]
        return padded_list
    
    def calcMeanPrice(self,order_depth: OrderDepth) -> float:
        prices = list(order_depth.buy_orders.keys()).extend(list(order_depth.sell_orders.keys()))
        currentMean = statistics.mean(prices) if prices else 0
        return currentMean
    