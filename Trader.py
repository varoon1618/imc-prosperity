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
                pred = self.predictKelp(order_depth)
                logger.print(f'Kelp prediction: {pred}')
                if position < 15 and position > -15:
                    if pred-best_ask>=1:
                        orders.append(Order('KELP', best_ask, 5))
                        orders.append(Order('KELP', best_ask-1, 5))
                    if pred-best_bid<=-1:
                        orders.append(Order('KELP', best_bid, -5))
                        orders.append(Order('KELP', best_bid+1, -5))
                elif position>=15:
                    orders.append(Order('KELP', sell, -5))
                    orders.append(Order('KELP', sell+1, -5))
                elif position<=-15:
                    orders.append(Order('KELP', buy, 10))
                    orders.append(Order('KELP', buy-1, 10))
            result[product] = orders
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    
    def predictKelp(self,order_depth: OrderDepth):
        bidPrices = list(order_depth.buy_orders.keys())
        bidVolumes = list(order_depth.buy_orders.values())
        askPrices = list(order_depth.sell_orders.keys())
        askVolumes = list(order_depth.sell_orders.values())
        bidPrices = self.fillNa(bidPrices)
        askPrices = self.fillNa(askPrices)
        bidVolumes = self.fillNa(bidVolumes)    
        askVolumes = self.fillNa(askVolumes)

        coeff = {'bid_1':0.504144,'ask_1':0.491322,'bid_volume_2':-0.088549,'ask_volume_2':0.067670,
                'ask_volume_1':0.056324,
                'bid_volume_1':-0.030049,
                'bid_volume_3':0.006537,
                'ask_volume_3':-0.001741,
                'bid_price_3':-0.000760,
                'ask_price_3':0.000418,
                'bid_price_2':0.000147,
                'ask_price_2':0.000091}
        
        pred = coeff['bid_1']*bidPrices[0] + coeff['ask_1']*askPrices[0] + coeff['bid_volume_2']*bidVolumes[1] + coeff['ask_volume_2']*askVolumes[1] + coeff['ask_volume_1']*askVolumes[0] + coeff['bid_volume_1']*bidVolumes[0] + coeff['bid_volume_3']*bidVolumes[2] + coeff['ask_volume_3']*askVolumes[2] + coeff['bid_price_3']*bidPrices[2] + coeff['ask_price_3']*askPrices[2] + coeff['bid_price_2']*bidPrices[1] + coeff['ask_price_2']*askPrices[1]
        return pred 
    
    def fillNa(self,list):
        padded_list = (list + [0]*3)[:3]
        return padded_list


    def tradeResin(self,order_depth: OrderDepth):
        resinOrders = []
        previousPrices = [10,000]
        buyPrices = list(order_depth.buy_orders.keys())
        best_bid = max(buyPrices)
        sellPrices = list(order_depth.sell_orders.keys())
        best_ask = min(sellPrices)
        historicMean = statistics.mean(previousPrices)
        allprices = []
        allprices.extend(buyPrices)
        allprices.extend(sellPrices)
        currentMean = statistics.mean(allprices)
        previousPrices.append(currentMean)
        if currentMean - historicMean >=2:
            if(best_bid > historicMean):
                logger.print(f' RESIN sell price : {best_bid}')
                resinOrders.append(Order('RAINFOREST_RESIN',best_bid,-10))
            else:
                logger.print(f' RESIN sell price : {best_bid+1}')
                resinOrders.append(Order('RAINFOREST_RESIN',best_bid+1,-10))
        if currentMean - historicMean <=-2:
            if(best_ask < historicMean):
                logger.print(f' RESIN buy price : {best_ask}')
                resinOrders.append(Order('RAINFOREST_RESIN',best_ask,10))
            else:
                logger.print(f' RESIN buy price : {best_ask-1}')
                resinOrders.append(Order('RAINFOREST_RESIN',best_ask-1,10))
        return resinOrders
    

    def calcMeanPrice(self,order_depth: OrderDepth) -> float:
        prices = list(order_depth.buy_orders.keys()).extend(list(order_depth.sell_orders.keys()))
        currentMean = statistics.mean(prices) if prices else 10,000
        return currentMean


