import logging
import gym                                      # RL Environment Framework
import numpy as np                              # For Mathematical Operations on Data
import pandas as pd                             # Data Processing
from gym import spaces                          # Discrete, Box, etc.
from gym.utils import seeding                   # Set seed to get reproduce same results.
from sklearn.preprocessing import MinMaxScaler  # Normalizing
from sklearn.model_selection import train_test_split

# Writing information into log files. Showing infos when Creating Environment.
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)

class DataSource:
    """
    Data source for TradingEnvironment:
    Loads & preprocesses daily price (close, low, high)
    """
    def __init__(self, trading_days, model, ticker='SPX', normalize=True):
        self.model = model
        self.ticker = ticker
        self.training = True
        self.trading_days = trading_days
        self.normalize = normalize
        self.data = self.load_data(self.ticker)
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess_data(self.model)
        self.counter = 0
        self.step = 0
        self.offset = None
        self.target = None
        self.date = None

    def load_data(self, ticker):

        log.info('loading data for {}...'.format(ticker))
        df = pd.read_csv(f"data/{ticker}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df.set_index(["date"], inplace=True)
        df = df.drop(["ticker", 'high', 'low'], axis=1).sort_index()
        log.info('got data for {}...'.format(ticker))

        return df

    def preprocess_data(self, model):

        if model == 0:
            spx = self.data.copy()
            spx['target'] = spx.close.pct_change()  # Target
            spx['ret_1'] = spx.close.pct_change()  # 1 day Return
            spx['ret_5'] = spx.close.pct_change(5)  # 5 day Return

            spx = (spx.replace((np.inf, -np.inf), np.nan)  # Removing missing values
                         .drop(['close'], axis=1)
                         .dropna())
            data = spx

        elif model == 1:
            spx = self.data.copy()
            rus = self.load_data("RUS")
            spx.columns = ["close_spx"]
            rus.columns = ["close_rus"]
            spx_rus = pd.merge(spx, rus, on=["date"], how="inner")

            spx_rus['target'] = spx_rus.close_spx.pct_change()        # Target
            spx_rus['ret_1_SPX'] = spx_rus.close_spx.pct_change()     # 1 day Return
            spx_rus['ret_5_SPX'] = spx_rus.close_spx.pct_change(5)    # 5 day Return
            spx_rus['ret_1_RUS'] = spx_rus.close_rus.pct_change()     # 1 day Return
            spx_rus['ret_5_RUS'] = spx_rus.close_rus.pct_change(5)    # 5 day Return

            spx_rus = (spx_rus.replace((np.inf, -np.inf), np.nan)   # Removing missing values
                         .drop(['close_spx', 'close_rus'], axis=1)
                         .dropna())
            data = spx_rus

        elif model == 2:
            spx = self.data.copy()
            rus = self.load_data("RUS")
            wti = self.load_data("WTI")
            spx.columns = ["close_spx"]
            rus.columns = ["close_rus"]
            wti.columns = ["close_wti"]

            spx_rus = pd.merge(spx, rus, on=["date"], how="inner")
            spx_rus_wti = pd.merge(spx_rus, wti, on=["date"], how="inner")

            spx_rus_wti['target'] = spx_rus_wti.close_spx.pct_change()        # Target
            spx_rus_wti['ret_1_SPX'] = spx_rus_wti.close_spx.pct_change()     # 1 day Return
            spx_rus_wti['ret_5_SPX'] = spx_rus_wti.close_spx.pct_change(5)    # 5 day Return
            spx_rus_wti['ret_1_RUS'] = spx_rus_wti.close_rus.pct_change()     # 1 day Return
            spx_rus_wti['ret_5_RUS'] = spx_rus_wti.close_rus.pct_change(5)    # 5 day Return
            spx_rus_wti['ret_1_WTI'] = spx_rus_wti.close_wti.pct_change()     # 1 day Return
            spx_rus_wti['ret_5_WTI'] = spx_rus_wti.close_wti.pct_change(5)    # 5 day Return

            spx_rus_wti = (spx_rus_wti.replace((np.inf, -np.inf), np.nan)
                           .drop(['close_spx', 'close_rus', "close_wti"], axis=1)
                           .dropna())
            data = spx_rus_wti

        elif model == 3:
            spx = self.data.copy()
            rus = self.load_data("RUS")
            wti = self.load_data("WTI")
            gold = self.load_data("GOLD")
            spx.columns = ["close_spx"]
            rus.columns = ["close_rus"]
            wti.columns = ["close_wti"]
            gold.columns = ["close_gold"]

            spx_rus = pd.merge(spx, rus, on=["date"], how="inner")
            spx_rus_wti = pd.merge(spx_rus, wti, on=["date"], how="inner")
            spx_rus_wti_gold = pd.merge(spx_rus_wti, gold, on=["date"], how="inner")

            spx_rus_wti_gold['target'] = spx_rus_wti_gold.close_spx.pct_change()            # Target
            spx_rus_wti_gold['ret_1_SPX'] = spx_rus_wti_gold.close_spx.pct_change()         # 1 day Return
            spx_rus_wti_gold['ret_5_SPX'] = spx_rus_wti_gold.close_spx.pct_change(5)        # 5 day Return
            spx_rus_wti_gold['ret_1_RUS'] = spx_rus_wti_gold.close_rus.pct_change()         # 1 day Return
            spx_rus_wti_gold['ret_5_RUS'] = spx_rus_wti_gold.close_rus.pct_change(5)        # 5 day Return
            spx_rus_wti_gold['ret_1_WTI'] = spx_rus_wti_gold.close_wti.pct_change()         # 1 day Return
            spx_rus_wti_gold['ret_5_WTI'] = spx_rus_wti_gold.close_wti.pct_change(5)        # 5 day Return
            spx_rus_wti_gold['ret_1_GOLD'] = spx_rus_wti_gold.close_gold.pct_change()       # 1 day Return
            spx_rus_wti_gold['ret_5_GOLD'] = spx_rus_wti_gold.close_gold.pct_change(5)      # 5 day Return

            spx_rus_wti_gold = (spx_rus_wti_gold.replace((np.inf, -np.inf), np.nan)                     # Removing missing values
                                .drop(['close_spx', 'close_rus', "close_wti", "close_gold"], axis=1)
                                .dropna())
            data = spx_rus_wti_gold

        target = data.target.copy()                                           # Copy of Target Variable
        features = data.loc[:, list(data.columns.drop('target'))]        # Explanatory Variables

        if self.normalize:
            N = 63                                                                  # Span length for exp. weigh. mov. std
            features_log = np.log(1 + features)                                     # Log Return
            features_std = features_log.ewm(span=N).std() * np.sqrt(252)            # Exp. weighted Std
            features_norm = features_log / features_std                             # Normalized Data

            y_X = pd.concat([target, features_norm], axis=1)                        # Merge DataFrame
            y_X = (y_X.replace((np.inf, -np.inf), np.nan).dropna())                 # Remove NA's
            y_X = y_X.iloc[252:]                                                    # Remove first 252 obs.

            y = y_X.target                                                          # Target Variable
            X = y_X.loc[:, list(y_X.columns.drop('target'))]                        # Explanatory Variables
        else:
            scaler = MinMaxScaler(feature_range=(-1,1))                             # Scaling between -1 and 1
            features = pd.DataFrame(
                scaler.fit_transform(features),
                columns=features.columns,
                index=features.index)

            y_X = pd.concat([target, features], axis=1)                             # Merge DataFrame
            y_X = (y_X.replace((np.inf, -np.inf), np.nan).dropna())                 # Remove NA's
            y_X = y_X.iloc[252:]                                                    # Remove first 252 obs.

            y = y_X.target                                                          # Target Variable
            X = y_X.loc[:, list(y_X.columns.drop('target'))]                        # Explanatory Variables

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.067)

        return X_train, X_test, y_train, y_test

        log.info(data.info())

    def test_setting(self, X_test):
        counter = int(np.floor(len(X_test.index) / self.trading_days))
        return counter

    def reset(self, training):

        self.training = training
        # Provides starting index for time series and resets step
        """start is everytime random in t = [0, max-252]"""
        if training:
            high = len(self.X_train.index) - (self.trading_days + 1)      # max - 252
            self.offset = np.random.randint(low=0, high=high)       # random start
            self.step = 0
        else:
            self.offset = int(self.counter * self.trading_days)
            self.counter += 1
            self.step = 0

    def take_step(self):
        """Returns data for current trading day and done signal"""
        # Takes all values of the Row (1 day Return, 5 day Return)
        if self.training:
            obs = self.X_train.iloc[self.offset + self.step].values         # Explanatory Variable
            self.target = self.y_train.iloc[self.offset + self.step]             # Target value
            self.step += 1                                                  # +1 Step
            self.date = self.y_train.index[self.offset + self.step]    # Get the date
            done = self.step > self.trading_days                            # If Step bigger than 252 (1 Year), done = True

            return obs, self.target, done
        else:
            obs = self.X_test.iloc[self.offset + self.step].values
            self.target = self.y_test.iloc[self.offset + self.step]
            self.step += 1  # +1 Step
            self.date = self.y_test.index[self.offset + self.step]
            done = self.step > self.trading_days

            return obs, self.target, done


class TradingSimulator:
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps, time_cost_bps):

        self.trading_cost_bps = trading_cost_bps        # Trading costs
        self.time_cost_bps = time_cost_bps              # Time costs (If you're not trading).
        self.steps = steps                              # trading_days = 252
        self.step = 0                                   # Start at step = 0

        self.actions = np.zeros(self.steps)             # [0,0,...,0] 256 times action = [0,1,2] (Short, Neutral, Long)
        self.positions = np.zeros(self.steps)           # [0,0,...,0] 256 times position = [-1,0,1] Am I currently Short, Neutral, Long.
        self.trades = np.zeros(self.steps)              # [0,0,...,0] 256 times How many Trades did the Agent take.

        self.strategy_returns = np.zeros(self.steps)    # [0,0,...,0] 256 times 1 day Return Trading Agent
        self.market_returns = np.zeros(self.steps)      # [0,0,...,0] 256 times 1 day Return (Long)
        self.navs = np.zeros(self.steps)                # [0,0,...,0] 256 times CumSum NAV Trading Agent
        self.market_navs = np.zeros(self.steps)         # [0,0,...,0] 256 times CumSum NAV Market Buy and Hold Strategy

        self.costs = np.zeros(self.steps)               # [0,0,...,0] 256 times (trade + time costs)

    def reset(self):                             # Same as Initial Function

        self.step = 0

        self.actions.fill(0)
        self.positions.fill(0)
        self.trades.fill(0)

        self.strategy_returns.fill(0)
        self.market_returns.fill(0)
        self.navs.fill(0)
        self.market_navs.fill(0)

        self.costs.fill(0)

    def take_step(self, action, market_return):     # Market Return = Observation Wednesday
        """ Calculates NAVs, trading costs and reward
            based on an action and latest market return
            and returns the reward and a summary of the day's activity. """

        self.actions[self.step] = action                                    # Action based on the DDQN
        cur_position = action - 1                                           # [0,1,2] --> [-1, 0, 1] Short, Neutral, Long
        self.positions[self.step] = cur_position                            # New position [-1, 0, 1] Short, Neutral, Long

        # TODO: What to do with the n trades
        prev_position = self.positions[max(0, self.step - 1)]               # Position from Day before
        n_trades = cur_position - prev_position
        #n_trades = 0
        # same to same = 0
        # neutral to long = 1, neutral to short = -1
        # long to short = -2, short to long = 2
        self.trades[self.step] = n_trades                                   # Number of Trades at that day.

        trade_costs = abs(n_trades) * self.trading_cost_bps                 # trades x trading cost per trade
        time_cost = 0 if abs(n_trades) else self.time_cost_bps              # time costs if there was no trade.
                                                                            # If trade then time cost = 0.

        self.costs[self.step] = trade_costs + time_cost                     # Trading costs at that day

        # TODO: Most Important
        reward = (cur_position * market_return) - self.costs[self.step]       # -1, 0, 1 * market return - trading costs
        # TODO: Most Important

        self.market_returns[self.step] = market_return                      # 1-day Return Long and Hold Strategy
        self.strategy_returns[self.step] = reward                           # 1-day Return Trading Agent Strategy

        if self.step != 0:
            self.navs[self.step] = self.navs[self.step - 1] + reward                   # CumSum Trading Agent
            self.market_navs[self.step] = self.market_navs[self.step - 1] + market_return   # CumSum Long and Hold Strategy
        else:
            self.navs[self.step] = reward                                   # CumSum Trading Agent
            self.market_navs[self.step] = market_return                     # CumSum Long and Hold Strategy

        info = {'reward': reward,
                'nav'   : self.navs[self.step],
                'costs' : self.costs[self.step]}

        self.step += 1
        return reward, info

    def result(self):
        """returns current state as pd.DataFrame """
        return pd.DataFrame({'action'         : self.actions,           # all actions
                             'market_return'  : self.market_returns,    # Market Returns
                             'strategy_return': self.strategy_returns,  # Trading Agent returns
                             'nav'            : self.navs,              # NAV Trading Agent (CumSum)
                             'market_nav'     : self.market_navs,       # NAV Long and hold (CumSum)
                             'position'       : self.positions,         # All positions
                             'cost'           : self.costs,             # All costs
                             'trade'          : self.trades})           # Number of trades

class TradingEnvironment(gym.Env):

    metadata = {'render.modes': ['human']}          # For Render Function (No Meaning)

    def __init__(self,
                 trading_days,
                 trading_cost_bps,
                 time_cost_bps,
                 model,
                 ticker='SPX'):

        self.model = model
        self.trading_days = trading_days            # 252 days for one Episode
        self.trading_cost_bps = trading_cost_bps    # 0.001     (0.1%)
        self.time_cost_bps = time_cost_bps          # 0.0001    (0.01%)
        self.ticker = ticker                        # Data is S&P500

        # Get the DataSource class (Data preparation).
        self.data_source = DataSource(trading_days=self.trading_days,
                                      ticker=self.ticker,
                                      model=self.model)

        # Get the Simulator with taking steps, new NAV etc.
        self.simulator = TradingSimulator(steps=self.trading_days,
                                          trading_cost_bps=self.trading_cost_bps,
                                          time_cost_bps=self.time_cost_bps)

        self.action_space = spaces.Discrete(3)      # 0,1,2 (Short, Neutral, Long)

        # Number of nodes (inputs) for the Neural Network
        self.observation_space = len(self.data_source.X_train.columns)
        self.reset(training=True)                                # Reset data_source and simulator Class

    def seed(self, seed=None):
        # Seed for your Customized Environment
        np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Checks if the action_space has the right format (spaces.Discrete(3))
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        # Returns all values of the specific row/day and done signal
        observation, target, done = self.data_source.take_step()

        # action from DDQN and 1-day market return is given to get new NAV, reward, etc.
        reward, info = self.simulator.take_step(action=action,
                                                market_return=target)

        return observation, reward, done, info              # Returns State, Reward, etc.

    def reset(self, training=True):
        """Resets DataSource and TradingSimulator; returns first observation"""

        self.data_source.reset(training=training)
        self.simulator.reset()

        return self.data_source.take_step()[0]  # State Space

# Do not need that
    def render(self, mode='human'):
        """Not implemented"""
        pass