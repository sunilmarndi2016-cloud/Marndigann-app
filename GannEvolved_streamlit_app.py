# GannEvolved_streamlit_app.py
# Streamlit app implementing the evolved Gann method described.
# Requirements: streamlit, yfinance, pandas, numpy, matplotlib, scipy

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title='GannEvolved', layout='wide')

st.title('GannEvolved — Enter a stock and get Gann-based signals')

with st.sidebar:
    ticker = st.text_input('Ticker (e.g. AAPL or TCS.NS)', value='AAPL')
    period = st.selectbox('Data period', ['6mo','1y','2y','5y','10y'], index=1)
    interval = st.selectbox('Interval', ['1d','1wk','1mo'], index=0)
    capital = st.number_input('Capital (₹ or $)', value=100000.0)
    risk_pct = st.number_input('Max risk per trade (%)', value=1.0)
    atr_w = st.number_input('ATR window', value=14, step=1)
    vol_w = st.number_input('Volatility window (for normalization)', value=20, step=1)
    Ls = st.number_input('Short slope window', value=20, step=1)
    Lm = st.number_input('Medium slope window', value=60, step=1)
    Ll = st.number_input('Long slope window', value=200, step=1)
    z_slope = st.number_input('Slope t-stat threshold', value=2.0, step=0.1)
    breakout_margin = st.number_input('Breakout margin (fraction)', value=0.005)
    run = st.button('Run analysis')

# Helper functions

def fetch_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return None
    df = df[['Open','High','Low','Close','Volume']].dropna()
    return df


def ATR(df, window=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr


def rolling_slope(series, L):
    # returns slope (per bar) and t-stat of slope
    if len(series) < L:
        return np.nan, np.nan
    y = series[-L:].values
    x = np.arange(L)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # slope per bar; t-stat approx = slope / std_err
    t_stat = slope / std_err if std_err != 0 else np.nan
    return slope, t_stat


def compute_signals(df, params):
    atr_w = params['atr_w']
    vol_w = params['vol_w']
    Ls, Lm, Ll = params['Ls'], params['Lm'], params['Ll']
    z_thresh = params['z_slope']
    breakout_margin = params['breakout_margin']

    df = df.copy()
    df['ATR'] = ATR(df, atr_w)
    df['diff'] = df['Close'].diff()
    df['vol'] = df['diff'].rolling(vol_w).std()
    vol_ref = df['vol'].mean()
    df['norm_close'] = df['Close'] / (vol_ref if vol_ref>0 else 1.0)

    slopes = {'S':[], 'M':[], 'L':[]}
    tstats = {'S':[], 'M':[], 'L':[]}

    for i in range(len(df)):
        window = df['norm_close'].iloc[:i+1]
        s, st = rolling_slope(window, int(Ls)) if i+1>=Ls else (np.nan,np.nan)
        m, mt = rolling_slope(window, int(Lm)) if i+1>=Lm else (np.nan,np.nan)
        l, lt = rolling_slope(window, int(Ll)) if i+1>=Ll else (np.nan,np.nan)
        slopes['S'].append(s)
        slopes['M'].append(m)
        slopes['L'].append(l)
        tstats['S'].append(st)
        tstats['M'].append(mt)
        tstats['L'].append(lt)

    df['s_slope'] = slopes['S']
    df['m_slope'] = slopes['M']
    df['l_slope'] = slopes['L']
    df['s_t'] = tstats['S']
    df['m_t'] = tstats['M']
    df['l_t'] = tstats['L']

    # projection: use last LS-window slope to project from the low point within that window
    proj = [np.nan]*len(df)
    for i in range(len(df)):
        if i+1>=Ls:
            base_idx = i - (Ls-1)
            base_price = df['Close'].iloc[base_idx]
            slope = df['s_slope'].iloc[i] * (vol_ref if vol_ref>0 else 1.0)
            proj_price = base_price + slope*(Ls-1)
            proj[i] = proj_price
    df['proj'] = proj

    # signals
    df['buy_signal'] = False
    for i in range(len(df)):
        if i==0 or pd.isna(df['proj'].iloc[i]):
            continue
        cond1 = (df['s_slope'].iloc[i] > 0) and (df['m_slope'].iloc[i] > 0) and (df['l_slope'].iloc[i] > 0)
        cond2 = ((df['s_t'].iloc[i] > z_thresh) + (df['m_t'].iloc[i] > z_thresh) + (df['l_t'].iloc[i] > z_thresh)) >= 2
        cond3 = df['Close'].iloc[i] > df['proj'].iloc[i]*(1+breakout_margin)
        cond4 = df['Volume'].iloc[i] > df['Volume'].rolling(20).mean().iloc[i]*0.8
        if cond1 and cond2 and cond3 and cond4:
            df.at[df.index[i], 'buy_signal'] = True

    return df


def backtest(df, params, capital):
    # extremely simple backtester: enter at close on buy_signal, size by risk
    df = df.copy()
    risk_pct = params['risk_pct']/100.0
    k_stop = 1.5
    trades = []
    position = 0
    cash = capital
    entry_price = 0
    stop = 0

    for i in range(len(df)):
        row = df.iloc[i]
        if position==0 and row['buy_signal']:
            atr = row['ATR'] if not pd.isna(row['ATR']) and row['ATR']>0 else 0.01
            stop_price = row['Close'] - k_stop*atr
            risk_per_share = row['Close'] - stop_price
            if risk_per_share<=0:
                continue
            max_risk_amount = capital * risk_pct
            qty = int(max_risk_amount / risk_per_share) if risk_per_share>0 else 0
            if qty<=0:
                continue
            position = qty
            entry_price = row['Close']
            stop = stop_price
            cash -= qty*entry_price
            trades.append({'entry_date':row.name, 'entry':entry_price, 'qty':qty, 'stop':stop})
        elif position>0:
            # check stop
            if row['Low']<=stop:
                # exit at stop
                exit_price = stop
                cash += position * exit_price
                trades[-1].update({'exit_date':row.name, 'exit':exit_price})
                position = 0
            # optional trailing: if slope drops to negative, exit
            elif row['s_slope']<=0:
                exit_price = row['Close']
                cash += position * exit_price
                trades[-1].update({'exit_date':row.name, 'exit':exit_price})
                position = 0
    # close open position at last close
    if position>0:
        exit_price = df['Close'].iloc[-1]
        cash += position*exit_price
        trades[-1].update({'exit_date':df.index[-1], 'exit':exit_price})
        position = 0

    pnl = cash - capital
    return trades, pnl

# Run
if run:
    df = fetch_data(ticker, period, interval)
    if df is None:
        st.error('No data. Check ticker or interval.')
    else:
        st.subheader(f'Data for {ticker} — last {len(df)} bars')
        st.dataframe(df.tail(10))

        params = dict(atr_w=int(atr_w), vol_w=int(vol_w), Ls=int(Ls), Lm=int(Lm), Ll=int(Ll),
                      z_slope=float(z_slope), breakout_margin=float(breakout_margin), risk_pct=float(risk_pct))

        with st.spinner('Calculating signals...'):
            res = compute_signals(df, params)

        st.line_chart(res[['Close','proj']].dropna())

        st.subheader('Recent buy signals')
        buy_df = res[res['buy_signal']].copy()
        if buy_df.empty:
            st.write('No recent buy signals found.')
        else:
            st.dataframe(buy_df[['Close','Volume','s_slope','m_slope','l_slope','s_t','m_t','l_t','proj']].tail(10))

        st.subheader('Backtest (simple)')
        trades, pnl = backtest(res, params, capital)
        st.write(f'PnL from simple backtest: {pnl:.2f} on capital {capital:.2f}')
        if trades:
            tr_df = pd.DataFrame(trades)
            st.table(tr_df)

        st.markdown('---')
        st.markdown('**Notes:** This is a starting point. Improve with better execution, slippage, fees, and stronger risk management.')


# End of file
