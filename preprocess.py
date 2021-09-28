import sqlite3
import pandas as pd
import datetime
from datetime import datetime


def load_data_split3(path_split3):
    con = sqlite3.connect(path_split3)
    trading_session = pd.read_sql('SELECT * FROM Trading_session', con)
    chart_data = pd.read_sql('SELECT * FROM Chart_data', con)
    trading_session["date"] = pd.to_datetime(trading_session["date"], format="%Y-%m-%d")
    chart_data["time"] = pd.to_datetime(chart_data["time"], format="%H:%M:%S")
    df = chart_data.join(trading_session.set_index("id"), on="session_id", how="inner")
    df["full_date"] = df.apply(lambda x: datetime.strptime(str(x["date"]) + " " + str(x["time"]), '%Y-%m-%d %H:%M:%S'),
                               axis=1)
    del df["time"]
    del df["date"]


def clean_up_deal(x):
    price = []
    lot = []
    deal = []
    datet = []
    for i, elem in enumerate(x["deal_id"]):
        if elem not in deal:
            price.append(x["price"][i])
            lot.append(x["lot_size"][i])
            deal.append(x["deal_id"][i])
            datet.append(x["full_date"][i])
    x["lot_size"] = lot
    x["price"] = price
    x["deal_id"] = deal
    x["full_date"] = datet
    x["counts"] = len(lot)
    return x


# def normilize(df):
#
#
#


def group_date(df):
    grop_ses = df.groupby(['session_id']).size().reset_index(name='counts').sort_values(['counts'], ascending=False)
    grop_ses["price"] = grop_ses.apply(lambda x: list(df[df["session_id"] == x["session_id"]]["price"]), axis=1)
    grop_ses["lot_size"] = grop_ses.apply(lambda x: list(df[df["session_id"] == x["session_id"]]["lot_size"]), axis=1)
    grop_ses["deal_id"] = grop_ses.apply(lambda x: list(df[df["session_id"] == x["session_id"]]["deal_id"]), axis=1)
    grop_ses["full_date"] = grop_ses.apply(lambda x: list(df[df["session_id"] == x["session_id"]]["full_date"]), axis=1)
    grop_ses["trading_type"] = grop_ses.apply(lambda x: list(df[df["session_id"] == x["session_id"]]["trading_type"]),
                                              axis=1)
    grop_ses["platform_id"] = grop_ses.apply(lambda x: list(df[df["session_id"] == x["session_id"]]["platform_id"]),
                                             axis=1)
    return grop_ses
