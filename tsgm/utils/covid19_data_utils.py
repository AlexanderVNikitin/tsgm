"""
Utils for COVID-19 graph time series dataset:
The dataset is based on data from The New York Times, based on reports from state and local health agencies [1].

And was adapted to graph case in [2].
[1] The New York Times. (2021). Coronavirus (Covid-19) Data in the United States. Retrieved [Insert Date Here], from https://github.com/nytimes/covid-19-data.
[2]

The code is an adapted version from:
https://github.com/AlexanderVNikitin/covid19-on-graphs
"""

import pandas as pd


STATE_ADJACENCIES = {
    "washington": ["oregon", "idaho"],
    "oregon": ["washington", "idaho", "nevada", "california"],
    "california": ["oregon", "nevada", "arizona"],
    "idaho": ["washington", "montana", "wyoming", "utah", "nevada", "oregon"],
    "montana": ["north dakota", "south dakota", "wyoming", "idaho"],
    "north dakota": ["minnesota", "south dakota", "montana"],
    "south dakota": ["north dakota", "minnesota", "iowa", "nebraska", "wyoming", "montana"],
    "minnesota": ["wisconsin", "iowa", "south dakota", "north dakota"],
    "michigan": ["indiana", "ohio", "wisconsin"],
    "ohio": ["michigan", "pennsylvania", "west virginia", "kentucky", "indiana"],
    "pennsylvania": ["new york", "new jersey", "delaware", "maryland", "west virginia", "ohio"],
    "new york": ["vermont", "massachusetts", "rhode island", "new jersey", "pennsylvania", "connecticut"],
    "vermont": ["new hampshire", "massachusetts", "new york"],
    "new hampshire": ["maine", "massachusetts", "vermont"],
    "maine": ["new hampshire"],
    "wyoming": ["montana", "south dakota", "nebraska", "colorado", "utah", "idaho"],
    "nebraska": ["south dakota", "iowa", "missouri", "kansas", "colorado", "wyoming"],
    "iowa": ["minnesota", "wisconsin", "illinois", "missouri", "nebraska", "south dakota"],
    "wisconsin": ["minnesota", "iowa", "illinois", "michigan"],
    "illinois": ["wisconsin", "indiana", "kentucky", "missouri", "iowa"],
    "indiana": ["michigan", "ohio", "kentucky", "illinois"],
    "west virginia": ["ohio", "pennsylvania", "maryland", "virginia", "kentucky"],
    "maryland": ["delaware", "pennsylvania", "west virginia", "virginia", "district of columbia"],
    "delaware": ["maryland", "pennsylvania", "new jersey"],
    "new jersey": ["delaware", "pennsylvania", "new york"],
    "connecticut": ["new york", "massachusetts", "rhode island"],
    "rhode island": ["connecticut", "massachusetts", "new york"],
    "district of columbia": ["maryland", "virginia"],
    "virginia": ["west virginia", "kentucky", "district of columbia", "maryland", "north carolina", "tennessee"],
    "kentucky": ["indiana", "ohio", "west virginia", "virginia", "tennessee", "missouri", "illinois"],
    "missouri": ["iowa", "illinois", "kentucky", "tennessee", "arkansas", "oklahoma", "kansas", "nebraska"],
    "kansas": ["nebraska", "missouri", "oklahoma", "colorado"],
    "colorado": ["wyoming", "nebraska", "kansas", "oklahoma", "new mexico", "utah", "arizona"],
    "utah": ["idaho", "wyoming", "colorado", "new mexico", "arizona", "nevada"],
    "nevada": ["oregon", "idaho", "utah", "arizona", "california"],
    "arizona": ["california", "nevada", "utah", "colorado", "new mexico"],
    "new mexico": ["arizona", "utah", "colorado", "oklahoma", "texas"],
    "oklahoma": ["colorado", "kansas", "missouri", "arkansas", "texas", "new mexico"],
    "texas": ["new mexico", "oklahoma", "arkansas", "louisiana"],
    "arkansas": ["oklahoma", "missouri", "tennessee", "mississippi", "louisiana", "texas"],
    "louisiana": ["texas", "arkansas", "mississippi"],
    "mississippi": ["louisiana", "arkansas", "tennessee", "alabama"],
    "tennessee": ["missouri", "kentucky", "virginia", "north carolina", "georgia", "alabama", "mississippi", "arkansas"],
    "alabama": ["mississippi", "tennessee", "georgia", "florida"],
    "georgia": ["tennessee", "north carolina", "south carolina", "florida", "alabama"],
    "florida": ["alabama", "georgia"],
    "south carolina": ["georgia", "north carolina"],
    "north carolina": ["south carolina", "tennessee", "virginia", "georgia"],
    "alaska": [],
    "hawaii": [],
    "massachusetts": ["new york", "vermont", "new hampshire", "rhode island", "connecticut"],
}

LIST_OF_STATES = sorted(STATE_ADJACENCIES.keys())

# July 1 2019
STATE_POPULATION = {
    "california": 39_512_223,
    "texas": 28_995_881,
    "florida": 21_477_737,
    "new york": 19_453_561,
    "pennsylvania": 12_801_989,
    "illinois": 12_671_821,
    "ohio": 11_689_100,
    "georgia": 10_617_423,
    "north carolina": 10_488_084,
    "michigan": 9_986_857,
    "new jersey": 8_882_190,
    "virginia": 8_535_519,
    "washington": 7_614_893,
    "arizona": 7_278_717,
    "massachusetts": 6_949_503,
    "tennessee": 6_833_174,
    "indiana": 6_732_219,
    "missouri": 6_137_428,
    "maryland": 6_045_680,
    "wisconsin": 5_822_434,
    "colorado": 5_758_736,
    "minnesota": 5_639_632,
    "south carolina": 5_148_714,
    "alabama": 4_903_185,
    "louisiana": 4_648_794,
    "kentucky": 4_467_673,
    "oregon": 4_217_737,
    "oklahoma": 3_956_971,
    "connecticut": 3_565_287,
    "utah": 3_205_958,
    "iowa": 3_155_070,
    "nevada": 3_080_156,
    "arkansas": 3_017_825,
    "mississippi": 2_976_149,
    "kansas": 2_913_314,
    "new mexico": 2_096_829,
    "nebraska": 1_934_408,
    "west virginia": 1_792_147,
    "idaho": 1_787_065,
    "hawaii": 1_415_872,
    "new hampshire": 1_359_711,
    "maine": 1_344_212,
    "montana": 1_068_778,
    "rhode island": 1_059_361,
    "delaware": 973_764,
    "south dakota": 884_659,
    "north dakota": 762_062,
    "alaska": 731_545,
    "district of columbia": 705_749,
    "vermont": 623_989,
    "wyoming": 578_759,
    "virgin islands": 104_914,
    "puerto rico": 3_193_694,
    "guam": 165_718,
}


def aggregate_by_weeks_max(df):
    df['date'] = pd.to_datetime(df['date'])  # + pd.to_timedelta(7, unit='d')
    df = df.groupby(['state', pd.Grouper(key='date', freq='W-MON')])\
           .agg({"cases": max, "deaths": max})\
           .reset_index()\
           .sort_values('date')
    return df


def get_adjacencies_graph():
    nodes, edges = [], []
    LIST_OF_STATES = sorted(STATE_ADJACENCIES.keys())

    for state_name in LIST_OF_STATES:
        nodes.append(state_name)

    for state, adj_states in STATE_ADJACENCIES.items():
        for adj_state in adj_states:
            edges.append((state, adj_state))
    return nodes, edges


def covid_dataset(path):
    covid_cases_df = pd.read_csv(path)
    covid_cases_df["state"] = covid_cases_df["state"].str.lower()
    covid_cases_df = aggregate_by_weeks_max(covid_cases_df)
    graph = get_adjacencies_graph()
    result = {}
    for row in covid_cases_df.to_dict(orient="records"):
        date = row["date"]
        cases = row["cases"]
        deaths = row["deaths"]
        state = row["state"]
        if date not in result:
            result[date] = {}
        if state in STATE_POPULATION:
            result[date][state] = {
                "deaths_normalized": deaths / STATE_POPULATION[state],
                "cases_normalized": cases / STATE_POPULATION[state],
                "deaths": deaths,
                "cases": cases,
            }
        else:
            print("[WARNING]: There is no data about population for: ", state)

    # fill missing values with zeros
    for date in result.keys():
        for state in LIST_OF_STATES:
            if state not in result[date]:
                result[date][state] = {
                    "deaths": 0,
                    "cases": 0,
                    "deaths_normalized": 0,
                    "cases_normalized": 0,
                }
    return result, graph
