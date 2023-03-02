import pandas as pd

from utils import (fetch_from_fangcovid_local, fetch_rawtext_from_wiki, split_classify_wiki_text)


class DatasetCreator():

    def __init__(
        self,
        df,
    ) -> None:
        self.df = df
        pass

    def loop_over_article_list(
        self,
        articles=[
            "Maschinelles Lernen", "Medizin", "Wissenschaft", "Krankheit", "Prävention", "Diagnose",
            "Politik", "COVID-19", "COVID-19-Pandemie", "Epidemie", "Mykose",
            "Sexuell übertragbare Erkrankung", "Infektionskrankheit", "Bundestag", "Bundesrat",
            "Zeitung", "Rundfunk", "Verlag", "Politisches System der Bundesrepublik Deutschland",
            "Politisches System", "Massenmedien", "Medienwissenschaft", "Publikation"
        ],
    ) -> pd.DataFrame:
        text = ''
        for name in articles:
            print(name)
            raw = fetch_rawtext_from_wiki(name)
            text = text + raw
        self.df = split_classify_wiki_text(text)
        return self.df
