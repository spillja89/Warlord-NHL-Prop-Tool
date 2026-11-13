"""Owner-approved books for future quote selection; historical records untouched."""
import json
import re

VERSION = 'my-books-2026-11-12'
BOOKS = ('Hard Rock Bet', 'bet365', 'BetMGM', 'BetRivers', 'FanDuel',
         'DraftKings', 'theScore Bet', 'Fanatics')
_ALIASES = {
    'hardrock': 'Hard Rock Bet', 'hardrockbet': 'Hard Rock Bet',
    'hardrocksportsbook': 'Hard Rock Bet',
    'bet365': 'bet365', 'bet365us': 'bet365',
    'betmgm': 'BetMGM', 'mgm': 'BetMGM',
    'betrivers': 'BetRivers',
    'fanduel': 'FanDuel', 'fanduelsportsbook': 'FanDuel',
    'draftkings': 'DraftKings', 'dk': 'DraftKings',
    'draftkingssportsbook': 'DraftKings',
    'thescore': 'theScore Bet', 'thescorebet': 'theScore Bet', 'espnbet': 'theScore Bet',
    'fanatics': 'Fanatics', 'fanaticssportsbook': 'Fanatics',
}


def preferred_book(value):
    """Exact normalized aliases only; never substitute a foreign/state variant."""
    return _ALIASES.get(re.sub(r'[^a-z0-9]', '', str(value or '').casefold()), '')


def prepare_quote_frame(frame):
    """Clear unsupported raw winners before merging, keeping roster/profile data."""
    result = frame.copy()
    for market in ('Points', 'Assists', 'SOG', 'Goal', 'ATG', 'PPP'):
        for suffix in ('', '_1', '_2', '_3', '_4'):
            book = f'BDL_{market}_Book{suffix}'
            odds = f'BDL_{market}_Odds{suffix}'
            if book not in result:
                if odds in result:
                    result[odds] = float('nan')
                continue
            unavailable = ~result[book].map(preferred_book).astype(bool)
            result.loc[unavailable, book] = ''
            if odds in result:
                result.loc[unavailable, odds] = float('nan')
    if 'PPP_Book_Prices_05' in result:
        def keep(raw):
            try:
                values = json.loads(raw)
                if not isinstance(values, list):
                    return '[]'
                return json.dumps([dict(item, book=preferred_book(item.get('book')))
                                   for item in values if isinstance(item, dict)
                                   and preferred_book(item.get('book'))])
            except (ValueError, TypeError):
                return '[]'
        result['PPP_Book_Prices_05'] = result['PPP_Book_Prices_05'].map(keep)
    result['Odds_Book_Policy'] = VERSION
    result['Odds_Selected_Books'] = json.dumps(BOOKS)
    return result
