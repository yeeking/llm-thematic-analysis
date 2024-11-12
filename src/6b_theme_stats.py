import pandas as pd
import json
from collections import defaultdict

# Load the CSV file into a DataFrame
data = pd.read_csv('your_file.csv')

# Initialize the dictionary to map themes to their quotes
theme_to_quotes = defaultdict(list)

# Iterate over the dataframe, decode the quotes JSON, and populate the theme_to_quotes dictionary
for index, row in data.iterrows():
    theme = row['theme']
    quotes = json.loads(row['quotes'])
    theme_to_quotes[theme].extend(quotes)

# For each theme, count unique, shared, and total quotes
results = []
all_quotes = {theme: set(quotes) for theme, quotes in theme_to_quotes.items()}

for theme, quotes in all_quotes.items():
    unique_quotes = quotes.copy()  # Start with all quotes assumed unique
    shared_count = 0
    shared_unique_quotes = set()  # Track quotes shared with at least one other theme

    for other_theme, other_quotes in all_quotes.items():
        if theme != other_theme:
            shared_quotes = quotes.intersection(other_quotes)
            shared_count += len(shared_quotes)
            shared_unique_quotes.update(shared_quotes)  # Add distinct shared quotes

    # Calculate unique quotes by removing shared ones
    unique_quotes -= shared_unique_quotes

    # Calculate counts
    unique_count = len(unique_quotes)
    shared_unique_count = len(shared_unique_quotes)
    total_count = len(quotes)

    # Assert that shared_unique_count + unique_count == total_count
    assert shared_unique_count + unique_count == total_count, (
        f"Assertion failed for theme '{theme}': shared_unique_count + unique_count "
        f"({shared_unique_count + unique_count}) != total_count ({total_count})"
    )

    # Calculate unique to shared ratio
    unique_to_shared_ratio = unique_count / shared_unique_count if shared_unique_count > 0 else None

    results.append({
        'theme': theme,
        'total_count': total_count,                   # Total quotes for the theme
        'unique_count': unique_count,
        'shared_count': shared_count,                 # Total number of times quotes are shared
        'shared_unique_count': shared_unique_count,   # Distinct quotes shared with other themes
        'unique_to_shared_ratio': unique_to_shared_ratio  # Ratio of unique to shared_unique quotes
    })

# Convert the results to a DataFrame, sort by unique quotes, and display it
result_df = pd.DataFrame(results)
result_df = result_df.sort_values(by='unique_count', ascending=False)
print(result_df)
