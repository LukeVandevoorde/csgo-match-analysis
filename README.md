Python notebook for querying data about your counterstrike matches. Fill in `config.py` with your steam id and the path to the downloaded html file with your counterstrike matches.
The notebook will identify the players you played with most often, and calculate some statistics about which players you win or lose with the most, your statistics on certain maps, compare player performances. There are a couple sections that try to identify the conditions (really just teammates and maps) that maximize your net winrate, either by brute force or with a decision tree.

Right now it's set up to analyze CS:GO matches. To analyze CS2 matches, you would need to change the data cleaning section to exclude matches before the CS2 release, and change the `is_full_long_match` condition to work for MR12 instead of MR16

You can get your match history from the personal game data section on your steam account, but make sure you expand the page to show your entire match history before you download it as html

Redacted sample output is available in the results folder. The notebook is set up so that it's easy to create your filters to analyze data. Querying the data uses a functional paradigm, and you only need pass in a predicate function to filter games you want to include or exclude.
