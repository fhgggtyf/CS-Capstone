import subprocess

# List of games to process
game_list = [
    # "https://www.metacritic.com/game/crash-bandicoot-4-its-about-time/",
    # "https://www.metacritic.com/game/dark-souls-remastered/",
    # "https://www.metacritic.com/game/sekiro-shadows-die-twice/",
    # "https://www.metacritic.com/game/cuphead/",
    # "https://www.metacritic.com/game/super-meat-boy/",
    # "https://www.metacritic.com/game/devil-may-cry-5/",
    # "https://www.metacritic.com/game/battlefield-4/",
    # "https://www.metacritic.com/game/assassins-creed-unity/",
    # "https://www.metacritic.com/game/cyberpunk-2077/",
    # "https://www.metacritic.com/game/the-master-chief-collection/",
    # "https://www.metacritic.com/game/fallout-76/",
    # "https://www.metacritic.com/game/mass-effect-andromeda/",
    # "https://www.metacritic.com/game/battlefield-2042/",
    # "https://www.metacritic.com/game/batman-arkham-knight/",
    # "https://www.metacritic.com/game/no-mans-sky/",
    # "https://www.metacritic.com/game/anthem/",
    # "https://www.metacritic.com/game/star-wars-battlefront-ii/",
    # "https://www.metacritic.com/game/mighty-no-9/",
    # "https://www.metacritic.com/game/dota-2/",
    # "https://www.metacritic.com/game/fortnite/",
    # "https://www.metacritic.com/game/league-of-legends/",
    # "https://www.metacritic.com/game/tom-clancys-rainbow-six-siege/",
    # "https://www.metacritic.com/game/the-binding-of-isaac-repentance/",
    # "https://www.metacritic.com/game/getting-over-it-with-bennett-foddy/",
    # "https://www.metacritic.com/game/jump-king/",
    # "https://www.metacritic.com/game/mortal-shell/",
    # "https://www.metacritic.com/game/street-fighter-v/",
    # "https://www.metacritic.com/game/counter-strike-2/",
    # "https://www.metacritic.com/game/apex-legends/",
    # "https://www.metacritic.com/game/call-of-duty-warzone/",
    # "https://www.metacritic.com/game/hogwarts-legacy/",
    # "https://www.metacritic.com/game/palworld/",
    # "https://www.metacritic.com/game/baldurs-gate-3/",
    # "https://www.metacritic.com/game/the-lord-of-the-rings-gollum/",
    # "https://www.metacritic.com/game/call-of-duty-modern-warfare-iii/",
    # "https://www.metacritic.com/game/playerunknowns-battlegrounds/",
    # "https://www.metacritic.com/game/war-thunder/",
    # "https://www.metacritic.com/game/helldivers-2/",
    # "https://www.metacritic.com/game/overwatch-2/",
    # "https://www.metacritic.com/game/delta-force/",
    "https://www.metacritic.com/game/ea-sports-fc-25/",
    "https://www.metacritic.com/game/warcraft-iii-reforged/",
    "https://www.metacritic.com/game/valorant/",
    "https://www.metacritic.com/game/genshin-impact/",
    "https://www.metacritic.com/game/furi/"
    # Add more as needed
]

for game_info in game_list:
    print(f"\n=== Running scraper for {game_info} ===")
    
    # Run the target script non-interactively by passing inputs via stdin
    # The 'y\ny\n' simulates user typing 'y' and pressing enter twice
    process = subprocess.run(
        ["python3", "Data_Extraction/Fetch/specific_website_fetches/metacritic_scraper.py"],
        input=f"{game_info}\ny\ny\n",
        text=True
    )

    print(f"=== Finished {game_info} ===\n")
