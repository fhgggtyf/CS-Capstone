import subprocess

# List of games to process
# Example structure: [["Elden Ring", 1245620], ["Hades", 1145360]]
game_list = [
    # ['Dark_Souls_Remastered',570940],
    # ['Sekiro',814380],
    # ['Cuphead',268910],
    # ['Super_Meat_Boy',40800],
    # ['Devil_May_Cry_5',601150],
    # ['Battlefield_4',1238860],
    # ['Assasins_Creed_Unity',289650],
    # ['Cyberpunk_2077',1091500],
    # ['Halo_MCC',976730],
    # ['Fallout_76',1151340],
    # ['Mass_Effect_Andromeda',1238000],
    # ['Battlefield_2042',1517290],
    # ['Batman_Arkham_Knight',208650],
    # ['No_Mans_Sky',275850],
    # ['Starwars_Battlefront_2',1237950],
    # ['Mighty_No_9',314710],
    # ['DOTA_2',570],
    # ['Rainbow_6_Siege',359550],
    # ['The_Binding_Of_Isaac_Repentance',1426300],
    # ['Getting_Over_It_with_Bennett_Foddy',240720],
    # ['Jump_King',1061090],
    # ['Mortal_Shell',1110910],
    # ['Street_Fighter_V',310950],
    # ['CS2',730],
    # ['APEX',1172470],
    # ['Call_of_Duty_Warzone',1962663],
    # ['Hogwarts_Legacy',990080],
    # ['PalWorld',1623730],
    # ['Baldurs_Gate_3',1086940],
    # ['LOTR_Gollum',1265780],
    # ['COD_Modern_Warfare_III',3595270],
    # ['PUBG_BATTLEGROUNDS',578080],
    # ['War_Thunder',236390],
    # ['Helldivers_2',553850],
    # ['Overwatch_2',2357570],
    # ['Delta_Force',2507950],
    ['EA_SPORTS_FC_25',2669320]
    # Add more as needed
]

for game_info in game_list:
    game_name = game_info[0]
    game_id = game_info[1]
    print(f"\n=== Running scraper for {game_name} (App ID: {game_id}) ===")
    
    # Run the target script non-interactively by passing inputs via stdin
    # The 'y\ny\n' simulates user typing 'y' and pressing enter twice
    process = subprocess.run(
        ["python3", "Data_Extraction/Fetch/steam_review_fetch.py"],
        input=f"{game_name}\n{game_id}\ny\ny\n",
        text=True
    )

    print(f"=== Finished {game_name} ===\n")
