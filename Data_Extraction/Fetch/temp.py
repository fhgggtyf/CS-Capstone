patch_versions = [
    "1_04", "1_05", "1_06", "1_1", "1_11", "1_12", "1_2", "1_21", "1_22", "1_23",
    "1_3", "1_31", "1_5", "1_52", "1_6", "1_61", "1_62", "1_63",
    "2_0", "2_01", "2_02", "2_1", "2_11", "2_12", "2_13", "2_2", "2_21"
]

for version in patch_versions:
    patch_readable = version.replace('_', '.')
    table = f"Cyberpunk_2077_Official_Forum_Reviews_{version}"
    print(f"""INSERT INTO Cyberpunk_2077_Official_Forum_Reviews
SELECT post_num, post_time, replied_text, main_text, upvote_number, '{patch_readable}' AS patch_version
FROM {table};\n""")
