from datetime import datetime
import requests
import pickle
import time
from pathlib import Path

def get_user_reviews(review_appid, params):

    user_review_url = f'https://store.steampowered.com/appreviews/{review_appid}'
    req_user_review = requests.get(
        user_review_url,
        params=params,
        timeout=15
    )

    if req_user_review.status_code != 200:
        print(f'Fail to get response. Status code: {req_user_review.status_code}')
        return {"success": 2}
    
    try:
        user_reviews = req_user_review.json()
    except:
        return {"success": 2}

    return user_reviews
    
review_appname = input("Input game name")                            # the game name
review_appid = input("Input the game appid on Steam")                            # the game appid on Steam 
review_appid = review_appid.strip()  # remove any leading or trailing whitespace

LANGUAGE = input("Input the language you want to fetch reviews in (default: english): ").strip() or 'english'  # the language of the reviews                                  # the game appid on Steam

# the params of the API
params = {
        'json':1,
        'language': LANGUAGE,
        'cursor': '*',                                  # set the cursor to retrieve reviews from a specific "page"
        'num_per_page': 100,
        'filter': 'all'
    }

# time_interval = timedelta(hours=24)                         # the time interval to get the reviews
# end_time = datetime.fromtimestamp(1716718910)               # the timestamp in the return result are unix timestamp (GMT+0)
end_time = datetime.now()
# start_time = end_time - time_interval
start_time = datetime(1970, 1, 1, 0, 0, 0)

print(f"Start time: {start_time}")
print(f"End time: {end_time}")
print(start_time.timestamp(), end_time.timestamp())

passed_start_time = False
passed_end_time = False

selected_reviews = []

while (not passed_start_time or not passed_end_time):
    print("fetching reviews...")
    try:
        reviews_response = get_user_reviews(review_appid, params)
        # not success?
        if reviews_response["success"] != 1:
            print("Not a success")
            print(reviews_response)
            raise Exception("API did not return success")
    except Exception as e:
        print(f"Error fetching reviews: {e}")
        user_input = input("Connection failed, retrying: ")
        time.sleep(1)
        continue  # retry the current iteration after user confirmation

    # not success?
    if reviews_response["success"] != 1:
        print("Not a success")
        print(reviews_response)

    if reviews_response["query_summary"]['num_reviews'] == 0:
        print("No reviews.")
        print(reviews_response)

    for review in reviews_response["reviews"]:
        recommendation_id = review['recommendationid']
        
        timestamp_created = review['timestamp_created']
        timestamp_updated = review['timestamp_updated']

        # skip the comments that beyond end_time
        if not passed_end_time:
            if timestamp_created > end_time.timestamp():
                continue
            else:
                passed_end_time = True
                
        # exit the loop once detected a comment that before start_time
        if not passed_start_time:
            if timestamp_created < start_time.timestamp():
                passed_start_time = True
                break

        # extract the useful (to me) data
        author_steamid = review['author'].get('steamid', '')
        playtime_forever = review['author'].get('playtime_forever', 0)
        playtime_last_two_weeks = review['author'].get('playtime_last_two_weeks', 0)
        playtime_at_review_minutes = review['author'].get('playtime_at_review', 0)
        last_played = review['author'].get('last_played', 0)

        review_text = review['review']
        voted_up = review['voted_up']
        votes_up = review['votes_up']
        votes_funny = review['votes_funny']
        weighted_vote_score = review['weighted_vote_score']
        steam_purchase = review['steam_purchase']
        received_for_free = review['received_for_free']
        written_during_early_access = review['written_during_early_access']

        my_review_dict = {
            'recommendationid': recommendation_id,
            'author_steamid': author_steamid,
            'playtime_at_review_minutes': playtime_at_review_minutes,
            'playtime_forever_minutes': playtime_forever,
            'playtime_last_two_weeks_minutes': playtime_last_two_weeks,
            'last_played': last_played,

            'review_text': review_text,
            'timestamp_created': timestamp_created,
            'timestamp_updated': timestamp_updated,

            'voted_up': voted_up,
            'votes_up': votes_up,
            'votes_funny': votes_funny,
            'weighted_vote_score': weighted_vote_score,
            'steam_purchase': steam_purchase,
            'received_for_free': received_for_free,
            'written_during_early_access': written_during_early_access,
        }

        selected_reviews.append(my_review_dict)

    # go to next page
    try:
        cursor = reviews_response['cursor']         # cursor field does not exist in the last page
    except Exception as e:
        cursor = ''

    # no next page
    # exit the loop
    if not cursor:
        print("Reached the end of all comments.")
        break

    if reviews_response["query_summary"]['num_reviews'] == 0:
        print("No reviews.")
        print(reviews_response)
        break 
    
    
    # set the cursor object to move to next page to continue
    params['cursor'] = cursor
    print('To next page. Next page cursor:', cursor)
    
    
# save the selected reviews to a file

foldername = f"{review_appid}_{review_appname}"
filename = f"{review_appid}_{review_appname}_{LANGUAGE}_reviews_{start_time.strftime('%Y%m%d-%H%M%S')}_{end_time.strftime('%Y%m%d-%H%M%S')}.pkl"
output_path = Path(
    foldername, filename
)
if not output_path.parent.exists():
    output_path.parent.mkdir(parents=True)

pickle.dump(selected_reviews, open(output_path, 'wb'))
