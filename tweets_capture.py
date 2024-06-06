#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from twarc import Twarc2, expansions
import json
import csv
import datetime as dt
from time import sleep
            
FILENAME = "./tweet_capture.csv"
MINUTE_DELAY = 60*24

# API Keys
apikey = ''
apikeysecret = ''
bearertoken = ''
accestoken = ''
accestokensecret = ''

# API Twiter
client = Twarc2(bearer_token=bearertoken)

# Dataset cols
keys = ['date', 'id_tweet', 'text', 'language', 'possibly_sensitive', 'hashtags', 'mentions',
        'truncated', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'id_user',
        'username', 'verified', 'location', 'extended_user', 'regla_de_bulo']

# CSV   
output_file = open(FILENAME, 'a', newline='')
dict_writer = csv.DictWriter(output_file, keys)
dict_writer.writeheader()

# Capture
hashtags_clave = [
    "(El Hormiguero)",
    ""
]


def keys_exists(element, *keys):
    '''
    Check if *keys (nested) exists in `element` (dict).
    '''
    if not isinstance(element, dict):
        raise AttributeError('keys_exists() expects dict as first argument.')
    if len(keys) == 0:
        raise AttributeError('keys_exists() expects at least two arguments, one given.')

    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True

def get_delete_user(t):
    author = json.dumps(t['author']) 
    del t['author'] 
    del t['__twarc']
    return author
    
def scrape(query, start_time, end_time):
    
    search_results = client.search_all(
        query=query + ' lang:es -is:retweet -is:reply -is:quote',
        start_time=start_time,
        end_time=end_time,
        max_results=100
    )
    
                       
    tweets = []            
    for page in search_results:
        for t in expansions.flatten(page): 
            d = {
                'date': t['created_at'],
                'id_tweet': t['id'],
                'text': t['text'],
                'language': t['lang'],
                'possibly_sensitive': t['possibly_sensitive'],
                'hashtags': ";".join(["#"+x['tag'] for x in t['entities']['hashtags']]) if keys_exists(t, "entities", "hashtags") else '',
                'mentions': ";".join(["@"+x['username'] for x in t['entities']['mentions']]) if keys_exists(t, "entities", "mentions") else '',
                'truncated': True if 'truncated' in t.keys() else False,
                'retweet_count':  t['public_metrics']['retweet_count'],
                'reply_count': t['public_metrics']['reply_count'],
                'like_count': t['public_metrics']['like_count'],
                'quote_count': t['public_metrics']['quote_count'],
                'id_user': t['author_id'],
                'username': t['author']['username'],
                'verified': t['author']['verified'],
                'location': t['author']['location'] if 'location' in t['author'].keys() else '',
                'regla_de_bulo': query
            }

            tweets.append(d)
    dict_writer.writerows(tweets)   
    return tweets



end_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=MINUTE_DELAY)
while 1:
    start_time = end_time 
    end_time = dt.datetime.now(dt.timezone.utc)
    for query in hashtags_clave:
        tweets = scrape(query, start_time, end_time - dt.timedelta(minutes=1))
    print("* [START TIME] " , start_time , '[END TIME] ', end_time)
      
    next_time = end_time + dt.timedelta(minutes=MINUTE_DELAY)
    while dt.datetime.now(dt.timezone.utc) < next_time:
        sleep(3600)
