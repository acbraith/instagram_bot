# Intelligent Instagram Bot

Automatically likes posts in a given list of tags and follows users, maintaining a maximum given number of followed users. 

The bot will search through hashtag feeds and targetted users follower lists to find potential targets. It will predict followback confidence for each potential target, and prioritise following and liking those it predicts to be most likely to follow back (aiming to maximise 1-day followback rate).

A sub-directory should be made in the same directory as instabot.py. This should contain a YAML settings.yml file, as shown in the example directory. The bot's model data, it's sliding windows (to track likes/follows/unfollows done) and it's queue of followed users will be stored here.

### Install requirements

pip3 install -r requirements.txt

### Usage

python3 instabot.py \<settings directory\>
