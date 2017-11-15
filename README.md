# Intelligent Instagram Bot

Automatically likes posts in a given list of tags and follows users, maintaining a maximum given number of followed users. 

Uses a hierarchical Bayesian model to predict 1-day followback rates for given hashtags. Randomly selects tags to target, weighted by their upper-confidence-bound followback rate. 

Uses a logistic regression model to predict followback probability (using a target users following and follower counts). Targets users based on this using a dynamically adjusting threshold dependant on how well the bot can achieve it's set rate limits. 

A higher number of threads will lead to the bot being better able to achieve it's set rate limit, so it will generally be stricter in which users it targets, but may be slightly more likely to detected by Instagram and blocked. Settings in the example YAML file are those suggested to avoid being blocked, but you can experiment with higher.

A /\<username\> directory should be made in the same directory as instabot.py. This should contain a YAML settings.yml file, as shown in the example directory. The bot's model data, it's sliding windows (to track likes/follows/unfollows done) and it's queue of followed users will be stored here.

### Install requirements

pip3 install -r requirements.txt

### Usage

python3 instabot.py \<username\>
