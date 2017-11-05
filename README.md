# Instagram Bot

Automatically likes posts in a given list of tags and follows users, maintaining a maximum of 50 followed users. Aims to learn which users will be most effective to target over time to maximise 1 day follow-back rate.

Settings can be changed in the code, planned to make more accessible from command line / saved settings in files later.

A /<username> directory should be made in the same directory as instabot.py. This should contain the files
* pass.txt: single line, password
* tags.txt: one tag per line (no #)

Usage: python3 instabot.py <USERNAME>
