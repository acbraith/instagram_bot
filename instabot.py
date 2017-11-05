from InstagramAPI import InstagramAPI
from persistqueue import SQLiteQueue
from sklearn.linear_model import LogisticRegression
from time import sleep
import random, pprint, requests, json, datetime, sys
import numpy as np

# persistent dict
from collections import MutableMapping
import sqlite3, pickle, os
class PersistentDict(MutableMapping):
    '''
    From
    https://stackoverflow.com/questions/9320463/persistent-memoization-in-python
    '''
    def __init__(self, dbpath, iterable=None, **kwargs):
        self.dbpath = dbpath+'/dict'
        if not os.path.exists(dbpath):
            os.makedirs(dbpath)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'create table if not exists memo '
                '(key blob primary key not null, value blob not null)'
            )
        if iterable is not None:
            self.update(iterable)
        self.update(kwargs)

    def encode(self, obj):
        return pickle.dumps(obj)

    def decode(self, blob):
        return pickle.loads(blob)

    def get_connection(self):
        return sqlite3.connect(self.dbpath, timeout=300)

    def  __getitem__(self, key):
        key = self.encode(key)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select value from memo where key=?',
                (key,)
            )
            value = cursor.fetchone()
        if value is None:
            raise KeyError(key)
        return self.decode(value[0])

    def __setitem__(self, key, value):
        key = self.encode(key)
        value = self.encode(value)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'insert or replace into memo values (?, ?)',
                (key, value)
            )

    def __delitem__(self, key):
        key = self.encode(key)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select count(*) from memo where key=?',
                (key,)
            )
            if cursor.fetchone()[0] == 0:
                raise KeyError(key)
            cursor.execute(
                'delete from memo where key=?',
                (key,)
            )

    def __iter__(self):
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select key from memo'
            )
            records = cursor.fetchall()
        for r in records:
            yield self.decode(r[0])

    def __len__(self):
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select count(*) from memo'
            )
            return cursor.fetchone()[0]

class Item:
	def __init__(self, value):
		self.value = value
		self.timestamp = datetime.datetime.now()
class SlidingWindow:
	def __init__(self, length, path):
		self.length = datetime.timedelta(hours=length)
		self.items = SQLiteQueue(path)
	def _clean(self):
		# remove old items
		items = []
		while len(self.items) > 0:
			items += [self.items.get()]
		items = [i for i in items if (datetime.datetime.now()-i.timestamp) < self.length]
		for i in items:
			self.items.put(i)
	def __len__(self):
		self._clean()
		return len(self.items)
	def put(self, item):
		self._clean()
		self.items.put(Item(item))
	def get(self):
		self._clean()
		raise Exception("Not Implemented")

class InstaBot:
	def __init__(self, username, password,
		tag_list, 
		max_hour_likes=30, max_hour_follows=15,
		likes_per_user=3,
		mean_wait_time=5,
		max_followed=50,
		max_photo_likes=100, min_photo_likes=0):
		self.username = username
		self.password = password
		self.tag_list = tag_list
		self.max_hour_likes = max_hour_likes
		self.max_hour_follows = max_hour_follows
		self.likes_per_user = likes_per_user
		self.mean_wait_time = mean_wait_time
		self.max_followed = max_followed
		self.max_photo_likes = max_photo_likes
		self.min_photo_likes = min_photo_likes

		self.followed_queue = SQLiteQueue(username+'/followed_users')
		self.hour_likes = SlidingWindow(1, username+'/hour_likes')
		self.hour_follows = SlidingWindow(1, username+'/hour_follows')
		self.hour_unfollows = SlidingWindow(1, username+'/hour_unfollows')

		self.target_data = PersistentDict(username+'/target_data')

		self.api = InstagramAPI(username, password)
		self.send_request(self.api.login)

	def wait(self):
		t = np.random.exponential(self.mean_wait_time)
		print("\tSleep",round(t,2),"seconds")
		sleep(t)

	def send_request(self, request, *args, **kwargs):
		print(request.__name__, args, kwargs)
		try:
			success = request(*args, **kwargs)
		except Exception as e:
			print(e)
			print("Sleep 10 minutes")
			sleep(600)
			return None
		if not(success):
			status_code = self.api.LastResponse.status_code
			print("HTTP", status_code)
			print(self.api.LastJson)
			if status_code in [400, 429]:
				print("HTTP", status_code)
				print("Sleep 6 hours")
				sleep(6*60*60)
				return None
			else:
				self.wait()
				return None

		self.wait()
		return self.api.LastJson

	def get_tag_feed(self, tag):
		return self.send_request(self.api.tagFeed, tag)

	def get_user_followers(self, user_id):
		ret = self.send_request(self.api.getUserFollowers, user_id)
		if ret is None:  return 0
		return len(ret['users'])
	def get_user_followings(self, user_id):
		ret = self.send_request(self.api.getUserFollowings, user_id)
		if ret is None: return 0
		return len(ret['users'])

	def get_friendship_info(self, user_id):
		ret = self.send_request(self.api.userFriendship, user_id)
		if ret is None: return {
			'following': False, 'followed_by': False, 'is_bestie': False, 
			'blocking': False, 'is_blocking_reel': False, 
			'is_muting_reel': False, 
			'outgoing_request': False, 'incoming_request': False, 
			'is_private': False, 'status': 'ok'}
		return ret

	def get_model_data(self):
		X = np.zeros((0,3))
		Y = np.zeros((0,))
		for target, data in self.target_data.items():
			if len(data) < 5 and \
				datetime.datetime.now() - data[0] > datetime.timedelta(days=1):
				y = 1 if get_friendship_info(target)['followed_by'] else 0
				data = (data)+(y,)
				self.target_data[target] = data

			if len(data) == 5:
				x = data[1:-1]
				y = data[-1]
				X = np.append(X, np.array(x).reshape(1,-1), axis=0)
				Y = np.append(Y, np.array(y).reshape(1,), axis=0)
		return X,Y

	def get_model(self):
		X,y = self.get_model_data()
		if len(np.unique(y)) > 1:
			return LogisticRegression().fit(X,y)
		else:
			return None

	def is_user_target(self, item):
		'''
		Reinforcement learning based decision
		inputs:
			user followers, followings
			photo likes
		output:
			follow back probability (trained from historical data)

		Need to track inputs alongside user id for all users targetted
		Then 1 day later check if followed back or not; log these as outputs
		Train model on this
		Use model to predict follow back probability for new targets
			logistic regression model
		epsilon greedy strategy;
			probability 1-epsilon: only target if followback prob > alpha
			probability epsilon always target

		'''
		user_id = item['user']['pk']
		# decide if good target
		# basic criteria
		if not(item['has_liked']) and \
			not(item['user']['friendship_status']['following']) and \
			not(item['user']['friendship_status']['is_bestie']) and \
			not(item['user']['friendship_status']['outgoing_request']):

			user_followers = self.get_user_followers(user_id)
			user_followings = self.get_user_followings(user_id)
			media_likes = item['like_count']

			X = np.array([user_followers, user_followings, media_likes]).reshape(1,-1)

			model = self.get_model()

			alpha = 0.5
			epsilon = 0.1
			p = 1 if model is None else model.predict_proba(X)[0,list(model.classes_).index(1)]
			print("Target Data",X)
			print("Followback Probability",p)

			if p > alpha or random.random() < epsilon: 
				self.target_data[user_id] = (
					datetime.datetime.now(), 
					user_followers, user_followings, media_likes)
				return True

		return False

	def target_user(self, user_id):
		print("Targetting user", user_id)
		items = self.send_request(self.api.getUserFeed, user_id)
		if items is None: 
			print("User Feed 404")
			return

		# like
		for i,user_item in enumerate(items['items']):
			if i >= self.likes_per_user: break

			user_media_id = user_item['pk']

			if not(user_item['has_liked']):
				while len(self.hour_likes) >= self.max_hour_likes:
					print("Too many likes in 1 hour ("+
						str(len(self.hour_likes))+"), sleep 10 minutes")
					sleep(600)
				self.send_request(self.api.like, user_media_id)
				self.hour_likes.put(user_media_id)

		# follow
		while len(self.hour_follows) >= self.max_hour_follows:
			print("Too many follows in 1 hour ("+
				str(len(self.hour_follows))+"), sleep 10 minutes")
			sleep(600)
		self.send_request(self.api.follow, user_id)
		self.hour_follows.put(user_id)
		self.followed_queue.put(user_id)

	def unfollow_users(self):
		# unfollow if following too many
		while len(self.followed_queue) >= self.max_followed:
			while len(self.hour_unfollows) >= self.max_hour_follows:
				print("Too many unfollows in 1 hour ("+
					str(len(self.hour_unfollows))+"), sleep 10 minutes")
				sleep(600)
			unfollow_id = self.followed_queue.get()
			self.send_request(self.api.unfollow, unfollow_id)
			self.hour_unfollows.put(unfollow_id)

	def run(self):
		while True:

			self.unfollow_users()

			tag = random.choice(tag_list)
			items = self.get_tag_feed(tag)
			if items is None:
				print("Tag Feed 404")
			else:
				for i,item in enumerate(items['items']):
					# change tag
					if i>0: break

					user_id = item['user']['pk']
					if self.is_user_target(item):
						self.target_user(user_id)

# usage: python3 instabot.py <USERNAME>
if __name__ == '__main__':
	username = sys.argv[1]
	password = open(username + '/pass.txt').readline().rstrip('\n')
	tag_list = [line.rstrip('\n') for line in open(username + '/tags.txt').readlines()]
	bot = InstaBot(username, password, tag_list)
	bot.run()