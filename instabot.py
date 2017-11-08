from InstagramAPI.InstagramAPI import InstagramAPI
from persistqueue import SQLiteQueue
from sklearn.linear_model import LogisticRegression
from time import sleep
import random, pprint, requests, datetime, sys, pickle, os, yaml
import numpy as np
import pandas as pd

class Item:
	def __init__(self, value):
		self.value = value
		self.timestamp = datetime.datetime.now()
class SlidingWindow:
	def __init__(self, length, path, check_time = 1/30):
		self.length = datetime.timedelta(hours=length)
		self.check_time = datetime.timedelta(hours=check_time)
		self.last_check = datetime.datetime.now()
		self.items = SQLiteQueue(path)
	def _clean(self):
		# only clean every self.check_time
		if datetime.datetime.now() - self.last_check < self.check_time:
			return
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
	def __init__(self, username='', password='',
		tag_list=['instagram'], 
		max_hour_likes=1000, max_hour_follows=500,
		likes_per_user=3,
		mean_wait_time=1,
		max_followed=100):

		self.username = username
		self.password = password
		self.tag_list = tag_list
		self.max_hour_likes = max_hour_likes
		self.max_hour_follows = max_hour_follows
		self.likes_per_user = likes_per_user
		self.mean_wait_time = mean_wait_time
		self.max_followed = max_followed

		self.followed_queue = SQLiteQueue(username+'/followed_users')
		self.hour_likes = SlidingWindow(1, username+'/hour_likes')
		self.hour_follows = SlidingWindow(1, username+'/hour_follows')
		self.hour_unfollows = SlidingWindow(1, username+'/hour_unfollows')

		self.target_data_path = username+'/target_data/data.pkl'
		if not os.path.exists(username+'/target_data'):
			os.makedirs(username+'/target_data')
		try: 
			self.target_data = pickle.load(open(self.target_data_path,'rb'))
		except Exception as e:
			self.target_data = pd.DataFrame(
				columns=['user_id','timestamp','followers','followings','follow_back','tag','likes'])

		self.api = InstagramAPI(username, password)
		self.send_request(self.api.login)

		self.train_data = None

	def wait(self):
		t = np.random.exponential(self.mean_wait_time)
		print("\tSleep",round(t,2),"seconds")
		sleep(t)

	def send_request(self, request, *args, **kwargs):
		print(request.__name__, args, kwargs)
		try:
			success = request(*args, **kwargs)
			self.wait()
			if not(success):
				status_code = self.api.LastResponse.status_code
				print("HTTP", status_code)
				print(self.api.LastResponse)
				print(self.api.LastResponse.text)
				if status_code in [400, 429]:
					if self.api.LastJson['spam']:
						print("Spam Detected, sleep 1 hour")
						sleep(60*60)
				return None
		except Exception as e:
			print(e)
			self.wait()
			return None

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

	def save_target_data(self):
		pickle.dump(self.target_data, open(self.target_data_path,'wb'), protocol=2)

	def update_target_data(self, row):
		if row[list(self.target_data.columns).index('user_id')] in self.target_data['user_id']:
			self.target_data.loc[self.target_data['user_id']==user_id,:] = row
		else:
			row = pd.Series(row, index=self.target_data.columns)
			self.target_data = self.target_data.append(row, ignore_index=True)
		self.save_target_data()

	def update_follow_backs(self):
		to_update = self.target_data.loc[
			(datetime.datetime.now()-self.target_data['timestamp'] > datetime.timedelta(days=1)) &
			~(pd.isnull(self.target_data['follow_back']))]

		for index, row in to_update.iterrows(): 
			user_id = row['user_id']
			follow_back = self.get_friendship_info(target)['followed_by']
			self.target_data.loc[index, 'follow_back'] = follow_back
		if len(to_update) > 0:
			self.save_target_data()

	def get_model_data(self):
		useful_data = self.target_data.loc[
			~pd.isnull(self.target_data['follow_back']),
			['followers','followings','likes','follow_back']]
		X = useful_data[['followers','followings','likes']].as_matrix()
		y = useful_data['follow_back'].as_matrix().astype(int)
		return X,y

	def get_model(self):
		X,y = self.get_model_data()
		if len(np.unique(y)) > 1:
			model = LogisticRegression().fit(X,y)
			model._train_data = (X,y)
			return model
		else:
			return None

	def get_mean_predicted_follow_back_rate(self, model):
		if model is None:
			return 0
		X,y = model._train_data
		y_pred = model.predict_proba(X)[:,list(model.classes_).index(1)]
		return np.mean(y_pred)

	def is_user_target(self, item, tag):
		'''
		Reinforcement learning inspired decision
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
			follow_back_rate = self.get_mean_predicted_follow_back_rate(model)

			# target users more likely than average to follow back
			alpha = follow_back_rate
			epsilon = 0.1
			p = 1 if model is None else model.predict_proba(X)[0,list(model.classes_).index(1)]
			print("Target Data",X)
			print("Predicted Followback Probability",p)
			print("Average Predicted Followback Rate",follow_back_rate)

			if p > alpha or random.random() < epsilon: 
				row = (user_id, datetime.datetime.now(), 
					user_followers, user_followings, np.nan,
					tag, media_likes)
				self.update_target_data(row)
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
			self.update_follow_backs()

			tag = random.choice(tag_list)
			items = self.get_tag_feed(tag)
			if items is None:
				print("Tag Feed 404")
			else:
				for i,item in enumerate(items['items']):
					# change tag
					if i>0: break

					user_id = item['user']['pk']
					if self.is_user_target(item, tag):
						self.target_user(user_id)

# usage: python3 instabot.py <USERNAME>
if __name__ == '__main__':
	username = sys.argv[1]
	try:
		settings = yaml.load(open(username+'/settings.yml','r'))
	except Exception as e:
		settings = {}
	bot = InstaBot(**settings)
	bot.run()