from InstagramAPI.InstagramAPI import InstagramAPI
from persistqueue import SQLiteQueue
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.exceptions import NotFittedError

from time import sleep
import threading
import random, pprint, requests, datetime, sys, pickle, os, yaml, json
import numpy as np
import pandas as pd
import pymc3 as pm

# modify API slightly so we can multithread properly
# breaks configureTimelineAlbum, direct_share, 
# getTotalFollowers, getTotalFollowings, getTotalUserFeed, getTotalLikedMedia
class ModifiedInstagramAPI(InstagramAPI):
	def SendRequest(self, endpoint, post = None, login = False):
		if (not self.isLoggedIn and not login):
			raise Exception("Not logged in!\n")
			return;
		self.s.headers.update ({'Connection' : 'close',
								'Accept' : '*/*',
								'Content-type' : 'application/x-www-form-urlencoded; charset=UTF-8',
								'Cookie2' : '$Version=1',
								'Accept-Language' : 'en-US',
								'User-Agent' : self.USER_AGENT})
		if (post != None): # POST
			response = self.s.post(self.API_URL + endpoint, data=post) # , verify=False
		else: # GET
			response = self.s.get(self.API_URL + endpoint) # , verify=False

		return response
	def login(self, force = False):
		if (not self.isLoggedIn or force):
			self.s = requests.Session()
			# if you need proxy make something like this:
			# self.s.proxies = {"https" : "http://proxyip:proxyport"}
			response = self.SendRequest('si/fetch_headers/?challenge_type=signup&guid=' + self.generateUUID(False), None, True)
			if response.status_code == 200:

				data = {'phone_id'   : self.generateUUID(True),
						'_csrftoken' : response.cookies['csrftoken'],
						'username'   : self.username,
						'guid'       : self.uuid,
						'device_id'  : self.device_id,
						'password'   : self.password,
						'login_attempt_count' : '0'}

				response = self.SendRequest('accounts/login/', self.generateSignature(json.dumps(data)), True)
				if response.status_code == 200:
					self.isLoggedIn = True
					self.username_id = json.loads(response.text)["logged_in_user"]["pk"]
					self.rank_token = "%s_%s" % (self.username_id, self.uuid)
					self.token = response.cookies["csrftoken"]

					self.syncFeatures()
					self.autoCompleteUserList()
					self.timelineFeed()
					self.getv2Inbox()
					self.getRecentActivity()
					print ("Login success!\n")
					return True;


class Item:
	def __init__(self, value):
		self.value = value
		self.timestamp = datetime.datetime.now()
class SlidingWindow:
	def __init__(self, path, length = 3600, check_time = 120):
		self.length = datetime.timedelta(seconds=length)
		self.check_time = datetime.timedelta(seconds=check_time)
		self.last_check = datetime.datetime.now() - self.check_time * 2
		self.lock = threading.Lock()
		self.filepath = path+'/data.pkl'
		if not os.path.exists(path):
			os.makedirs(path)
		try:
			self._load()
		except:
			self.items = []
			self._save()
	def _save(self):
		with open(self.filepath, 'wb') as f:
			pickle.dump(self.items, f)
	def _load(self):
		with open(self.filepath, 'rb') as f:
			self.items = pickle.load(f)
	def _clean(self):
		# only clean every self.check_time
		if datetime.datetime.now() - self.last_check < self.check_time:
			return
		# remove old items
		self.lock.acquire()
		l = len(self.items)
		self.items = [i for i in self.items if (datetime.datetime.now()-i.timestamp) < self.length]
		if len(self.items) != l:
			self._save()
		self.lock.release()
	def __len__(self):
		self._clean()
		return len(self.items)
	def put(self, item):
		self._clean()
		self.lock.acquire()
		self.items += [Item(item)]
		self._save()
		self.lock.release()
	def get(self):
		self._clean()
		return [i.value for i in self.items]

class InstaBot:
	def __init__(self, username='', password='',
		tag_list=['instagram'], 
		max_hour_likes=1000, max_hour_follows=500,
		likes_per_user=3,
		mean_wait_time=1,
		max_followed=100,
		n_jobs=1, verbosity=1):

		self.username = username
		self.password = password
		self.tag_list = tag_list
		self.max_hour_likes = max_hour_likes
		self.max_hour_follows = max_hour_follows
		self.likes_per_user = likes_per_user
		self.mean_wait_time = mean_wait_time
		self.max_followed = max_followed
		self.n_jobs = n_jobs
		self.verbosity = verbosity

		self.hour_likes = SlidingWindow(username+'/hour_likes')
		self.hour_follows = SlidingWindow(username+'/hour_follows')
		self.hour_unfollows = SlidingWindow(username+'/hour_unfollows')

		self.target_data_path = username+'/target_data/data.pkl'
		if not os.path.exists(username+'/target_data'):
			os.makedirs(username+'/target_data')
		try: 
			self.target_data = pickle.load(open(self.target_data_path,'rb'))
		except Exception as e:
			self.target_data = pd.DataFrame(
				columns=['user_id','timestamp','followers','followings','follow_back','tag','likes'])

		self.target_data_lock = threading.Lock()

		self.model = LogisticRegression(solver='sag',warm_start=True)

		# MCMC traces for followback rates for each tag
		self.thetas = None
		# used for confidence threshold to target a user
		self.target_confidence_rate = np.log(2) / (2*np.log(2) - np.log(3))

		self.api = ModifiedInstagramAPI(username, password)
		self.api.login()

	def wait(self):
		t = np.random.exponential(self.mean_wait_time)
		if self.verbosity > 2:
			print("\tSleep",round(t,2),"seconds")
		sleep(t)

	def send_request(self, request, *args, **kwargs):
		self.wait()
		if self.verbosity > 1:
			print(request.__name__, args, kwargs)
		try:
			response = request(*args, **kwargs)
			if response.status_code == 200:
				return json.loads(response.text)
			else:
				if self.verbosity > 0:
					print("HTTP", response.status_code)
					print(response)
					print(response.text)
				if response.status_code in [400, 429]:
					if json.loads(response.text)['spam']:
						if self.verbosity > 0:
							print("Spam Detected, sleep 1 hour")
						sleep(60*60)
		except Exception as e:
			if self.verbosity > 0:
				print(e)
		return None

	def get_tag_feed(self, tag):
		return self.send_request(self.api.tagFeed, tag)
	def get_user_feed(self, user_id):
		return self.send_request(self.api.getUserFeed, user_id)
	def like_media(self, media_id):
		return self.send_request(self.api.like, media_id)
	def follow_user(self, user_id):
		return self.send_request(self.api.follow, user_id)
	def unfollow_user(self, user_id):
		return self.send_request(self.api.unfollow, user_id)
	def get_user_followers(self, user_id):
		ret = self.send_request(self.api.getUserFollowers, user_id)
		if ret is None:  return 0
		return len(ret['users'])
	def get_user_followings(self, user_id):
		ret = self.send_request(self.api.getUserFollowings, user_id)
		if ret is None: return 0
		return len(ret['users'])
	def followed_by(self, user_id):
		ret = self.send_request(self.api.userFriendship, user_id)
		try: 
			return ret['followed_by']
		except:
			return False


	def save_target_data(self):
		pickle.dump(self.target_data, open(self.target_data_path,'wb'), protocol=2)

	def update_target_data(self, row):
		self.target_data_lock.acquire()
		if row[list(self.target_data.columns).index('user_id')] in self.target_data['user_id']:
			self.target_data.loc[self.target_data['user_id']==user_id,:] = row
		else:
			row = pd.Series(row, index=self.target_data.columns)
			self.target_data = self.target_data.append(row, ignore_index=True)
		self.save_target_data()
		self.target_data_lock.release()

	def encode_target_data(self, x):
		x = np.reshape(x[:-2], (1,-1))
		#useful_data = self.target_data.loc[
		#	~pd.isnull(self.target_data['follow_back'])]
		#useful_tags = len(np.unique(useful_data['tag'].as_matrix()))
		#if useful_tags >= len(self.tag_list):
		#	x[:,3] = self.label_enc.transform(x[:,3])
		#	x = self.oh_enc.transform(x)
		return x

	def update_thread_local(self, thread_local):
		thread_local.followed_queue = SQLiteQueue(self.username+'/followed_users')

	def print_info(self):
		thread_local = threading.local()
		self.update_thread_local(thread_local)
		while True:
			if self.verbosity > 0:
				print(datetime.datetime.now().strftime('%x %X'))
				print("\tFollowed Users :", len(thread_local.followed_queue))
				print("\tHour Likes     :", len(self.hour_likes))
				print("\tHour Follows   :", len(self.hour_follows))
				print("\tHour Unfollows :", len(self.hour_unfollows))
				print("\tFollowback Rate:", self.get_follow_back_rate())
			sleep(15*60)

	def background_unfollows_update_followbacks(self):
		thread_local = threading.local()
		self.update_thread_local(thread_local)
		while True:
			self.update_follow_backs()
			self.update_model()
			self.fit_hierarchical_model()
			self.unfollow_users(thread_local)
			sleep(15*60)

	def update_follow_backs(self):
		to_update = self.target_data.loc[
			(datetime.datetime.now()-self.target_data['timestamp'] > datetime.timedelta(days=1)) &
			pd.isnull(self.target_data['follow_back'])]

		for index, row in to_update.iterrows(): 
			user_id = row['user_id']
			follow_back = self.followed_by(user_id)
			self.target_data.loc[index, 'follow_back'] = follow_back
		if len(to_update) > 0:
			self.save_target_data()

	def fit_hierarchical_model(self):
		def get_tag_n_Y(t):
			n = self.target_data.loc[
				(~pd.isnull(self.target_data['follow_back'])) &
				(self.target_data['tag'] == t)]
			Y = n[n['follow_back'] == True]
			return (len(n),len(Y))
		ns_Ys = list(map(get_tag_n_Y, self.tag_list))
		n,Y = map(np.array, zip(*ns_Ys))
		
		with pm.Model() as model:
			'''
			Fit hierarchical Bayes model

			alpha  beta
			    \   /
			  |----------|
			  | theta  n |
			  |   |___/  |
			  |   Y      |
			  |----------|
			'''
			alpha = pm.Exponential('alpha', lam=100)
			beta = pm.Exponential('beta', lam=100)
			theta = pm.Beta('theta', alpha=alpha, beta=beta, shape=len(n))
			obv = pm.Binomial('obv', n=n, p=theta, observed=Y)

			approx = pm.fit()
			trace = approx.sample(500)

			self.thetas = trace['theta']

	def get_model_data(self):
		useful_data = self.target_data.loc[
			~pd.isnull(self.target_data['follow_back'])]
		useful_tags = len(np.unique(useful_data['tag'].as_matrix()))
		if True:#useful_tags < len(self.tag_list):
			X = useful_data[['followers','followings']].as_matrix()
		else:
			X = useful_data[['followers','followings','likes','tag']].as_matrix()
			self.label_enc = LabelEncoder().fit(X[:,3])
			X[:,3] = self.label_enc.transform(X[:,3])
			self.oh_enc = OneHotEncoder(categorical_features=[3]).fit(X)
			X = self.oh_enc.transform(X)
		y = useful_data['follow_back'].as_matrix().astype(int)
		return X,y

	def update_model(self):
		X,y = self.get_model_data()
		if len(np.unique(y)) > 1:
			self.model.fit(X,y)

	def unfollow_users(self, thread_local):
		# unfollow if following too many
		while len(thread_local.followed_queue) >= self.max_followed:
			while len(self.hour_unfollows) >= self.max_hour_follows:
				if self.verbosity > 1:
					print("Too many unfollows in 1 hour ("+
						str(len(self.hour_unfollows))+"), stop unfollowing")
				return
			unfollow_id = thread_local.followed_queue.get()
			self.unfollow_user(unfollow_id)
			self.hour_unfollows.put(unfollow_id)

	def like_follow_users(self):
		thread_local = threading.local()
		self.update_thread_local(thread_local)
		while True:
			tag = self.select_tag()
			items = self.get_tag_feed(tag)
			if items is None:
				if self.verbosity > 1:
					print("Tag Feed 404")
			else:
				for i,item in enumerate(items['items']):
					# change tag
					if i>0: break

					user_id = item['user']['pk']
					if self.is_user_target(item, tag):
						self.target_user(user_id, thread_local)

	def select_tag(self):
		if self.thetas is None:
			return random.choice(self.tag_list)

		theta_means = np.mean(self.thetas, axis=0)
		theta_std = np.std(self.thetas, axis=0)
		theta_ucb = theta_means + theta_std

		#tag = self.tag_list[np.argmax(theta_ucb)]

		w = theta_ucb / np.sum(theta_ucb)
		tag = np.random.choice(self.tag_list, p=w)
		return tag

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

			X = self.encode_target_data([user_followers, user_followings, media_likes,tag])
			if self.verbosity > 1:
				print("Target Data",X)

			follow_back_rate = self.get_follow_back_rate()

			# target users more likely than average to follow back
			# confidence threshold for follow back:
			# frac = fraction hour like limit used
			# k, m constants
			# threshold = k / frac^m
			k = 0.1
			m = self.target_confidence_rate
			frac = len(self.hour_likes) / self.max_hour_likes
			# threshold is an exponent
			# higher threshold = lower confidence required
			# at frac=1, thresold = k
			# higher m = higher threshold for lower frac
			if frac < .9:
				self.target_confidence_rate += .1
			elif frac > .7:
				self.target_confidence_rate -= .1
			threshold = k / frac**m
			if self.verbosity > 1:
				print("Followback Rate",follow_back_rate)
				print("Confidence Threshold",follow_back_rate**threshold)
			try:
				p = self.model.predict_proba(X)[0,list(self.model.classes_).index(1)]
			except NotFittedError:
				p = 1
				
			if self.verbosity > 1:
				print("Followback Confidence",p)

			if p > follow_back_rate**threshold: 
				row = (user_id, datetime.datetime.now(), 
					user_followers, user_followings, np.nan,
					tag, media_likes)
				self.update_target_data(row)
				return True

		return False

	def get_follow_back_rate(self):
		follow_backs = len(self.target_data.loc[self.target_data['follow_back'] == True])
		no_follow_backs = len(self.target_data.loc[self.target_data['follow_back'] == False])
		if no_follow_backs == 0: 
			return 1
		return follow_backs / no_follow_backs

	def target_user(self, user_id, thread_local):
		if self.verbosity > 1:
			print("Targetting user", user_id)
		items = self.get_user_feed(user_id)
		if items is None: 
			if self.verbosity > 1:
				print("User Feed 404")
			return

		# like
		if self.max_hour_likes > 0:
			for i,user_item in enumerate(items['items']):
				if i >= self.likes_per_user: break

				user_media_id = user_item['pk']

				if not(user_item['has_liked']):
					while len(self.hour_likes) >= self.max_hour_likes:
						if self.verbosity > 1:
							print("Too many likes in 1 hour ("+
								str(len(self.hour_likes))+"), sleep 10 minutes")
						sleep(600)
					self.like_media(user_media_id)
					self.hour_likes.put(user_media_id)

		# follow
		if self.max_hour_follows > 0 and self.max_followed > 0:
			while len(self.hour_follows) >= self.max_hour_follows:
				if self.verbosity > 1:
					print("Too many follows in 1 hour ("+
						str(len(self.hour_follows))+"), sleep 10 minutes")
				sleep(600)
			self.follow_user(user_id)
			self.hour_follows.put(user_id)
			thread_local.followed_queue.put(user_id)

	def run(self):
		self.background_thread = threading.Thread(
			target=self.background_unfollows_update_followbacks)
		self.background_thread.start()
		if self.verbosity > 0:
			self.info_printer = threading.Thread(
				target=self.print_info)
			self.info_printer.start()

		self.worker_threads = []
		for i in range(self.n_jobs):
			t = threading.Thread(
				target=self.like_follow_users)
			t.start()
			self.worker_threads += [t]

# usage: python3 instabot.py <USERNAME>
if __name__ == '__main__':
	username = sys.argv[1]
	try:
		settings = yaml.load(open(username+'/settings.yml','r'))
	except Exception as e:
		settings = {}
	bot = InstaBot(**settings)
	bot.run()
