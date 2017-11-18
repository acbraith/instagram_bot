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


class SlidingWindow:
	class Item:
		def __init__(self, value):
			self.value = value
			self.timestamp = datetime.datetime.now()
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
		self.items += [self.Item(item)]
		self._save()
		self.lock.release()
	def get(self):
		self._clean()
		return [i.value for i in self.items]

class PriorityQueue:
	class Item:
		def __init__(self, value, priority):
			self.value = value
			self.priority = priority
	def __init__(self, size):
		self.items = []
		self.size = size
	def __len__(self):
		return len(self.items)
	def is_full(self):
		return len(self) >= self.size
	def put(self, item, priority):
		self.items += [self.Item(item, priority)]
		if len(self) > self.size:
			idx = np.argmin([i.priority for i in self.items])
			del self.items[idx]
	def get(self):
		if len(self) == 0: return None
		idx = np.argmax([i.priority for i in self.items])
		item = self.items[idx]
		del self.items[idx]
		return item.value

class InstaBot:
	def __init__(self, username='', password='',
		tag_list=['instagram'], 
		max_hour_likes=1000, max_hour_follows=500,
		likes_per_user=3,
		mean_wait_time=1,
		max_followed=100,
		verbosity=1):

		self.username = username
		self.password = password
		self.tag_list = tag_list
		self.max_hour_likes = max_hour_likes
		self.max_hour_follows = max_hour_follows
		self.likes_per_user = likes_per_user
		self.mean_wait_time = mean_wait_time
		self.max_followed = max_followed
		self.verbosity = verbosity

		self.targets_queue = PriorityQueue(self.max_hour_follows * 2)

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

		self.api_lock = threading.Lock()

		self.api = ModifiedInstagramAPI(username, password)
		self.api.login()

	def wait(self):
		t = np.random.exponential(self.mean_wait_time)
		if self.verbosity > 2:
			print("\tSleep",round(t,2),"seconds")
		sleep(t)

	def send_request(self, request, *args, **kwargs):
		self.api_lock.acquire()
		self.wait()
		ret = None
		if self.verbosity > 1:
			print(request.__name__, args, kwargs)
		try:
			response = request(*args, **kwargs)
			if response.status_code == 200:
				ret = json.loads(response.text)
			else:
				if self.verbosity > 0:
					print("HTTP", response.status_code)
					print(response)
					if self.verbosity > 1:
						print(response.text)
				if response.status_code in [400, 429]:
					if json.loads(response.text)['spam']:
						if self.verbosity > 0:
							print("Spam Detected, sleep 1 hour")
						sleep(60*60)
		except Exception as e:
			if self.verbosity > 0:
				print(e)
		self.api_lock.release()
		return ret

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
		#self.target_data_lock.acquire()
		if row[list(self.target_data.columns).index('user_id')] in self.target_data['user_id']:
			self.target_data.loc[self.target_data['user_id']==user_id,:] = row
		else:
			row = pd.Series(row, index=self.target_data.columns)
			self.target_data = self.target_data.append(row, ignore_index=True)
		self.save_target_data()
		#self.target_data_lock.release()

	def one_hot_encode_tags(self, tags):
		tags = np.array(tags).reshape(-1)
		def tag_idx(tag):
			if tag in self.tag_list: return self.tag_list.index(tag)
			return len(self.tag_list)
		tags = [tag_idx(t) for t in tags]
		one_hot_tags = np.eye(len(self.tag_list)+1)[tags]
		return one_hot_tags

	# # # # # # # # # # 
	# Bot Workers
	# # # # # # # # # # 
	def fit_model(self):

		def update_follow_backs():
			#self.target_data_lock.acquire()
			to_update = self.target_data.loc[
				(datetime.datetime.now()-self.target_data['timestamp'] > datetime.timedelta(days=1)) &
				pd.isnull(self.target_data['follow_back'])]

			for index, row in to_update.iterrows(): 
				user_id = row['user_id']
				follow_back = self.followed_by(user_id)
				self.target_data.loc[index, 'follow_back'] = follow_back
			if len(to_update) > 0:
				self.save_target_data()
			#self.target_data_lock.release()

		def update_model():

			def get_model_data():
				useful_data = self.target_data.loc[
					~pd.isnull(self.target_data['follow_back'])]
				X = useful_data[['followers','followings']].as_matrix()
				tags = useful_data[['tag']].as_matrix()
				tags = self.one_hot_encode_tags(tags)
				X = np.append(X, tags, axis=1)
				y = useful_data['follow_back'].as_matrix().astype(int)
				return X,y

			X,y = get_model_data()
			if len(np.unique(y)) > 1:
				self.model.fit(X,y)

		while True:
			update_model()
			update_follow_backs()
			sleep(15*60)

	def print_info(self):
		followed_queue = SQLiteQueue(self.username+'/followed_users')
		while True:
			if self.verbosity > 0:
				print(datetime.datetime.now().strftime('%x %X'))
				print("\tFollowed Users   :", len(followed_queue))
				print("\tHour Likes       :", len(self.hour_likes))
				print("\tHour Follows     :", len(self.hour_follows))
				print("\tHour Unfollows   :", len(self.hour_unfollows))
				print("\tTargets Queue Len:", len(self.targets_queue))
				print("\tTargets Queue Max:", max([0]+[i.priority for i in self.targets_queue.items]))
			sleep(15*60)

	def find_targets(self):

		def select_tag():
			return random.choice(self.tag_list)

		def get_followback_confidence(user_info):
			x = [user_info['followers'], user_info['followings']]
			x = np.reshape(x,(1,-1))
			tag = self.one_hot_encode_tags([user_info['tag']])
			x = np.append(x, tag, axis=1)
			try:
				followback_confidence = \
					self.model.predict_proba(x)[0,list(self.model.classes_).index(1)]
			except NotFittedError:
				followback_confidence = 1
			return followback_confidence

		# build up self.targets_queue
		while True:
			for _ in range(self.max_hour_follows):
				tag = select_tag()
				items = self.get_tag_feed(tag)
				if items is None:
					if self.verbosity > 1:
						print("Tag Feed 404")
				else:
					for i,item in enumerate(items['items']):
						if i>5 or self.targets_queue.is_full(): break

						user_id = item['user']['pk']
						user_followers = self.get_user_followers(user_id)
						user_followings = self.get_user_followings(user_id)

						user_info = {
							'user_id':user_id,
							'followers':user_followers,
							'followings':user_followings,
							'likes':item['like_count'],
							'tag':tag,
							'discovery_time':datetime.datetime.now()}

						followback_confidence = get_followback_confidence(user_info)

						self.targets_queue.put(user_info, followback_confidence)
			sleep(15*60)

	def like_follow_unfollow(self):

		def target_users():

			def target_user(user_id):
				if self.verbosity > 1:
					print("Targetting user", user_id)
				items = self.get_user_feed(user_id)
				if items is None: 
					if self.verbosity > 1:
						print("User Feed 404")
					return

				# like
				if self.max_hour_likes > 0:
					for i,item in enumerate(items['items']):
						if i >= self.likes_per_user: break

						media_id = item['pk']

						if not(item['has_liked']):
							self.like_media(media_id)
							self.hour_likes.put(media_id)

				# follow
				if self.max_hour_follows > 0 and self.max_followed > 0:
					self.follow_user(user_id)
					self.hour_follows.put(user_id)
					followed_queue.put(user_id)

			while (len(self.hour_likes)+self.likes_per_user < self.max_hour_likes) and \
				(len(self.hour_follows)+1 < self.max_hour_follows):
				user_info = self.targets_queue.get()
				if user_info is not None:
					target_user(user_info['user_id'])

					row = (user_info['user_id'], datetime.datetime.now(), 
							user_info['followers'], user_info['followings'], np.nan,
							user_info['tag'], user_info['likes'])
					self.update_target_data(row)

		def unfollow_users():
			while (len(followed_queue) >= self.max_followed) and \
				(len(self.hour_unfollows)+1 < self.max_hour_follows):
				user_id = followed_queue.get()
				self.unfollow_user(user_id)
				self.hour_unfollows.put(user_id)

		followed_queue = SQLiteQueue(self.username+'/followed_users')
		while True:
			target_users()
			unfollow_users()
			sleep(15*60)

	def run(self):

		# model data gathering and fitting
		self.fit_model_thread = threading.Thread(
			target=self.fit_model)
		self.fit_model_thread.start()

		# print information
		if self.verbosity > 0:
			self.print_info_thread = threading.Thread(
				target=self.print_info)
			self.print_info_thread.start()

		# locate potential targets
		self.find_targets_thread = threading.Thread(
			target = self.find_targets)
		self.find_targets_thread.start()

		# likes, follows and unfollows
		self.like_follow_unfollow_thread = threading.Thread(
			target=self.like_follow_unfollow)
		self.like_follow_unfollow_thread.start()



# usage: python3 instabot.py <USERNAME>
if __name__ == '__main__':
	username = sys.argv[1]
	try:
		settings = yaml.load(open(username+'/settings.yml','r'))
	except Exception as e:
		settings = {}
	bot = InstaBot(**settings)
	bot.run()
