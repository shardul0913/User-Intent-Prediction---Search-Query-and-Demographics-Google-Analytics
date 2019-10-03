from copy import deepcopy
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from sklearn import preprocessing
from nltk.corpus import stopwords
from sklearn.preprocessing import scale
from nltk.stem import PorterStemmer
import nltk as nltk
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


class KeyProcess:

	def cluster(self):
		plt.style.use('ggplot')
		df_offers_first = pd.read_excel('D:\\keywords\\All events.xlsx')
		original_df = df_offers_first
		matrix = pd.DataFrame.copy(df_offers_first)
		df_offers_first.drop(['Search Query', "Product List Views", "Unique Purchases", ], 1, inplace=True)
		df_offers_first.columns = ['Product Adds To Basket', "Product List Clicks", 'Product Checkouts']

		matrix_norm = pd.DataFrame.copy(df_offers_first)

		df_offers_first['index'] = range(1, len(df_offers_first) + 1)

		# df_offers_first.convert_objects(convert_numeric=True)
		df_offers_first.fillna(0, inplace=True)

		# transaction table read
		'''df_transactions = pd.read_excel("sheet_4.xlsx")
		df_transactions.columns = ["ga:adGroup", "ga:adwordsCampaignID"]
		df_transactions['n'] = 1'''
		'''
		# join the offers and transactions table
		df = pd.merge(df_offers, df_transactions)
		# create a "pivot table" which will give us the number of times each customer responded to a given offer
		matrix = df.pivot_table(index=['ga:adGroup'], columns=['ga:adwordsCampaignID'], values='n')
		# a little tidying up. fill NA values with 0 and make the index into a column
		matrix = matrix.fillna(0).reset_index()
		# save a list of the 0/1 columns. we'll use these a bit later
		x_cols = matrix.columns[1:]
		print(matrix)
		'''

		# Applying the clustering for the offers taken by the users
		cluster = KMeans(n_clusters=8)
		# slice matrix so we only include the 0/1 indicator columns in the clustering

		# normalization
		x_cols = (matrix_norm)

		original_df['cluster'] = cluster.fit_predict(x_cols)

		matrix['cluster'] = cluster.fit_predict(x_cols)

		centroid = cluster.cluster_centers_

		labels = cluster.labels_

		print(centroid)

		print ('Labels')

		print(labels)

		# print(x_cols)

		pca = PCA(n_components=1)

		original_df['x'] = df_offers_first['Product Checkouts']
		original_df['y'] = df_offers_first['Product Adds To Basket']
		original_df['z'] = df_offers_first['Product List Clicks']

		matrix['x'] = df_offers_first['index']
		matrix['y'] = pca.fit_transform(x_cols)[:]

		customer_clusters = matrix[["Search Query", "cluster", "Product List Views",
									"Product List Clicks", "Product Adds To Basket", 'Product Checkouts',
									'Unique Purchases', 'cluster']]
		customer_clusters.head()
		print(original_df)

		'''Write static excel output keyword cluster'''
		# writer = pd.ExcelWriter('D:\\keywords\\cluster_detail.xlsx', engine='xlsxwriter')
		# customer_clusters.to_excel(writer)
		# writer.save()

		centers = np.array(centroid)

		colors = 10 * ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

		fig = plt.figure()

		ax = fig.add_subplot(111, projection='3d')

		for i, row in original_df.iterrows():
			ax.scatter(row['x'], row['y'], row['z'], s=2, c=colors[int(row['cluster'])], depthshade=True)

		ax.set_xlabel('Product Checkouts')
		ax.set_ylabel('Product Adds To Basket')
		ax.set_zlabel('Product List Clicks')

		plt.show()

		return customer_clusters

		# plt.scatter(centroid[:,0], centroid[:,1],color="g", marker="x", s=2, linewidths=1)
		# plt.scatter(matrix['x'],matrix['y'],s=1)
		# plt.scatter(matrix['x1'],matrix['y1'], marker = "x", s=15, linewidths = 1, zorder = 10)


	def keyword_value(self, df, event):
		df_key_product_list_view = df[['Search Query',event]]
		df_key_product_list_view.fillna(0)
		df_ = pd.DataFrame.copy(df_key_product_list_view)
		df_final = pd.DataFrame.copy(df_key_product_list_view)
		exclude_words = {u'www',u'com'}
		keyword_list = []
		final_keywords = []
		final_count = []
		final_dict = {}
		df_final.columns = (['Search word',event])
		df_.columns = (['Search word',event])

		for i,row in df_key_product_list_view.iterrows():
			keyword_list = []
			ps = PorterStemmer()
			stop_words = set(stopwords.words("english"))
			data = row['Search Query']
			for i,data_word in enumerate(re.findall(r"[\w']+", data)):
				if data_word in exclude_words:
					continue
				elif data_word in stop_words:
					continue
				else:
					keyword_list.append(data_word)

			'''
			for word in keyword_list:
				if word not in stop_words:
					keyword_list_final.append(word)
			'''

			#all_words.append(str(keyword_list))

			for word in keyword_list:
				final_keywords.append(word)
				final_count.append(int(row[event]))
		print(keyword_list)
		#print(all_words)

		for i in range(len(final_keywords)):

			if final_keywords[i] in final_dict.keys():
				final_dict[final_keywords[i]] = final_dict[final_keywords[i]] + final_count[i]
			else:
				final_dict[final_keywords[i]] = final_count[i]
		return final_dict

	def calculate_value(self,cluster_details,event):

		features = ['Product List Views', 'Unique Purchases', 'Product List Clicks']

		df_keyword_dict = cluster_details
		print cluster_details
		# Separating out the features
		x = cluster_details.loc[:, features].values

		x = StandardScaler().fit_transform(x)

		pca = PCA(n_components=1)
		principalComponents = pca.fit_transform(x)
		principalDf = pd.DataFrame(data=principalComponents
								   , columns=['principal component 1'])

		print principalDf
		cluster_details['scaled value'] = scale(principalDf['principal component 1'],axis=0, with_mean=False, with_std=True, copy=True )

		keyword_cluster = []

		writer = pd.ExcelWriter('D:\\keywords\\opp.xlsx', engine='xlsxwriter')
		df_keyword_dict.to_excel(writer)
		writer.save()

		return df_keyword_dict
	'''
	for key_primary in df_final[str(word)]:
		for key_secondary in df_final[str(word)]:
			if key_primary == key_secondary:
				print('1')
			else:
				continue
	'''

	def compare(self, scaled_df):
		keywords_df = pd.read_excel('D:\\keywords\\cluster_detail.xlsx')
		user_df = pd.read_excel('D:\\keywords-test-pepperfry.xlsx')
		keyword_value = pd.read_excel('D:\\keywords\\Unique Purchases.xlsx')
		user_df.drop(['Day of Week',"Product List Views","Product Adds To Basket","Product List Clicks"
					  ,"Product Checkouts","Unique Purchases",'Category','City'], 1, inplace=True)
		user_df.columns = ['userID','Search Query']
		all_word_list = []
		data = user_df['Search Query']
		keywords_cluster = scaled_df
		keywords_cluster.drop(['Product List Views','Product List Clicks','Product Adds To Basket','Product Checkouts'], 1, inplace=True)
		k = 0
		keyword_dict = {}
		data_dict = user_df
		user_cluster = {}
		for i, row in user_df.iterrows():  							#The Search query of the user

			a = {}

			for z, word in keywords_df.iterrows():									#Dictionary of the clustered keywords

				if str(word['Search Query']) in row['Search Query']:						#The data_dict assigns the cluster number to the query index
					if not a:
						a = {'Query': row['Search Query'], 'cluster': word['cluster'], 'scaled value': word["scaled value"]}

					elif a['scaled value'] < word['scaled value']:

						a['cluster'] = word['cluster']
					else:
						continue
				else:
					continue
				print(a)

			if a.has_key('cluster'):

				user_cluster[row['userID']] = a['cluster']
			else:

				user_cluster[row['userID']] = 10

			print(user_cluster)

			'''
			for j, data_word in enumerate(re.findall(r"[\w']+", row)):
				#dict_keywords[i] =
				if data_word in keyword_dict.keys():
					data_dict[i] = keyword_dict[data_word]
				else:
					print('no')
			for keyword in keywords_df['Search Query']:
				print(keyword + '||' + word)
				if word is keyword:
						print('match')
				else:
						print('no match')
			'''

		print(user_cluster)

		user_cluster_df = pd.DataFrame(user_cluster.items(), columns=['userID', 'cluster'])

		writer = pd.ExcelWriter('D:\\keywords\\decision-tree-input.xlsx', engine='xlsxwriter')

		user_cluster_df.to_excel(writer)

		writer.save()

	def prep(self,df_demographics_train, df_demographics_test, df_keywords):

		df = pd.merge(df_demographics_train, df_keywords, on=['userID'])

		df["Target"] = ""

		target_list = []

		for i, row in df.iterrows():

			if row['Product Checkouts'] >= 200:
				target_list.append(7)

			elif 100 <= row['Product Checkouts'] <= 200:
				target_list.append(6)

			elif 50 <= row['Product Checkouts'] <= 100:
				target_list.append(5)

			elif 40 <= row['Product Checkouts'] <= 50:
				target_list.append(4)

			elif 30 <= row['Product Checkouts'] <= 40:
				target_list.append(3)

			elif 20 <= row['Product Checkouts'] <= 30:
				target_list.append(2)

			elif 10 <= row['Product Checkouts'] <= 20:
				target_list.append(1)

			elif 0 <= row['Product Checkouts'] <= 10:
				target_list.append(0)

		df['Target'] = target_list

		df_test = df
		df_test.drop(['Product List Views', 'Product List Clicks', 'Product Adds To Basket', 'Product Checkouts', 'Unique Purchases'], 1,
					 inplace=True)

		'''Encode the text into numbers'''
		label_encoder = LabelEncoder()
		df_test['Search Query'] = label_encoder.fit_transform(df_test['Search Query'])
		df_test['Day of Week'] = label_encoder.fit_transform(df_test['Day of Week'])
		df_test['City'] = label_encoder.fit_transform(df_test['City'])
		df_test['Category'] = label_encoder.fit_transform(df_test['Category'])

		df1 = df_test[df_test['cluster'] == 1]
		df2 = df_test[df_test['cluster'] == 2]
		df3 = df_test[df_test['cluster'] == 3]
		df4 = df_test[df_test['cluster'] == 4]
		df5 = df_test[df_test['cluster'] == 5]
		df6 = df_test[df_test['cluster'] == 6]
		df7 = df_test[df_test['cluster'] == 7]
		df8 = df_test[df_test['cluster'] == 8]

		list_of_df = [df2, df3, df4, df5, df6, df7, df8]

		for i, dfs in enumerate(list_of_df):
			features = dfs.columns[2:5]

			X_train, X_test, y_train, y_test = train_test_split(dfs[features], dfs['Target'], test_size=0.75, random_state=123456)

			clf = RandomForestClassifier(n_estimators=10, n_jobs=2, random_state=0)

			clf.fit(X_train, y_train)

			predicted = clf.predict(X_test)
			importance = clf.feature_importances_

			accuracy = accuracy_score(y_test, predicted)

			print "-------"

			print (importance)
			# print ( dfs )

			print("Accuracy of the algo - " + str(accuracy))

			print "-------"

	def __init__(self):

		df = pd.read_excel('keywords.xlsx') '''Input format given in the attachment files'''
		event = 'Product List Views'
		keyowrd_list = self.keyword_value(df, event)
		cluster_details = self.cluster()
		scaled_df = self.calculate_value(cluster_details, event)
		#self.compare(scaled_df)
		df_demographics_train = pd.read_excel('train.xlsx')
		df_demographics_test = pd.read_excel('test.xlsx')
		df_keywords = pd.read_excel('decision-tree-input.xlsx')			'''Output of the keyword analysis and Clustering'''
		self.prep(df_demographics_train, df_demographics_test, df_keywords)

KeyProcess()