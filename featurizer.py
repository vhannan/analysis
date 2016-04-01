import os

import numpy as np
import pandas as pd
from user_agents import parse

# Define a Featurizer class that will take a set of clickstream rows by user and
# convert it into a feature vector
class ClickstreamFeaturizer(object):
    """
    The ClicksteamFeaturizer class converts a set of clickstream, session and/or
    search rows from a single user into a feature matrix suitable for building
    predictive models
    """
    def __init__(self, ref_utm_params, page_keys):
        self.utm_sources = ref_utm_params['utm_source_cat']
        self.utm_mediums = ref_utm_params['utm_medium_cat']
        self.utm_campaigns = ref_utm_params['utm_campaign_cat']
        self.referrer = ref_utm_params['referrer_cleaned']
        self.referrer_cat = ref_utm_params['referrer_cat']
        self.ref_utm_features = self.utm_sources + self.utm_mediums + \
                        self.utm_campaigns + self.referrer + self.referrer_cat
        self.ref_utm_keys = ref_utm_params.keys()
        self.page_keys = page_keys

    def get_feature_names(self):
        """
        Ouput: Attribute names for constructed feature matrix
        """
        names = []
        _ = None
        for fname, function in ClickstreamFeaturizer.__dict__.iteritems():
            if '__' not in fname and fname not in ['get_feature_names', 'featurize']:
                names += function(self, _, get_names=True)
        return names

    def featurize(self, clickstream_group):
        """
        Input: All session, pageview and search data for users or sessions.
        This function iterates through all class methods, and bypasses having to
        call each one separately. Dataset specifies the data source to accomodate
        different levels of granularity
        Output: Feature matrix where each row captures the behavior of a single user
        or session, and each column corresponds to a specific behavioral attribute
        """
        features = []
        for fname, function in ClickstreamFeaturizer.__dict__.iteritems():
            if '__' not in fname and fname not in ['get_feature_names', 'featurize']:
                features += function(self, clickstream_group)
        return features

    def session_user_agent(self, clickstream_group, get_names=False):
        """
        Input: all session data from a user, including all user_agents afiliated
        with the user_id
        Output: boolean vector defining user agent and platform (desktop,
        mobile web or app) Users will be defined by the user agent they use most
        often to visit the site
        is_app will take a value of 1 for any site visit regardless of frequency
        """
        if get_names:
            return ['is_pc','is_tablet','is_mobile','is_iphone','is_android', 'is_app']
        features = np.zeros(6, dtype=np.int)
        for ua_string, protocol in clickstream_group[['user_agent','page_protocol']].itertuples(index=False):
            agent = parse(ua_string)
            is_iphone = True if 'iPhone' in str(agent) else False
            is_android = True if 'Android' in str(agent) else False
            try:
                is_app = True if protocol.startswith('file') and agent.is_mobile else False
            except:
                is_app = False
            temp_features = np.array([agent.is_pc, agent.is_tablet, agent.is_mobile, is_iphone, is_android, is_app], dtype=np.int)
            features += temp_features
        used_app = 1 if features[5] > 0 else 0
        most_popular = max(features)
        final_features = (features.astype(float) / float(most_popular)).round()
        final_features[5] = used_app
        return final_features.tolist()

    def session_ref_utm(self, clickstream_group, get_names=False):
        """
        Input: all session data from a user, including utm keys attached to a
        cookie on the first landing
        Output: boolean vector with a length equivalent to the number of
        unique utm sources, contents, mediums, and campaigns for the dataset.
        If a user ever has a utm parameter attached to his or her cookie,
        the corresponding feature will take a 1
        """
        if get_names:
            return self.ref_utm_features
        features = [0] * len(self.ref_utm_features)
        for utm_category in self.ref_utm_keys:
            utm_parameter_list = clickstream_group[utm_category].dropna().tolist()
            for utm_parameter in utm_parameter_list:
                if utm_parameter in self.ref_utm_features:
                    features[self.ref_utm_features.index(utm_parameter)] = 1
            return features

    def clickstream_page_visits(self, clickstream_group, get_names=False):
        """
        Input: all clicksteam rows from one user
        Output: weighted pageview vector capturing the percent of time a user spent
        on each page per session
        """
        if get_names:
            name_list = ['{0}'.format(pk.replace(' ','_')) for pk in self.page_keys]
            name_list.append('total_pageviews')
            return name_list
        features = np.zeros(len(self.page_keys))
        running_total = 0
        for i, page_cat in enumerate(self.page_keys):
            count = (clickstream_group['page_category'] == page_cat).sum()
            running_total += count
            features[i] = count
        features = np.round(features / running_total, 4)
        final_features = np.append(features, running_total)
        return final_features.tolist()

    def session_number(self, clickstream_group, get_names=False):
        """
        Computes number of sessions a single user takes to book an appointment
        """
        if get_names:
            return ['num_sessions']
        features = [clickstream_group['session_id'].unique().shape[0]]
        return features

    def search(self, clickstream_group, get_names=False):
        """
        Input: Disaggregate search and click data for a single user
        Output: Vector of key search parameters
        """
        if get_names:
            return ['num_searches', 'searched_pro', 'searched_pro_first', 'pros_clicked']
        num_searches = clickstream_group['query'].unique().shape[0]
        searched_pro = 1 if 'pro_or_salon' in clickstream_group['category'].unique() else 0
        searched_pro_first = 1 if 'pro_or_salon' in clickstream_group['category']. \
            loc[clickstream_group['creation_time'] == clickstream_group['creation_time'].min()].tolist() else 0
        clickstream_temp = clickstream_group.loc[clickstream_group['clicklog_count'] > 0].reset_index()
        pros_clicked = clickstream_temp['provider_id'].unique().shape[0]
        features = [num_searches, searched_pro, searched_pro_first, pros_clicked]
        return features
