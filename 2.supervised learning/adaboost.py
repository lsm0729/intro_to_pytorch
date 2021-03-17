

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#model = AdaBoostClassifier()
#model.fit()
#model.predict()

# TODO: When we define the model, we can specify the hyperparameters. In practice, the most common ones are
#
# TODO: base_estimator: The model utilized for the weak learners
#  (Warning: Don't forget to import the model that you decide to use for the weak learner).
#
#TODO: n_estimators: The maximum number of weak learners used.
# For example, here we define a model which uses decision trees of max_depth 2 as the weak learners,
# and it allows a maximum of 4 of them.
# model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators = 4)


bagging = BaggingClassifier(n_estimators = 200)
randomforest = RandomForestClassifier()
Ada = AdaBoostClassifier()

