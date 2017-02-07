import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale

location = r'D:\Projects\Wine\wine.data'
dff = open(location, 'r')

da = dff.read().split()
da = [item.split(',') for item in da]
df = pd.DataFrame(da, columns=list(['Class', 'Alch', 'MAcid', 'Ash', 'AshAlcal', 'Magnesium',
                                    'Phenols', 'Flavanoids', 'NonflavPhenols', 'Proanthocyanins',
                                    'ColorIntens', 'Hue', 'OD280/OD315', 'Proline']))

classes = df['Class']
features = df.iloc[:,1:]

kf = KFold(n=178, n_folds=5, shuffle=True, random_state=42)

score = 0
index = 0
for i in range(1, 50):
    neigh = KNeighborsClassifier(i)
    current_score = cross_val_score(neigh, scoring='accuracy', cv=kf, X=features, y=classes).mean()

    if max(score, current_score) == current_score:
        index = i
        score = current_score
print(score.round(1), index)

features = scale(features)
score1 = 0
index1 = 0
for i in range(1, 50):
    neigh = KNeighborsClassifier(i)
    current_score = cross_val_score(neigh, scoring='accuracy', cv=kf, X=features, y=classes).mean()

    if max(score1, current_score) == current_score:
        index1 = i
        score1 = current_score
print(score1.round(2), index1)