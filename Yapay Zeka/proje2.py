{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "B    357\n",
      "M    212\n",
      "Name: target, dtype: int64\n",
      "569\n",
      "   target  radius_mean  texture_mean  perimeter_mean  area_mean  \\\n",
      "0       1        17.99         10.38          122.80     1001.0   \n",
      "1       1        20.57         17.77          132.90     1326.0   \n",
      "2       1        19.69         21.25          130.00     1203.0   \n",
      "3       1        11.42         20.38           77.58      386.1   \n",
      "4       1        20.29         14.34          135.10     1297.0   \n",
      "\n",
      "   smoothness_mean  compactness_mean  concavity_mean  concave points_mean  \\\n",
      "0          0.11840           0.27760          0.3001              0.14710   \n",
      "1          0.08474           0.07864          0.0869              0.07017   \n",
      "2          0.10960           0.15990          0.1974              0.12790   \n",
      "3          0.14250           0.28390          0.2414              0.10520   \n",
      "4          0.10030           0.13280          0.1980              0.10430   \n",
      "\n",
      "   symmetry_mean  ...  radius_worst  texture_worst  perimeter_worst  \\\n",
      "0         0.2419  ...         25.38          17.33           184.60   \n",
      "1         0.1812  ...         24.99          23.41           158.80   \n",
      "2         0.2069  ...         23.57          25.53           152.50   \n",
      "3         0.2597  ...         14.91          26.50            98.87   \n",
      "4         0.1809  ...         22.54          16.67           152.20   \n",
      "\n",
      "   area_worst  smoothness_worst  compactness_worst  concavity_worst  \\\n",
      "0      2019.0            0.1622             0.6656           0.7119   \n",
      "1      1956.0            0.1238             0.1866           0.2416   \n",
      "2      1709.0            0.1444             0.4245           0.4504   \n",
      "3       567.7            0.2098             0.8663           0.6869   \n",
      "4      1575.0            0.1374             0.2050           0.4000   \n",
      "\n",
      "   concave points_worst  symmetry_worst  fractal_dimension_worst  \n",
      "0                0.2654          0.4601                  0.11890  \n",
      "1                0.1860          0.2750                  0.08902  \n",
      "2                0.2430          0.3613                  0.08758  \n",
      "3                0.2575          0.6638                  0.17300  \n",
      "4                0.1625          0.2364                  0.07678  \n",
      "\n",
      "[5 rows x 31 columns]\n",
      "Data shape  (569, 31)\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 569 entries, 0 to 568\n",
      "Data columns (total 31 columns):\n",
      "target                     569 non-null int64\n",
      "radius_mean                569 non-null float64\n",
      "texture_mean               569 non-null float64\n",
      "perimeter_mean             569 non-null float64\n",
      "area_mean                  569 non-null float64\n",
      "smoothness_mean            569 non-null float64\n",
      "compactness_mean           569 non-null float64\n",
      "concavity_mean             569 non-null float64\n",
      "concave points_mean        569 non-null float64\n",
      "symmetry_mean              569 non-null float64\n",
      "fractal_dimension_mean     569 non-null float64\n",
      "radius_se                  569 non-null float64\n",
      "texture_se                 569 non-null float64\n",
      "perimeter_se               569 non-null float64\n",
      "area_se                    569 non-null float64\n",
      "smoothness_se              569 non-null float64\n",
      "compactness_se             569 non-null float64\n",
      "concavity_se               569 non-null float64\n",
      "concave points_se          569 non-null float64\n",
      "symmetry_se                569 non-null float64\n",
      "fractal_dimension_se       569 non-null float64\n",
      "radius_worst               569 non-null float64\n",
      "texture_worst              569 non-null float64\n",
      "perimeter_worst            569 non-null float64\n",
      "area_worst                 569 non-null float64\n",
      "smoothness_worst           569 non-null float64\n",
      "compactness_worst          569 non-null float64\n",
      "concavity_worst            569 non-null float64\n",
      "concave points_worst       569 non-null float64\n",
      "symmetry_worst             569 non-null float64\n",
      "fractal_dimension_worst    569 non-null float64\n",
      "dtypes: float64(30), int64(1)\n",
      "memory usage: 137.9 KB\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1000x1000 with 4 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1000x1000 with 4 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 4058.87x4000 with 272 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 4058.87x4000 with 272 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Score:  0.9532163742690059\n",
      "CM:  [[108   1]\n",
      " [  7  55]]\n",
      "Basic KNN Acc:  0.9532163742690059\n",
      "\n",
      "Best training score: 0.9672544080604534 with parameters: {'n_neighbors': 4, 'weights': 'uniform'}\n",
      "\n",
      "Test Score: 0.9590643274853801, Train Score: 0.9773299748110831\n",
      "\n",
      "CM Test:  [[107   2]\n",
      " [  5  57]]\n",
      "CM Train:  [[248   0]\n",
      " [  9 140]]\n",
      "\n",
      "Best training score: 0.9420654911838791 with parameters: {'n_neighbors': 9, 'weights': 'uniform'}\n",
      "\n",
      "Test Score: 0.9239766081871345, Train Score: 0.947103274559194\n",
      "\n",
      "CM Test:  [[103   6]\n",
      " [  7  55]]\n",
      "CM Train:  [[241   7]\n",
      " [ 14 135]]\n",
      "\n",
      "Best training score: 0.9874055415617129 with parameters: {'n_neighbors': 1, 'weights': 'uniform'}\n",
      "\n",
      "Test Score: 0.9941520467836257, Train Score: 1.0\n",
      "\n",
      "CM Test:  [[108   1]\n",
      " [  0  62]]\n",
      "CM Train:  [[248   0]\n",
      " [  0 149]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x1a1a6b94d0>"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.colors import ListedColormap\n",
    "\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix\n",
    "from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "# warning library\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "data = pd.read_csv(\"cancer.csv\")\n",
    "data.drop(['Unnamed: 32','id'], inplace = True, axis = 1)\n",
    "\n",
    "data = data.rename(columns = {\"diagnosis\":\"target\"})\n",
    "\n",
    "sns.countplot(data[\"target\"])\n",
    "print(data.target.value_counts())\n",
    "\n",
    "data[\"target\"] = [1 if i.strip() == \"M\" else 0 for i in data.target]\n",
    "\n",
    "print(len(data))\n",
    "\n",
    "print(data.head())\n",
    "\n",
    "print(\"Data shape \", data.shape)\n",
    "\n",
    "data.info()\n",
    "\n",
    "describe = data.describe()\n",
    "\n",
    "\"\"\"\n",
    "standardization\n",
    "missing value: none\n",
    "\"\"\"\n",
    "\n",
    "# %% EDA\n",
    "\n",
    "# Correlation\n",
    "corr_matrix = data.corr()\n",
    "sns.clustermap(corr_matrix, annot = True, fmt = \".2f\")\n",
    "plt.title(\"Correlation Between Features\")\n",
    "plt.show()\n",
    "\n",
    "# \n",
    "threshold = 0.5\n",
    "filtre = np.abs(corr_matrix[\"target\"]) > threshold\n",
    "corr_features = corr_matrix.columns[filtre].tolist()\n",
    "sns.clustermap(data[corr_features].corr(), annot = True, fmt = \".2f\")\n",
    "plt.title(\"Correlation Between Features w Corr Threshold 0.75\")\n",
    "\n",
    "\"\"\"\n",
    "there some correlated features\n",
    "\"\"\"\n",
    "\n",
    "# box plot \n",
    "data_melted = pd.melt(data, id_vars = \"target\",\n",
    "                      var_name = \"features\",\n",
    "                      value_name = \"value\")\n",
    "\n",
    "plt.figure()\n",
    "sns.boxplot(x = \"features\", y = \"value\", hue = \"target\", data = data_melted)\n",
    "plt.xticks(rotation = 90)\n",
    "plt.show()\n",
    "\n",
    "\"\"\"\n",
    "standardization-normalization\n",
    "\"\"\"\n",
    "\n",
    "# pair plot \n",
    "sns.pairplot(data[corr_features], diag_kind = \"kde\", markers = \"+\",hue = \"target\")\n",
    "plt.show()\n",
    "\n",
    "\"\"\"\n",
    "skewness\n",
    "\"\"\"\n",
    "\n",
    "# %% outlier\n",
    "y = data.target\n",
    "x = data.drop([\"target\"],axis = 1)\n",
    "columns = x.columns.tolist()\n",
    "\n",
    "clf = LocalOutlierFactor()\n",
    "y_pred = clf.fit_predict(x)\n",
    "X_score = clf.negative_outlier_factor_\n",
    "\n",
    "outlier_score = pd.DataFrame()\n",
    "outlier_score[\"score\"] = X_score\n",
    "\n",
    "# threshold\n",
    "threshold = -2.5\n",
    "filtre = outlier_score[\"score\"] < threshold\n",
    "outlier_index = outlier_score[filtre].index.tolist()\n",
    "\n",
    "\n",
    "plt.figure()\n",
    "plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1],color = \"blue\", s = 50, label = \"Outliers\")\n",
    "plt.scatter(x.iloc[:,0], x.iloc[:,1], color = \"k\", s = 3, label = \"Data Points\")\n",
    "\n",
    "radius = (X_score.max() - X_score)/(X_score.max() - X_score.min())\n",
    "outlier_score[\"radius\"] = radius\n",
    "plt.scatter(x.iloc[:,0], x.iloc[:,1], s = 1000*radius, edgecolors = \"r\",facecolors = \"none\", label = \"Outlier Scores\")\n",
    "plt.legend()\n",
    "plt.show()\n",
    "\n",
    "# drop outliers\n",
    "x = x.drop(outlier_index)\n",
    "y = y.drop(outlier_index).values\n",
    "\n",
    "# %% Train test split\n",
    "test_size = 0.3\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_size, random_state = 42)\n",
    "\n",
    "# %% \n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)\n",
    "\n",
    "X_train_df = pd.DataFrame(X_train, columns = columns)\n",
    "X_train_df_describe = X_train_df.describe()\n",
    "X_train_df[\"target\"] = Y_train\n",
    "# box plot \n",
    "data_melted = pd.melt(X_train_df, id_vars = \"target\",\n",
    "                      var_name = \"features\",\n",
    "                      value_name = \"value\")\n",
    "\n",
    "plt.figure()\n",
    "sns.boxplot(x = \"features\", y = \"value\", hue = \"target\", data = data_melted)\n",
    "plt.xticks(rotation = 90)\n",
    "plt.show()\n",
    "\n",
    "\n",
    "# pair plot \n",
    "sns.pairplot(X_train_df[corr_features], diag_kind = \"kde\", markers = \"+\",hue = \"target\")\n",
    "plt.show()\n",
    "\n",
    "\n",
    "# %% Basic KNN Method\n",
    "\n",
    "knn = KNeighborsClassifier(n_neighbors = 2)\n",
    "knn.fit(X_train, Y_train)\n",
    "y_pred = knn.predict(X_test)\n",
    "cm = confusion_matrix(Y_test, y_pred)\n",
    "acc = accuracy_score(Y_test, y_pred)\n",
    "score = knn.score(X_test, Y_test)\n",
    "print(\"Score: \",score)\n",
    "print(\"CM: \",cm)\n",
    "print(\"Basic KNN Acc: \",acc)\n",
    "\n",
    "# %% choose best parameters\n",
    "\n",
    "def KNN_Best_Params(x_train, x_test, y_train, y_test):\n",
    "    \n",
    "    k_range = list(range(1,31))\n",
    "    weight_options = [\"uniform\",\"distance\"]\n",
    "    print()\n",
    "    param_grid = dict(n_neighbors = k_range, weights = weight_options)\n",
    "    \n",
    "    knn = KNeighborsClassifier()\n",
    "    grid = GridSearchCV(knn, param_grid, cv = 10, scoring = \"accuracy\")\n",
    "    grid.fit(x_train, y_train)\n",
    "    \n",
    "    print(\"Best training score: {} with parameters: {}\".format(grid.best_score_, grid.best_params_))\n",
    "    print()\n",
    "    \n",
    "    knn = KNeighborsClassifier(**grid.best_params_)\n",
    "    knn.fit(x_train, y_train)\n",
    "    \n",
    "    y_pred_test = knn.predict(x_test)\n",
    "    y_pred_train = knn.predict(x_train)\n",
    "    \n",
    "    cm_test = confusion_matrix(y_test, y_pred_test)\n",
    "    cm_train = confusion_matrix(y_train, y_pred_train)\n",
    "    \n",
    "    acc_test = accuracy_score(y_test, y_pred_test)\n",
    "    acc_train = accuracy_score(y_train, y_pred_train)\n",
    "    print(\"Test Score: {}, Train Score: {}\".format(acc_test, acc_train))\n",
    "    print()\n",
    "    print(\"CM Test: \",cm_test)\n",
    "    print(\"CM Train: \",cm_train)\n",
    "    \n",
    "    return grid\n",
    "    \n",
    "    \n",
    "grid = KNN_Best_Params(X_train, X_test, Y_train, Y_test)\n",
    "\n",
    "# %% PCA\n",
    "\n",
    "scaler = StandardScaler()\n",
    "x_scaled = scaler.fit_transform(x)\n",
    "\n",
    "pca = PCA(n_components = 2)\n",
    "pca.fit(x_scaled)\n",
    "X_reduced_pca = pca.transform(x_scaled)\n",
    "pca_data = pd.DataFrame(X_reduced_pca, columns = [\"p1\",\"p2\"])\n",
    "pca_data[\"target\"] = y\n",
    "sns.scatterplot(x = \"p1\", y = \"p2\", hue = \"target\", data = pca_data)\n",
    "plt.title(\"PCA: p1 vs p2\")\n",
    "\n",
    "\n",
    "X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_reduced_pca, y, test_size = test_size, random_state = 42)\n",
    "\n",
    "grid_pca = KNN_Best_Params(X_train_pca, X_test_pca, Y_train_pca, Y_test_pca)\n",
    "\n",
    "# visualize \n",
    "cmap_light = ListedColormap(['orange',  'cornflowerblue'])\n",
    "cmap_bold = ListedColormap(['darkorange', 'darkblue'])\n",
    "\n",
    "h = .05 # step size in the mesh\n",
    "X = X_reduced_pca\n",
    "x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1\n",
    "y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1\n",
    "xx, yy = np.meshgrid(np.arange(x_min, x_max, h),\n",
    "                     np.arange(y_min, y_max, h))\n",
    "\n",
    "Z = grid_pca.predict(np.c_[xx.ravel(), yy.ravel()])\n",
    "\n",
    "# Put the result into a color plot\n",
    "Z = Z.reshape(xx.shape)\n",
    "plt.figure()\n",
    "plt.pcolormesh(xx, yy, Z, cmap=cmap_light)\n",
    "\n",
    "# Plot also the training points\n",
    "plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,\n",
    "            edgecolor='k', s=20)\n",
    "plt.xlim(xx.min(), xx.max())\n",
    "plt.ylim(yy.min(), yy.max())\n",
    "plt.title(\"%i-Class classification (k = %i, weights = '%s')\"\n",
    "          % (len(np.unique(y)),grid_pca.best_estimator_.n_neighbors, grid_pca.best_estimator_.weights))\n",
    "\n",
    "\n",
    "#%% NCA\n",
    "\n",
    "nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state = 42)\n",
    "nca.fit(x_scaled, y)\n",
    "X_reduced_nca = nca.transform(x_scaled)\n",
    "nca_data = pd.DataFrame(X_reduced_nca, columns = [\"p1\",\"p2\"])\n",
    "nca_data[\"target\"] = y\n",
    "sns.scatterplot(x = \"p1\",  y = \"p2\", hue = \"target\", data = nca_data)\n",
    "plt.title(\"NCA: p1 vs p2\")\n",
    "\n",
    "X_train_nca, X_test_nca, Y_train_nca, Y_test_nca = train_test_split(X_reduced_nca, y, test_size = test_size, random_state = 42)\n",
    "\n",
    "grid_nca = KNN_Best_Params(X_train_nca, X_test_nca, Y_train_nca, Y_test_nca)\n",
    "\n",
    "# visualize \n",
    "cmap_light = ListedColormap(['orange',  'cornflowerblue'])\n",
    "cmap_bold = ListedColormap(['darkorange', 'darkblue'])\n",
    "\n",
    "h = .2 # step size in the mesh\n",
    "X = X_reduced_nca\n",
    "x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1\n",
    "y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1\n",
    "xx, yy = np.meshgrid(np.arange(x_min, x_max, h),\n",
    "                     np.arange(y_min, y_max, h))\n",
    "\n",
    "Z = grid_nca.predict(np.c_[xx.ravel(), yy.ravel()])\n",
    "\n",
    "# Put the result into a color plot\n",
    "Z = Z.reshape(xx.shape)\n",
    "plt.figure()\n",
    "plt.pcolormesh(xx, yy, Z, cmap=cmap_light)\n",
    "\n",
    "# Plot also the training points\n",
    "plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,\n",
    "            edgecolor='k', s=20)\n",
    "plt.xlim(xx.min(), xx.max())\n",
    "plt.ylim(yy.min(), yy.max())\n",
    "plt.title(\"%i-Class classification (k = %i, weights = '%s')\"\n",
    "          % (len(np.unique(y)),grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights))\n",
    "\n",
    "# %% find wrong decision\n",
    "knn = KNeighborsClassifier(**grid_nca.best_params_)\n",
    "knn.fit(X_train_nca,Y_train_nca)\n",
    "y_pred_nca = knn.predict(X_test_nca)\n",
    "acc_test_nca = accuracy_score(y_pred_nca,Y_test_nca)\n",
    "knn.score(X_test_nca,Y_test_nca)\n",
    "\n",
    "test_data = pd.DataFrame()\n",
    "test_data[\"X_test_nca_p1\"] = X_test_nca[:,0]\n",
    "test_data[\"X_test_nca_p2\"] = X_test_nca[:,1]\n",
    "test_data[\"y_pred_nca\"] = y_pred_nca\n",
    "test_data[\"Y_test_nca\"] = Y_test_nca\n",
    "\n",
    "plt.figure()\n",
    "sns.scatterplot(x=\"X_test_nca_p1\", y=\"X_test_nca_p2\", hue=\"Y_test_nca\",data=test_data)\n",
    "\n",
    "diff = np.where(y_pred_nca!=Y_test_nca)[0]\n",
    "plt.scatter(test_data.iloc[diff,0],test_data.iloc[diff,1],label = \"Wrong Classified\",alpha = 0.2,color = \"red\",s = 1000)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([106])"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "diff"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 0, 0, ..., 1, 1, 1],\n",
       "       [0, 0, 0, ..., 1, 1, 1],\n",
       "       [0, 0, 0, ..., 1, 1, 1],\n",
       "       ...,\n",
       "       [0, 0, 0, ..., 1, 1, 1],\n",
       "       [0, 0, 0, ..., 1, 1, 1],\n",
       "       [0, 0, 0, ..., 1, 1, 1]])"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Z"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, \"2-Class classification (k = 1, weights = 'uniform')\")"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.title(\"%i-Class classification (k = %i, weights = '%s')\"\n",
    "          % (len(np.unique(y)),grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, \"2-Class classification (k = 1, weights = 'uniform')\")"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.xlim(xx.min(), xx.max())\n",
    "plt.ylim(yy.min(), yy.max())\n",
    "plt.title(\"%i-Class classification (k = %i, weights = '%s')\"\n",
    "          % (len(np.unique(y)),grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, \"2-Class classification (k = 1, weights = 'uniform')\")"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,\n",
    "            edgecolor='k', s=20)\n",
    "plt.xlim(xx.min(), xx.max())\n",
    "plt.ylim(yy.min(), yy.max())\n",
    "plt.title(\"%i-Class classification (k = %i, weights = '%s')\"\n",
    "          % (len(np.unique(y)),grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.QuadMesh at 0x1a203ca090>"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "Z = Z.reshape(xx.shape)\n",
    "plt.figure()\n",
    "plt.pcolormesh(xx, yy, Z, cmap=cmap_light)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "cmap_light = ListedColormap(['orange',  'cornflowerblue'])\n",
    "cmap_bold = ListedColormap(['darkorange', 'darkblue'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.colors.ListedColormap at 0x1a1920c710>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cmap_light"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.colors.ListedColormap at 0x1a1920cb50>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cmap_bold"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
