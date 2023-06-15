#Imports requis

from random import sample
from numpy import sqrt, mean, logspace, abs, arange
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from pandas import DataFrame
from sklearn import linear_model, metrics, kernel_ridge, dummy
from sklearn.ensemble import RandomForestRegressor
import timeit
from xgboost import XGBRegressor






#Section : Préparation et formatage des données

def suppression_aberrantes(data, colonnes, minimum, maximum):
    for c in colonnes: 
        data[c].where(data[c] >= minimum, inplace=True)
        data[c].where(data[c] <= maximum, inplace=True)
    return(data)

def separ_train_test(data, etiquette, variables, proportion):
    test_index = sample(data.index.tolist(), int(proportion*data.shape[0]))
    X_test = data.loc[test_index][variables]
    y_test = data.loc[test_index][etiquette]
    X_train = data.drop(index = test_index)[variables]
    y_train = data.drop(index = test_index)[etiquette]
    return (X_train, y_train, X_test, y_test)

def centrage_reduction(X_train, X_test = DataFrame()):
    n = X_train.shape[0]
    mean = X_train.mean()
    deviation = sqrt(((X_train-mean)**2).sum()/n)
    X_train_std = (X_train-mean)/deviation
    if X_test.empty:
        X_test_std = X_test
    else:
        X_test_std = (X_test-mean)/deviation
    return (X_train_std, X_test_std)

def indicatrice(groupe, donnee):
    if donnee == groupe:
        return 1
    else:
        return 0

def indicatrices_variables_qualitative(data):
    liste_variables = data.columns
    result = data.copy()
    for variable in liste_variables:
        liste_modalites = data[variable].unique()
        for modalite in liste_modalites:
            result[modalite] = data[variable].apply(lambda x: indicatrice(modalite,x))
    return result.drop(columns=liste_variables)

def detection_mots(values, dictionnaire):
    result = []
    for v in values:
        detection = False
        if v == v: #pour éviter les nan
            for mot in dictionnaire:
                if mot in v:
                    detection = True
        result.append(detection)
    return result

def most_common_words(labels):
    words = []
    for lab in labels:
        words += lab.split(" ")
    counter = Counter(words)
    for word in counter.most_common(100):
        print(word)

def correction_outliers_IQR(data, colonnes):
    quartiles = data[colonnes].quantile(q = [.25,.75], axis = 'rows')
    bords = pd.DataFrame(index = ['IQR', 'minimum', 'maximum'], columns = colonnes)
    bords.loc['IQR'] = 1.5*(quartiles.loc[0.75]-quartiles.loc[0.25])
    bords.loc['minimum'] = quartiles.loc[0.25] - bords.loc['IQR']
    bords.loc['maximum'] = quartiles.loc[0.75] + bords.loc['IQR']
    for c in colonnes:
        data[c] = data[c].where(data[c] <= bords.loc['maximum'][c], other = bords.loc['maximum'][c])
        data[c] = data[c].where(data[c] >= bords.loc['minimum'][c], other = bords.loc['minimum'][c])
    return(data)













#Section : Analyse des variables

def ACP_cercles_correlation(pcs, n_comp, pca, axis_ranks, labels=None, importantes_var=['Poukaya'], label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        if labels[i] in importantes_var:
                            plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="red", alpha=0.5)
                        else:
                            plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def ACP_plans_factoriels(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,6))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def ACP_eboulis(pca, n):
    scree = pca.explained_variance_ratio_*100
    plt.plot([0.5,n+0.5], [100/n, 100/n], 'g-')
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

def eta_squared_ANOVA(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT

    
    
    
    
    
    
    
    
    

#Section : Baselines des modèles prédictifs

def regression_naive_base(X_train, y_train, methode = 'median'):
    dummR = dummy.DummyRegressor(strategy=methode)
    start_time = timeit.default_timer()
    dummR.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    return (dummR, elapsed)

def regression_ridge_base(X_train, y_train, a = 1):
    ridge = linear_model.Ridge()
    ridge.set_params(alpha=a)
    start_time = timeit.default_timer()
    ridge.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    return (ridge, elapsed)

def ridge_noyau_base(X_train, y_train, a = 1, noyau = 'rbf'):
    predicteur = kernel_ridge.KernelRidge(alpha=a, kernel=noyau, gamma=0.01) 
    start_time = timeit.default_timer()
    predicteur.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    return (predicteur, elapsed)
           
def random_forest_base(X_train, y_train, nbr_estimateurs = 100):
    random_forest = RandomForestRegressor(n_estimators = nbr_estimateurs) 
    start_time = timeit.default_timer()
    random_forest.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    return (random_forest, elapsed)              
                                    
def XGBoost_base(X_train, y_train, nbr_estim=100):
    XGBoost = XGBRegressor(n_estimators = nbr_estim)
    start_time = timeit.default_timer()
    XGBoost.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    return (XGBoost, elapsed)

def XGBoost_final(X_train, y_train, params):
    XGBoost = XGBRegressor(**params, seed=42)
    XGBoost.fit(X_train, y_train)
    return XGBoost

def indic_performances(modele, X_train, y_train, X_test, y_test):
    y_simul = modele.predict(X_train)
    y_pred = modele.predict(X_test)    
    rmse_train = np.sqrt(metrics.mean_squared_error(y_train, y_simul))
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2_train = metrics.r2_score(y_train, y_simul)
    r2_test = metrics.r2_score(y_test, y_pred)
    mape_train = metrics.mean_absolute_percentage_error(y_train_exp, y_simul)
    mape_test = metrics.mean_absolute_percentage_error(y_test, y_pred)
    return (rmse_train, rmse_test, r2_train, r2_test, mape_train, mape_test)                     

        
        
        
        
        
        
        
        
#Section : Optimisation des modèles prédictifs

def regression_ridge_params(X_train, y_train, X_test, y_test, min_alpha, max_alpha, n_alpha):
    alphas = logspace(min_alpha, max_alpha, n_alpha)
    ridge = linear_model.Ridge()
    coefs = []
    errors = []
    for a in alphas:
        ridge.set_params(alpha=a)
        ridge.fit(X_train, y_train)
        coefs.append(ridge.coef_)
        errors.append(mean((ridge.predict(X_test) - y_test) ** 2))
    return (alphas, coefs, errors)                     
                     
def regression_lasso_params(X_train, y_train, X_test, y_test, min_alpha, max_alpha, n_alpha):
    alphas = logspace(min_alpha, max_alpha, n_alpha)
    lasso = linear_model.Lasso(fit_intercept=False)
    coefs = []
    errors = []
    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(X_train, y_train)
        coefs.append(lasso.coef_)
        errors.append(mean(([lasso.predict(X_test)] - y_test) ** 2))
    return (alphas, coefs, errors)

def XGBoost_Grid_CV(X_train, y_train, params):
    XGBoost = XGBRegressor(seed = 42)
    clf = GridSearchCV(estimator=XGBoost, param_grid=params, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    #print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
    return (clf, elapsed)

def XGBoost_learning_curve_plot(X_train, y_train, X_test, y_test, params):
    eval_set = [(X_train, y_train), (X_test, y_test)]
    XGBoost = XGBRegressor(**params, seed=42)
    XGBoost.fit(X_train, y_train, eval_metric=["rmse", "mape"], eval_set=eval_set, verbose=0)
    results = XGBoost.evals_result()
    x_axis = range(0, len(results['validation_0']['rmse']))
    fig = plt.figure(figsize=(18, 5))
    ax = fig.add_subplot(121)
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
    ax.legend()
    ax.set_title('XGBoost training RMSE', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(122)
    ax.plot(x_axis, results['validation_0']['mape'], label='Train')
    ax.plot(x_axis, results['validation_1']['mape'], label='Test')
    ax.legend()
    ax.set_title('XGBoost training MAPE', fontsize=14, fontweight='bold')
    plt.show()
    return XGBoost

def XGBoost_early_stop(X_train, y_train, X_test, y_test, params):
    eval_set = [(X_test, y_test)]
    XGBoost = XGBRegressor(**params, seed=42)
    XGBoost.fit(X_train, y_train, early_stopping_rounds=10, eval_metric=["rmse"], eval_set=eval_set, verbose=0)
    return XGBoost