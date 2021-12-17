import os
import pickle
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve
from sklearn import preprocessing
import seaborn as sns
from mord import LogisticAT

class AdjustGrading(object):
    """
    Compute grading adjusted by age and sex using an Ordered Logistic Regression model
    """
    def fit(self, pheno, grading_col = 'grading'):
        """
        pheno: pd.DataFrame with columns:
            age: age in years
            gender: sex, 0 = male, 1 = female
            grading: WHO 0-5 grading
        """
        self.grading_col = grading_col
        self.model_males = LogisticAT(alpha=0)
        self.model_females = LogisticAT(alpha=0)
        inds_males = pheno['gender'] == 0
        inds_females = pheno['gender'] == 1
        print('Fitting olr using {} males'.format(np.sum(inds_males)))
        print('\t',np.unique(pheno.loc[inds_males, self.grading_col], return_counts = True))
        self.model_males.fit(pheno.loc[inds_males,'age'].astype(int).values.reshape((-1,1)), pheno.loc[inds_males,self.grading_col].astype(int).values)
        print('Fitting olr using {} females'.format(np.sum(inds_females)))
        print('\t',np.unique(pheno.loc[inds_females, self.grading_col], return_counts = True))
        self.model_females.fit(pheno.loc[inds_females,'age'].astype(int).values.reshape((-1,1)), pheno.loc[inds_females,self.grading_col].astype(int).values)
    def dump(self, model_pkl_males, model_pkl_females):
        with open(model_pkl_males, 'wb') as fout:
            pickle.dump(self.model_males, fout)
        with open(model_pkl_females, 'wb') as fout:
            pickle.dump(self.model_females, fout)
    def load(self, model_pkl_males, model_pkl_females):
        with open(model_pkl_males, 'rb') as fin:
            self.model_males = pickle.load(fin)
        with open(model_pkl_females, 'rb') as fin:
            self.model_females = pickle.load(fin)
    def predict(self, pheno):
        def f_(t):
            grading_positive = [1, 2, 3, 4, 5, 6]
            grading_negative = [-1, -2, -3, -4, -5, -6] 
            if t in grading_positive:
                return 1
            elif t in grading_negative:
                return 0
            else:
                return np.nan 
        inds_males = pheno['gender'] == 0
        pred_males = pd.Series(self.model_males.predict(pheno.loc[inds_males,'age'].astype(int).values.reshape((-1,1))), index = pheno[inds_males].index)
        delta_pred_males = pheno.loc[inds_males,self.grading_col] - pred_males
        grade_adj_males = delta_pred_males.apply(f_)
        output_males = pd.DataFrame([pred_males, delta_pred_males, grade_adj_males], index = ['olr_prediction', 'olr_delta', 'adjusted_{}'.format(self.grading_col)]).T
        inds_females = pheno['gender'] == 1
        pred_females = pd.Series(self.model_females.predict(pheno.loc[inds_females,'age'].astype(int).values.reshape((-1,1))), index = pheno[inds_females].index)
        delta_pred_females = pheno.loc[inds_females,self.grading_col] - pred_females
        grade_adj_females = delta_pred_females.apply(f_)
        output_females = pd.DataFrame([pred_females, delta_pred_females, grade_adj_females], index = ['olr_prediction', 'olr_delta', 'adjusted_{}'.format(self.grading_col)]).T
        output = pd.concat((output_males, output_females), axis = 0)
        return output
    def plot(self, pheno):
        f = plt.figure()
        ax = f.add_subplot(2,1,1)
        inds_expected = np.logical_and(pheno['olr_delta'] == 0,pheno['gender'] == 0)
        ax.plot(pheno.loc[inds_expected,'age'], pheno.loc[inds_expected,self.grading_col], 'ok', label = 'as expected')
        inds_severe = np.logical_and(pheno['olr_delta'] > 0,pheno['gender'] == 0)
        ax.plot(pheno.loc[inds_severe,'age'], pheno.loc[inds_severe,self.grading_col], 'or', label = 'more severe')
        inds_mild = np.logical_and(pheno['olr_delta'] < 0,pheno['gender'] == 0)
        ax.plot(pheno.loc[inds_mild,'age'], pheno.loc[inds_mild,self.grading_col], 'og', label = 'less severe')
        plt.xticks([], [])
        print('MALES')
        for grade, chunk in pheno.groupby(self.grading_col):
            ages = chunk.loc[(chunk['olr_delta'] == 0) & (chunk['gender'] == 0),'age'].values
            if len(ages):
                print('\tGRADE {}:{}-{}'.format(grade,np.min(ages),np.max(ages)))
        plt.title('males')
        plt.xlabel('')
        plt.ylabel('grading')
        ax = f.add_subplot(2,1,2)
        inds_expected = np.logical_and(pheno['olr_delta'] == 0,pheno['gender'] == 1)
        ax.plot(pheno.loc[inds_expected,'age'], pheno.loc[inds_expected,self.grading_col], 'ok', label = 'as expected')
        inds_severe = np.logical_and(pheno['olr_delta'] > 0,pheno['gender'] == 1)
        ax.plot(pheno.loc[inds_severe,'age'], pheno.loc[inds_severe,self.grading_col], 'or', label = 'more severe')
        inds_mild = np.logical_and(pheno['olr_delta'] < 0,pheno['gender'] == 1)
        ax.plot(pheno.loc[inds_mild,'age'], pheno.loc[inds_mild,self.grading_col], 'og', label = 'less severe')
        plt.title('female')
        plt.xlabel('age')
        plt.ylabel('grading')
        print('FEMALES')
        for grade, chunk in pheno.groupby(self.grading_col):
            ages = chunk.loc[(chunk['olr_delta'] == 0) & (chunk['gender'] == 1),'age'].values
            if len(ages):
                print('\tGRADE {}:{}-{}'.format(grade,np.min(ages),np.max(ages)))
        pdf.savefig()
        plt.close()

class SelectFeatures(object):
    """
    Estimate (likely) relevant features by Logistic Regression models with L1 regularization
    """
    def __init__(self, booleans_folder):
        """
        """
        self.test_size = 0.1 # percentage of samples in the testing set
        self.num_trials_grid = 1 # number of times the grid search is carried out (if > 1 the average of grid search results is taken)
        self.fold = 10 # number of folds for the cross-validation
        self.std_reduction = 1 # the best grid search param is the most parsimonious with score > highest_score - std_dev_highest_score / std_reduction 
        self.grid_search_metric = 'accuracy' # metric for the grid search
        self.analysis_list = [
            ['al1_ultrarare', 'X', 'female','_al1_ultrarare'], ['al1_ultrarare', 'female','_al1_ultrarare'], ['al2_ultrarare', 'female','_al2_ultrarare'],
            ['al1_just_rare', 'X', 'female','_al1_just_rare'], ['al1_just_rare', 'female','_al1_just_rare'], ['al2_just_rare', 'female','_al2_just_rare'],  
            ['al1_just_medium', 'X', 'female','_al1_just_medium'], ['al1_just_medium', 'female','_al1_just_medium'], ['al2_just_medium', 'female','_al2_just_medium'],
            ['gc_unique_hetero', 'X', 'female',''], ['gc_unique_hetero', 'female',''], ['gc_unique_homo', 'female','_homo'],

            ['al1_ultrarare', 'X', 'male','_al1_ultrarare'], ['al1_ultrarare', 'male','_al1_ultrarare'], ['al2_ultrarare', 'male','_al2_ultrarare'],
            ['al1_just_rare', 'X', 'male','_al1_just_rare'], ['al1_just_rare', 'male','_al1_just_rare'], ['al2_just_rare', 'male','_al2_just_rare'],  
            ['al1_just_medium', 'X', 'male','_al1_just_medium'], ['al1_just_medium', 'male','_al1_just_medium'], ['al2_just_medium', 'male','_al2_just_medium'],
            ['gc_unique_hetero', 'X', 'male',''], ['gc_unique_hetero', 'male',''], ['gc_unique_homo', 'male','_homo'],

            ['al1_ultrarare', 'male','female','_al1_ultrarare'], ['al2_ultrarare', 'male','female','_al2_ultrarare'],
            ['al1_just_rare', 'male','female','_al1_just_rare'], ['al2_just_rare', 'male','female','_al2_just_rare'],  
            ['al1_just_medium', 'male','female','_al1_just_medium'], ['al2_just_medium', 'male','female','_al2_just_medium'],
            ['gc_unique_hetero', 'male','female',''], ['gc_unique_homo', 'male','female','_homo'],

            ]
        self.genes_x = pd.read_csv('./list_genes_chrX.csv')['genes_x'].values
        self.booleans = {}
        for suffix in ['al1_ultrarare','al2_ultrarare','al1_just_rare','al2_just_rare','al1_just_medium','al2_just_medium','gc_unique_hetero','gc_unique_homo']:
            file_name = os.path.join(booleans_folder,'data_{}.csv'.format(suffix))
            print('Reading booleans from: {}'.format(file_name))
            d = pd.read_csv(file_name).set_index('Unnamed: 0')
            self.booleans[suffix] = d #pd.read_csv(file_name).set_index('Unnamed: 0')
            print('\tshape: {}'.format(self.booleans[suffix].shape))
    def bootstrap(self, pheno, i_seed_start, i_seed_end):
        """
        pheno: pd.DataFrame with columns:
            gender: sex, 0 = male, 1 = female
            age: age in years
            adjusted_grading: 0 = less severe than expected, 1 = more severe than expected
        i_seed_[start/end]: int, first/last+1 index of the bootstrap repetitions
        """
        from pathlib import Path
        from kneed import KneeLocator
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import auc
        seed_list = np.arange(i_seed_start, i_seed_end).tolist() # choose here the number of iterations (as many as the random seeds)
        for seed in seed_list: # cycle over all the seeds, each cycle include a complete training/testing of the model
            Path('./Results_{}'.format(seed)).mkdir(parents=True, exist_ok=True)
            for code, sex in enumerate(['male', 'female']):
                inds_2_keep = np.logical_and(pheno['gender'] == code, np.logical_not(pheno['adjusted_{}'.format(self.grading_col)].isnull()))
                print('Number of {} samples: {}'.format(sex,np.sum(inds_2_keep)))
                pheno_gender = pheno.loc[inds_2_keep]
                y_age = pd.qcut(pheno_gender['age'], 2, labels=False)
                idx_train, idx_test = train_test_split(pheno_gender, test_size=self.test_size, stratify=y_age, random_state=seed)
                idx_train.to_csv(os.path.join('Results_' + str(seed), 'train_'+ sex +'.csv'))
                idx_test.to_csv(os.path.join('Results_' + str(seed), 'test_'+ sex + '.csv'))
            for analysis in self.analysis_list: # cycle over the analyses 
                print('Running analysis:', analysis)
                if 'male' in analysis and 'female' in analysis:
                    sex = ['male', 'female']
                elif 'male' in analysis:
                    sex = ['male']
                elif 'female' in analysis:
                    sex = ['female']
                else:
                    raise ValueError()
                # read train test indices
                if sex == ['male', 'female']:
                    train_index_male = pd.read_csv(os.path.join('Results_' + str(seed) , 'train_male.csv'), index_col = 'sample').index
                    train_index_female = pd.read_csv(os.path.join('Results_' + str(seed) , 'train_female.csv'), index_col = 'sample').index
                    train_index = train_index_male.union(train_index_female)
                    test_index_male = pd.read_csv(os.path.join('Results_' + str(seed) , 'test_male.csv'), index_col = 'sample').index
                    test_index_female = pd.read_csv(os.path.join('Results_' + str(seed) , 'test_female.csv'), index_col = 'sample').index
                    test_index = test_index_male.union(test_index_female)
                elif sex == ['male'] or sex==['female']:
                    train_index = pd.read_csv(os.path.join('Results_' + str(seed) , 'train_' + sex[0] + '.csv'), index_col = 'sample').index
                    test_index = pd.read_csv(os.path.join('Results_' + str(seed) , 'test_' + sex[0] + '.csv'), index_col = 'sample').index
                features = self.booleans[analysis[0]]
                features_train = features.loc[train_index].copy()
                features_test= features.loc[test_index].copy()
                target_train = pheno.loc[train_index, 'adjusted_{}'.format(self.grading_col)]
                target_test = pheno.loc[test_index, 'adjusted_{}'.format(self.grading_col)]
                print('Number of samples in train: {} {}'.format(features_train.shape[0], len(target_train)))
                print('Number of samples in test: {}'.format(features_test.shape[0], len(target_test)))
                print('Number of genes: {} {}'.format(features_train.shape[1], features_test.shape[1]))
                # remove outliers
                if ('al1' in analysis[0]) or ('al2' in analysis[0]): 
                    numb = features_train.sum(axis = 0)
                    y, x = np.histogram(numb, bins=100)
                    cdf = np.cumsum(y)
                    x_ = x[:-1]
                    y_ = cdf/cdf[-1]
                    kneedle = KneeLocator(x_.tolist(), y_.tolist(), S=1, curve="concave", direction="increasing")
                    f = plt.figure()
                    ax = f.add_subplot(1,1,1)
                    ax.plot(x[:-1], cdf/cdf[-1], 'o-')
                    ax.plot([round(kneedle.knee, 3)]*2, [0, 1.1], 'k--')
                    plt.xlabel('number of patients with mutated gene')
                    plt.ylim(0, 1)
                    ax.plot(round(kneedle.knee, 3), round(kneedle.knee_y, 3), 'ro')
                    plt.savefig(os.path.join('Results_' + str(seed), 'outlier_removing_' + '_'.join(analysis)))
                    plt.close()
                    inds = numb < kneedle.knee
                    outlier_threshold = kneedle.knee/features_train.shape[0]
                    to_rem = []
                    for col_ in features_train.columns:
                        if features_train[col_].sum()/features_train.shape[0] > outlier_threshold:
                            to_rem.append(col_)
                    print('Number of excluded genes (overmutated): {}'.format(len(to_rem)))
                    features_train = features_train.drop(to_rem, axis=1)
                    features_test = features_test.drop(to_rem, axis=1)
                    print('Number of genes: {} {}'.format(features_train.shape[1], features_test.shape[1]))
                # remove genes with frequency lower than 5%
                if ('gc' in analysis[0]): 
                    to_del_0 = []
                    for gene in features_train.columns:
                        if features_train[gene].sum()/features_train.shape[0] < 0.05:
                            to_del_0.append(gene)
                    print('Number of excluded genes (gc too common): {}'.format(len(to_del_0)))
                    features_train = features_train.drop(to_del_0, axis=1)
                    features_test = features_test.drop(to_del_0, axis=1)
                    print('Number of genes: {} {}'.format(features_train.shape[1], features_test.shape[1]))
                # remove genes on chrX
                if 'X' not in analysis: 
                    to_excl = []            
                    for col_df in features_train.columns: 
                        if col_df.split('_')[0] in self.genes_x.tolist():
                            to_excl.append(col_df)
                    to_excl = list(set(to_excl))
                    print('Number of excluded genes (chr X): {}'.format(len(to_excl)))
                    features_train = features_train.drop(to_excl, axis=1)
                    features_test = features_test.drop(to_excl, axis=1)
                    print('Number of genes: {} {}'.format(features_train.shape[1], features_test.shape[1]))
                else: # keep only genes on chrX
                    to_excl_non_x = []            
                    for col_df in features_train.columns: 
                        if col_df.split('_')[0] not in self.genes_x.tolist():
                            to_excl_non_x.append(col_df)
                    to_excl_non_x = list(set(to_excl_non_x))
                    print('Number of excluded genes (chr X): {}'.format(len(to_excl_non_x)))
                    features_train = features_train.drop(to_excl_non_x, axis=1)
                    features_test = features_test.drop(to_excl_non_x, axis=1)
                    print('Number of genes: {} {}'.format(features_train.shape[1], features_test.shape[1]))
                # remove non-informative genes
                to_excl_gene = []
                for jj in features_train.columns:
                    if (1 == features_train[jj]).all() or (0 == features_train[jj]).all():
                        to_excl_gene.append(jj)
                print('Number of excluded genes (non informative): {}'.format(len(to_excl_gene)))
                features_train = features_train.drop(to_excl_gene, axis=1)
                features_test = features_test.drop(to_excl_gene, axis=1)
                X_train = features_train.values
                y_train = target_train.values.astype(int)
                X_test = features_test.values
                y_test = target_test.values.astype(int)
                print('Postprocessing training dataset dimension', X_train.shape,y_train.shape)
                print('Postprocessing testing dataset dimension', X_test.shape,y_test.shape)
                # grid search
                param = np.logspace(np.log10(1e-2), np.log10(1e1), 51).tolist()
                parameters = {'C':param}
                logreg = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', max_iter=500, class_weight='balanced')
                matrix_score = np.zeros((self.num_trials_grid, len(param)))
                matrix_score_std = np.zeros((self.num_trials_grid, len(param)))
                for i in range(self.num_trials_grid):
                    cv_i = StratifiedKFold(n_splits=self.fold, shuffle=True, random_state=i)
                    clf = GridSearchCV(logreg, parameters, cv=cv_i, scoring=self.grid_search_metric, n_jobs=-1)
                    clf.fit(X_train, y_train)
                    matrix_score[i, :] = clf.cv_results_['mean_test_score']
                    matrix_score_std[i, :] = clf.cv_results_['std_test_score']
                scores_lr_cv = np.mean(matrix_score, 0)
                scores_std_lr = np.mean(matrix_score_std, 0)
                highest_param_lr = np.max(scores_lr_cv) - scores_std_lr[np.argmax(scores_lr_cv)]/2./self.std_reduction
                for par, sco in zip(param, scores_lr_cv):
                    if sco >= highest_param_lr:
                        best_param_lr = par
                        break
                # Grid search plot
                f = plt.figure()
                ax = f.add_subplot(1,1,1)
                ax.errorbar(1./np.array(param), scores_lr_cv, scores_std_lr/2., marker='o', linestyle='-'); plt.xscale('log')
                ax.plot(1./best_param_lr, sco, marker='o', color='r', markersize=12)
                plt.ylabel('cross-validation score')
                plt.xlabel('LASSO parameter')
                plt.savefig(os.path.join('Results_' + str(seed), 'grid_search_' + '_'.join(analysis)))
                plt.close()
                # Model fitting
                logreg_model = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', max_iter=500,
                                                  C=best_param_lr, class_weight='balanced')
                # Feature Rank
                logreg_model.fit(X_train, y_train)
                print('Param log regression: ', best_param_lr)
                print('Score fit log regression (train): ', logreg_model.score(X_train, y_train))
                print('Score fit log regression (test): ', logreg_model.score(X_test, y_test))
                df_clean = pd.DataFrame(features_train.T.reset_index()['index']).rename(columns={'index':'gene'})
                df_res = pd.DataFrame(copy.deepcopy(df_clean['gene']))
                df_res['importance'] = logreg_model.coef_[0]
                df_res['abs_score'] = np.abs(logreg_model.coef_[0])
                df_plot = df_res.sort_values(['abs_score'], ascending=False)
                if np.sum(df_plot['abs_score']>0) > 0: 
                    f = plt.figure()
                    f.add_subplot(1,1,1)
                    ax = df_plot[df_plot['abs_score']>0].iloc[:75].plot.bar(x='gene', y='importance', rot=270)
                    plt.xticks(fontsize=5)
                    plt.tight_layout()
                    plt.savefig(os.path.join('Results_' + str(seed), 'feature_importance_' + '_'.join(analysis)))
                    plt.close()
                else:
                    print('No features selected')
                # ROC curve
                cv = StratifiedKFold(n_splits=self.fold, shuffle=True, random_state=0)
                classifier = logreg_model
                tprs = []
                aucs = []
                mean_fpr = np.linspace(0, 1, 100)
                fig, ax = plt.subplots()
                for i, (train, test) in enumerate(cv.split(X_train, y_train)):
                    classifier.fit(X_train[train], y_train[train])
                    viz = plot_roc_curve(classifier, X_train[test], y_train[test],
                                         name='ROC fold {}'.format(i),
                                         alpha=0.3, lw=1, ax=ax)
                    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    aucs.append(viz.roc_auc)
                ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                        label='Chance', alpha=.8)
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(aucs)
                ax.plot(mean_fpr, mean_tpr, color='b',
                        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                        lw=2, alpha=.8)
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                                label=r'$\pm$ 1 std. dev.')
                ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05]),
                ax.legend(loc="lower right", prop={'size': 7})
                plt.savefig('Results_' + str(seed) + '/roc_auc_' + '_'.join(analysis))
                plt.close()
                # save table of extracted genes
                df_final = df_plot[df_plot['abs_score']>0]
                df_final['gene'] = df_final['gene']+analysis[-1]
                df_final = df_final.set_index('gene')
                df_final.to_csv(os.path.join('Results_' + str(seed), 'genes_' + '_'.join(analysis[:-1]) + '.csv'))

class Ipgs(object):
    def __init__(self, booleans_folder):
        """
        mm: tuple
            weigths in the IPGS formula for just_medium variants
        booleans: pd.DataFrame
        ipgs.load()
            all the booleans
        """
        self.mm = [None, None]
        self.mr = [None, None]
        self.mu = [None, None]
        self.booleans = pd.DataFrame()
        suffixes = [('data_al1_ultrarare.csv','_al1_ultrarare'),
                ('data_al2_ultrarare.csv','_al2_ultrarare'),
                ('data_al1_just_rare.csv','_al1_just_rare'),
                ('data_al2_just_rare.csv','_al2_just_rare'),
                ('data_al1_just_medium.csv','_al1_just_medium'),
                ('data_al2_just_medium.csv','_al2_just_medium'),
                ('data_gc_unique_hetero.csv',''),
                ('data_gc_unique_homo.csv','_homo')]
        self.indexes = set()
        for suffix in suffixes:
            file_name = os.path.join(booleans_folder,suffix[0])
            print('Reading booleans from: {}'.format(file_name))
            df = pd.read_csv(file_name).set_index('Unnamed: 0') 
            df.index = df.index.str.replace('_hg38','')
            col_name = []
            for gene_ in df.columns:
                col_name.append(gene_ + suffix[1])
            df.columns = col_name
            self.indexes = self.indexes | set(df.index)
            self.booleans = pd.concat((self.booleans, df), axis = 1)
        print('Complete dataset shape: {}'.format(self.booleans.shape))
        print('Indexes: {}'.format(len(self.indexes)))
    def load_genes(self, sex, folder):
        #--- read data for sex
        df_gene_al1_ultrarare = pd.read_csv(os.path.join(folder, 'genes_al1_ultrarare_' + sex + '.csv'))
        df_gene_al2_ultrarare = pd.read_csv(os.path.join(folder, 'genes_al2_ultrarare_' + sex + '.csv'))
        df_gene_al1_just_rare = pd.read_csv(os.path.join(folder, 'genes_al1_just_rare_' + sex + '.csv'))
        df_gene_al2_just_rare = pd.read_csv(os.path.join(folder, 'genes_al2_just_rare_' + sex + '.csv'))
        df_gene_al1_just_medium = pd.read_csv(os.path.join(folder, 'genes_al1_just_medium_' + sex + '.csv'))
        df_gene_al2_just_medium = pd.read_csv(os.path.join(folder, 'genes_al2_just_medium_' + sex + '.csv'))
        df_gene_gc = pd.read_csv(os.path.join(folder, 'genes_gc_unique_hetero_' + sex + '.csv'))
        df_gene_gc_homo = pd.read_csv(os.path.join(folder, 'genes_gc_unique_homo_' + sex + '.csv'))
        #--- read data for the other sex
        other_sex = ['male', 'female']
        other_sex.remove(sex)
        df_gene_al1_ultrarare_other_sex = pd.read_csv(os.path.join(folder, 'genes_al1_ultrarare_' + other_sex[0] + '.csv'))
        df_gene_al2_ultrarare_other_sex = pd.read_csv(os.path.join(folder, 'genes_al2_ultrarare_' + other_sex[0] + '.csv'))
        df_gene_al1_just_rare_other_sex = pd.read_csv(os.path.join(folder, 'genes_al1_just_rare_' + other_sex[0] + '.csv'))
        df_gene_al2_just_rare_other_sex = pd.read_csv(os.path.join(folder, 'genes_al2_just_rare_' + other_sex[0] + '.csv'))
        df_gene_al1_just_medium_other_sex = pd.read_csv(os.path.join(folder, 'genes_al1_just_medium_' + other_sex[0] + '.csv'))
        df_gene_al2_just_medium_other_sex = pd.read_csv(os.path.join(folder, 'genes_al2_just_medium_' + other_sex[0] + '.csv'))
        df_gene_gc_other_sex = pd.read_csv(os.path.join(folder, 'genes_gc_unique_hetero_' + other_sex[0] + '.csv'))
        df_gene_gc_homo_other_sex = pd.read_csv(os.path.join(folder, 'genes_gc_unique_homo_' + other_sex[0] + '.csv'))
        #--- read data for both sex together
        df_gene_al1_ultrarare_mf = pd.read_csv(os.path.join(folder, 'genes_al1_ultrarare_male_female' + '.csv'))
        df_gene_al2_ultrarare_mf = pd.read_csv(os.path.join(folder, 'genes_al2_ultrarare_male_female' + '.csv'))
        df_gene_al1_just_rare_mf = pd.read_csv(os.path.join(folder, 'genes_al1_just_rare_male_female' + '.csv'))
        df_gene_al2_just_rare_mf = pd.read_csv(os.path.join(folder, 'genes_al2_just_rare_male_female' + '.csv'))
        df_gene_al1_just_medium_mf = pd.read_csv(os.path.join(folder, 'genes_al1_just_medium_male_female' + '.csv'))
        df_gene_al2_just_medium_mf = pd.read_csv(os.path.join(folder, 'genes_al2_just_medium_male_female' + '.csv'))
        df_gene_gc_mf = pd.read_csv(os.path.join(folder, 'genes_gc_unique_hetero_male_female' + '.csv'))
        df_gene_gc_homo_mf = pd.read_csv(os.path.join(folder, 'genes_gc_unique_homo_male_female'+ '.csv'))
        #---- from both sex keep only the ones that are not in each sex separately
        df_gene_al1_ultrarare_mf = df_gene_al1_ultrarare_mf[~df_gene_al1_ultrarare_mf['gene'].isin(df_gene_al1_ultrarare_other_sex['gene'].values.tolist() + df_gene_al1_ultrarare['gene'].values.tolist())]
        df_gene_al2_ultrarare_mf = df_gene_al2_ultrarare_mf[~df_gene_al2_ultrarare_mf['gene'].isin(df_gene_al2_ultrarare_other_sex['gene'].values.tolist() + df_gene_al2_ultrarare['gene'].values.tolist())]
        df_gene_al1_just_rare_mf = df_gene_al1_just_rare_mf[~df_gene_al1_just_rare_mf['gene'].isin(df_gene_al1_just_rare_other_sex['gene'].values.tolist() + df_gene_al1_just_rare['gene'].values.tolist())]
        df_gene_al2_just_rare_mf = df_gene_al2_just_rare_mf[~df_gene_al2_just_rare_mf['gene'].isin(df_gene_al2_just_rare_other_sex['gene'].values.tolist() + df_gene_al2_just_rare['gene'].values.tolist())]
        df_gene_al1_just_medium_mf = df_gene_al1_just_medium_mf[~df_gene_al1_just_medium_mf['gene'].isin(df_gene_al1_just_medium_other_sex['gene'].values.tolist() + df_gene_al1_just_medium['gene'].values.tolist())]
        df_gene_al2_just_medium_mf = df_gene_al2_just_medium_mf[~df_gene_al2_just_medium_mf['gene'].isin(df_gene_al2_just_medium_other_sex['gene'].values.tolist() + df_gene_al2_just_medium['gene'].values.tolist())]
        df_gene_gc_mf = df_gene_gc_mf[~df_gene_gc_mf['gene'].isin(df_gene_gc_other_sex['gene'].values.tolist() + df_gene_gc['gene'].values.tolist())]
        df_gene_gc_homo_mf = df_gene_gc_homo_mf[~df_gene_gc_homo_mf['gene'].isin(df_gene_gc_homo_other_sex['gene'].values.tolist() + df_gene_gc_homo['gene'].values.tolist())]
        #--- now merge everything
        df_gene = pd.concat((df_gene_al1_ultrarare, df_gene_al2_ultrarare,
                             df_gene_al1_just_rare, df_gene_al2_just_rare,
                             df_gene_al1_just_medium, df_gene_al2_just_medium,
                             df_gene_gc, df_gene_gc_homo,
                             df_gene_al1_ultrarare_mf, df_gene_al2_ultrarare_mf,
                             df_gene_al1_just_rare_mf, df_gene_al2_just_rare_mf,
                             df_gene_al1_just_medium_mf, df_gene_al2_just_medium_mf,
                             df_gene_gc_mf, df_gene_gc_homo_mf,
                             ), axis = 0)
        #--- add stuf f in X chr --> no sex merging
        df_gene_al1_x_ultrarare = pd.read_csv(os.path.join(folder, 'genes_al1_ultrarare_X_' + sex + '.csv'))
        df_gene_al1_x_just_rare = pd.read_csv(os.path.join(folder, 'genes_al1_just_rare_X_' + sex + '.csv'))
        df_gene_al1_x_just_medium = pd.read_csv(os.path.join(folder, 'genes_al1_just_medium_X_' + sex + '.csv'))
        df_gene_gc_x = pd.read_csv(os.path.join(folder, 'genes_gc_unique_hetero_X_' + sex + '.csv'))
        else:
            df_gene_gc_x = pd.read_csv(os.path.join(folder, 'genes_GC_X_' + sex + '.csv'))
        df_gene = pd.concat((df_gene, df_gene_al1_x_ultrarare, df_gene_al1_x_just_rare, df_gene_al1_x_just_medium, df_gene_gc_x), axis = 0)
        #--- add columns to track where features are coming from
        df_gene['from_this_sex'] = 0
        df_gene['from_other_sex'] = 0
        df_gene.loc[df_gene['gene'].isin(df_gene_al1_ultrarare['gene']), 'from_this_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_al2_ultrarare['gene']), 'from_this_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_al1_just_rare['gene']), 'from_this_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_al2_just_rare['gene']), 'from_this_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_al1_just_medium['gene']), 'from_this_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_al2_just_medium['gene']), 'from_this_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_gc['gene']), 'from_this_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_gc_homo['gene']), 'from_this_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_al1_x_ultrarare['gene']), 'from_this_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_al1_x_just_rare['gene']), 'from_this_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_al1_x_just_medium['gene']), 'from_this_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_gc_x['gene']), 'from_this_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_al1_ultrarare_mf['gene']), 'from_other_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_al2_ultrarare_mf['gene']), 'from_other_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_al1_just_rare_mf['gene']), 'from_other_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_al2_just_rare_mf['gene']), 'from_other_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_al1_just_medium_mf['gene']), 'from_other_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_al2_just_medium_mf['gene']), 'from_other_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_gc_mf['gene']), 'from_other_sex'] = 1
        df_gene.loc[df_gene['gene'].isin(df_gene_gc_homo_mf['gene']), 'from_other_sex'] = 1
        return df_gene
    def filter(self, pheno):
        indexes = set(pheno.index) & self.indexes
        indexes_out = set(pheno.index) - self.indexes
        with open('missing_samples_in_booleans.txt','wt') as fout:
            for index in indexes_out:
                fout.write('{}\n'.format(index))
        return pheno.loc[indexes]
    def predict(self, pheno, min_boot_reps = 1, file_pathway = None):
        if file_pathway is not None:
            with open(file_pathway,'rt') as fin:
                l = fin.readline()
                genes_pathway = []
                for l in fin.readlines():
                    gene = l.split(',')[0].strip()
                    if gene:
                        genes_pathway.append(gene)
        output = pd.DataFrame(index = pheno.index)
        for sex_code, sex in enumerate(['male', 'female']):    
            samples = pheno[pheno['gender'] == sex_code].index
            if len(samples) == 0:
                continue
            # load file of discovered features 
            df_gene = pd.read_csv(sex + '_overall_genes.csv', usecols = ['gene', 'mean', 'min', 'max', 'count'])
            df_gene = df_gene[df_gene['count']>min_boot_reps]
            df_gene = df_gene[np.sign(df_gene['min'])==np.sign(df_gene['max'])]
            df_gene['importance'] = df_gene['mean']
            df_gene[~df_gene['gene'].isin(self.booleans.columns.tolist())].to_csv('not_found_features_' + sex + '.csv')
            df_gene = df_gene[df_gene['gene'].isin(self.booleans.columns.tolist())]
            # identify protective/suscentibility features
            susc_ultrarare = [kk for kk in df_gene[df_gene['importance']>=0]['gene'].tolist() if ('ultrarare' in kk)]
            susc_just_rare = [kk for kk in df_gene[df_gene['importance']>=0]['gene'].tolist() if ('just_rare' in kk)]
            susc_just_medium = [kk for kk in df_gene[df_gene['importance']>=0]['gene'].tolist() if ('just_medium' in kk)]
            susc_common = [kk for kk in df_gene[df_gene['importance']>=0]['gene'].tolist() if ('rare' not in kk and 'medium' not in kk)]
            prot_ultrarare = [kk for kk in df_gene[df_gene['importance']<0]['gene'].tolist() if ('ultrarare' in kk)]
            prot_just_rare = [kk for kk in df_gene[df_gene['importance']<0]['gene'].tolist() if ('just_rare' in kk)]
            prot_just_medium = [kk for kk in df_gene[df_gene['importance']<0]['gene'].tolist() if ('just_medium' in kk)]
            prot_common = [kk for kk in df_gene[df_gene['importance']<0]['gene'].tolist() if ('rare' not in kk and 'medium' not in kk)]
            if file_pathway is not None:
                susc_ultrarare = [kk for kk in susc_ultrarare if any([gene in kk for gene in genes_pathway])]
                susc_just_rare = [kk for kk in susc_just_rare if any([gene in kk for gene in genes_pathway])]
                susc_just_medium = [kk for kk in susc_just_medium if any([gene in kk for gene in genes_pathway])]
                susc_common = [kk for kk in susc_common if any([gene in kk for gene in genes_pathway])]
                prot_ultrarare = [kk for kk in prot_ultrarare if any([gene in kk for gene in genes_pathway])]
                prot_just_rare = [kk for kk in prot_just_rare if any([gene in kk for gene in genes_pathway])]
                prot_just_medium = [kk for kk in prot_just_medium if any([gene in kk for gene in genes_pathway])]
                prot_common = [kk for kk in prot_common if any([gene in kk for gene in genes_pathway])]
            print('Number of selected features:')
            print('\t{} {} {}'.format('ultrarare',len(susc_ultrarare),len(prot_ultrarare)))
            print('\t{} {} {}'.format('rare',len(susc_just_rare),len(prot_just_rare)))
            print('\t{} {} {}'.format('medium',len(susc_just_medium),len(prot_just_medium)))
            print('\t{} {} {}'.format('common',len(susc_common),len(prot_common)))
            print('Total: {}'.format(len(susc_ultrarare)+len(prot_ultrarare)+len(susc_just_rare)+len(prot_just_rare)+len(susc_just_medium)+len(prot_just_medium)+len(susc_common+prot_common)))
            # compute the 4 components of the IPGS
            df_gender = self.booleans.loc[samples]
            output.loc[samples, 'common'] = df_gender[susc_common].sum(1) - df_gender[prot_common].sum(1)
            output.loc[samples, 'ultra_rare'] = df_gender[susc_ultrarare].sum(1) - df_gender[prot_ultrarare].sum(1)
            output.loc[samples, 'just_rare'] = df_gender[susc_just_rare].sum(1) - df_gender[prot_just_rare].sum(1)
            output.loc[samples, 'just_medium'] = df_gender[susc_just_medium].sum(1) - df_gender[prot_just_medium].sum(1)
            output.loc[samples, 'common_severe'] = df_gender[susc_common].sum(1)
            output.loc[samples, 'ultra_rare_severe'] = df_gender[susc_ultrarare].sum(1)
            output.loc[samples, 'just_rare_severe'] = df_gender[susc_just_rare].sum(1)
            output.loc[samples, 'just_medium_severe'] = df_gender[susc_just_medium].sum(1)
            output.loc[samples, 'common_protective'] = df_gender[prot_common].sum(1)
            output.loc[samples, 'ultra_rare_protective'] = df_gender[prot_ultrarare].sum(1)
            output.loc[samples, 'just_rare_protective'] = df_gender[prot_just_rare].sum(1)
            output.loc[samples, 'just_medium_protective'] = df_gender[prot_just_medium].sum(1)

            # compute IPGS score for all the patients
            output.loc[samples,'ipgs'] = output.loc[samples,'common']+self.mm[sex_code]*output.loc[samples,'just_medium']+self.mr[sex_code]*output.loc[samples,'just_rare']+self.mu[sex_code]*output.loc[samples,'ultra_rare']

            # output features
            output_bools = pd.DataFrame()

            output_bools = pd.concat((output_bools, pheno.loc[samples,['gender',]]), axis = 1)
            output_bools = pd.concat((output_bools, output.loc[samples,'ipgs']), axis = 1)

            output_bools = pd.concat((output_bools, df_gender[susc_common]), axis = 1)
            output_bools.rename(columns = {col:col+'_severe_common' for col in susc_common}, inplace = True)
            output_bools = pd.concat((output_bools, df_gender[prot_common]), axis = 1)
            output_bools.rename(columns = {col:col+'_mild_common' for col in prot_common}, inplace = True)

            output_bools = pd.concat((output_bools, df_gender[susc_just_medium]), axis = 1)
            output_bools.rename(columns = {col:col+'_severe_just_medium' for col in susc_just_medium}, inplace = True)
            output_bools = pd.concat((output_bools, df_gender[prot_just_medium]), axis = 1)
            output_bools.rename(columns = {col:col+'_mild_just_medium' for col in prot_just_medium}, inplace = True)

            output_bools = pd.concat((output_bools, df_gender[susc_just_rare]), axis = 1)
            output_bools.rename(columns = {col:col+'_severe_just_rare' for col in susc_just_rare}, inplace = True)
            output_bools = pd.concat((output_bools, df_gender[prot_just_rare]), axis = 1)
            output_bools.rename(columns = {col:col+'_mild_just_rare' for col in prot_just_rare}, inplace = True)

            output_bools = pd.concat((output_bools, df_gender[susc_ultrarare]), axis = 1)
            output_bools.rename(columns = {col:col+'_severe_ultrarare' for col in susc_ultrarare}, inplace = True)
            output_bools = pd.concat((output_bools, df_gender[prot_ultrarare]), axis = 1)
            output_bools.rename(columns = {col:col+'_mild_ultrarare' for col in prot_ultrarare}, inplace = True)


            df_gene.set_index('gene', inplace = True)
            df_beta = df_gene.loc[susc_common+prot_common+susc_just_medium+prot_just_medium+susc_just_rare+prot_just_rare+susc_ultrarare+prot_ultrarare,['importance','min','max']].transpose()
            df_beta.rename(columns = {col:col+'_severe_common' for col in susc_common}, inplace = True)
            df_beta.rename(columns = {col:col+'_mild_common' for col in prot_common}, inplace = True)
            df_beta.rename(columns = {col:col+'_severe_just_medium' for col in susc_just_medium}, inplace = True)
            df_beta.rename(columns = {col:col+'_mild_just_medium' for col in prot_just_medium}, inplace = True)
            df_beta.rename(columns = {col:col+'_severe_just_rare' for col in susc_just_rare}, inplace = True)
            df_beta.rename(columns = {col:col+'_mild_just_rare' for col in prot_just_rare}, inplace = True)
            df_beta.rename(columns = {col:col+'_severe_ultrarare' for col in susc_ultrarare}, inplace = True)
            df_beta.rename(columns = {col:col+'_mild_ultrarare' for col in prot_ultrarare}, inplace = True)
            cols = output_bools.columns
            output_bools = pd.concat((df_beta, output_bools), axis = 0)
            output_bools[cols].to_csv('booleans_{}.csv'.format(sex), index = True)
            output_bools[cols].transpose().to_csv('booleansT_{}.csv'.format(sex), index = True)
        return output
    def fit_features(self, results_folder_prefix, results_list):
        """
        Define which features are used for computing the IPGS
        """
        df_male = pd.DataFrame()
        df_female = pd.DataFrame()
        for num in results_list:
            print('Reading data from {}{}'.format(results_folder_prefix, num))
            df_i_m = self.load_genes('male', results_folder_prefix + str(num))
            df_male = pd.concat((df_male, df_i_m))
            df_i_f = self.load_genes('female', results_folder_prefix + str(num))
            df_female = pd.concat((df_female, df_i_f))   
        #df_male.groupby('gene').count()['importance'].sort_values(ascending=False)
        df_m = df_male.groupby('gene').agg({'importance':['mean', 'min', 'max', 'count']})['importance'].sort_values('count', ascending=False)
        df_m_same_sex = df_male.groupby('gene').agg({'from_this_sex':['mean','count']})['from_this_sex'].sort_values('count', ascending=False)
        df_m_other_sex = df_male.groupby('gene').agg({'from_other_sex':['mean','count']})['from_other_sex'].sort_values('count', ascending=False)
        df_m['from_this_sex'] = np.round(df_m_same_sex['count']*df_m_same_sex['mean']).astype(int)
        df_m['from_other_sex'] = np.round(df_m_other_sex['count']*df_m_other_sex['mean']).astype(int)
        df_m.to_csv('male_overall_genes.csv')
        df_f = df_female.groupby('gene').agg({'importance':['mean', 'min', 'max', 'count']})['importance'].sort_values('count', ascending=False)
        df_f_same_sex = df_female.groupby('gene').agg({'from_this_sex':['mean','count']})['from_this_sex'].sort_values('count', ascending=False)
        df_f_other_sex = df_female.groupby('gene').agg({'from_other_sex':['mean','count']})['from_other_sex'].sort_values('count', ascending=False)
        df_f['from_this_sex'] = np.round(df_f_same_sex['count']*df_f_same_sex['mean']).astype(int)
        df_f['from_other_sex'] = np.round(df_f_other_sex['count']*df_f_other_sex['mean']).astype(int)
        df_f.to_csv('female_overall_genes.csv')
    def fit_f_factors(self, pheno):
        from sklearn.metrics import silhouette_score
        for code, sex in enumerate(['male', 'female']):
            distances = []
            inp = []
            for mm in [1, 2, 3, 4]:
                for mr in [2, 3, 4, 5, 6, 8]:
                    for mu in [5, 7, 10, 20, 50, 100]:
                        if mu > mr > mm:
                            self.mm[code], self.mr[code], self.mu[code] = mm, mr, mu
                            inds = np.logical_and(pheno['gender'] == code, np.logical_not(pheno['adjusted_grading'].isnull()))
                            ipgs = self.predict(pheno[inds])
                            distance = silhouette_score(ipgs['ipgs'].values.reshape((-1,1)), pheno.loc[inds,'adjusted_grading'])
                            print('F-factors fitting {} {} {}: {}'.format(self.mm[code], self.mr[code], self.mu[code], distance))
                            distances.append(distance)
                            inp.append([mm, mr, mu])
            self.mm[code], self.mr[code], self.mu[code] = inp[np.argmax(distances)]
        pd.DataFrame({'F_medium':self.mm,'F_rare':self.mr,'F_ultrarare':self.mu}, index = ['male','female']).to_csv('factors.csv'.format(sex))
    def fit(self, pheno, results_folder_prefix, results_list):
        """
        Compute everything that is needed to calculate the IPGS:
            * select which features are used using the results of SelectFeatures (folders Results_*)
            * estimate the optimal value of F-factors
        """
        self.fit_features(results_folder_prefix, results_list)
        #self.fit_f_factors(pheno)
    def load(self):
        """
        Recover the resutls of a previous fitting from pickle files
        """
        factors = pd.read_csv('factors.csv',index_col=0)
        self.mm = [factors.loc['male','F_medium'], factors.loc['female','F_medium']]
        self.mr = [factors.loc['male','F_rare'], factors.loc['female','F_rare']]
        self.mu = [factors.loc['male','F_ultrarare'], factors.loc['female','F_ultrarare']]
    def __str__(self):
        output = ''
        output += 'Factors for IPGS\n'
        output += '\tmales: {} {} {}\n'.format(self.mm[0], self.mr[0], self.mu[0])
        output += '\tfemales: {} {} {}\n'.format(self.mm[1], self.mr[1], self.mu[1])
        return output[:-1]

class PMM(object):
    def __init__(self):
        self.bins = 100
    def normalize(self, features):
        if 'ipgs_scaled' not in features.columns:
            scaler = preprocessing.KBinsDiscretizer(n_bins=self.bins, encode='ordinal').fit(features[['ipgs']])
            features['ipgs_scaled'] = scaler.transform(features[['ipgs']])/(self.bins-1)
        if 'age_norm' in features.columns:
            features['age'] = features['age_norm']
        else:
            features['age'] = features['age'] / 99.0
        return features
    def fit(self, features):
        features = features.copy()
        model_1 = LogisticRegression(class_weight='balanced', penalty='none') 
        model_1.fit(features[['age','gender']], features['target'])
        model_2 = LogisticRegression(class_weight='balanced', penalty='none')
        model_2.fit(features[['ipgs_scaled']], features['target'])
        model_3 = LogisticRegression(class_weight='balanced', penalty='none')
        model_3.fit(features[['age','gender','ipgs_scaled']], features['target'])
        with open('models.pkl', 'wb') as fout:
            pickle.dump(model_1, fout)
            pickle.dump(model_2, fout)
            pickle.dump(model_3, fout)
    def predict(self, features):
        features = features.copy()
        with open('models.pkl', 'rb') as fin:
            model_1 = pickle.load(fin)
            model_2 = pickle.load(fin)
            model_3 = pickle.load(fin)
        features['pred_1'] = model_1.predict(features[['age','gender']])
        features['prob_1'] = model_1.predict_proba(features[['age','gender']])[:,1]
        features['pred_2'] = model_2.predict(features[['ipgs_scaled']])
        features['prob_2'] = model_2.predict_proba(features[['ipgs_scaled']])[:,1]
        features['pred_3'] = model_3.predict(features[['age','gender','ipgs_scaled']])
        features['prob_3'] = model_3.predict_proba(features[['age','gender','ipgs_scaled']])[:,1]
        return features[['pred_1','pred_2','pred_3','prob_1','prob_2','prob_3']]
    def show_score(self, features):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score
        n_classes = 2
        features = features.copy()
        results = self.predict(features)
        #--- Coefficients
        with open('models.pkl', 'rb') as fin:
            model_1 = pickle.load(fin)
            model_2 = pickle.load(fin)
            model_3 = pickle.load(fin)
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        ax.bar([0,1], model_1.coef_.flatten(),color='r',label='AGE+SEX')
        ax.bar([3,], model_2.coef_.flatten(),color='b',label='IPGS')
        ax.bar([5,6,7], model_3.coef_.flatten(),color='g',label='AGE+SEX+IPGS')
        plt.xticks([0,1,3,5,6,7],['age','sex','ipgs','age','sex','ipgs'])
        plt.grid()
        plt.legend()
        pdf.savefig()
        plt.close()
        #--- Confusion matrix plot
        f = plt.figure()
        cm = confusion_matrix(features['target'], results['pred_1'])
        cm_df = pd.DataFrame(cm, index=np.arange(n_classes), columns=[str(ll)+' pred' for ll in np.arange(n_classes)])
        sns.heatmap(cm_df, center=0, cmap=sns.diverging_palette(220, 15, as_cmap=True), annot=True, fmt='g')
        plt.xticks(rotation=0)
        plt.yticks(rotation=90, va='center')
        plt.title('age + gender')
        pdf.savefig()
        plt.close()
        f = plt.figure()
        cm = confusion_matrix(features['target'], results['pred_2'])
        cm_df = pd.DataFrame(cm, index=np.arange(n_classes), columns=[str(ll)+' pred' for ll in np.arange(n_classes)])
        sns.heatmap(cm_df, center=0, cmap=sns.diverging_palette(220, 15, as_cmap=True), annot=True, fmt='g')
        plt.xticks(rotation=0)
        plt.yticks(rotation=90, va='center')
        plt.title('ipgs')
        pdf.savefig()
        plt.close()
        f = plt.figure()
        cm = confusion_matrix(features['target'], results['pred_3'])
        cm_df = pd.DataFrame(cm, index=np.arange(n_classes), columns=[str(ll)+' pred' for ll in np.arange(n_classes)])
        sns.heatmap(cm_df, center=0, cmap=sns.diverging_palette(220, 15, as_cmap=True), annot=True, fmt='g')
        plt.xticks(rotation=0)
        plt.yticks(rotation=90, va='center')
        plt.title('age + gender + ipgs')
        pdf.savefig()
        plt.close()
        #--- Compare the results
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        plt.title('roc on test')
        viz = plot_roc_curve(model_1, features[['age','gender']], features['target'], name='age + gender', alpha=1, lw=2, ax=ax)
        viz = plot_roc_curve(model_2, features[['ipgs_scaled']], features['target'], name='ipgs', alpha=1, lw=2, ax=ax)
        viz = plot_roc_curve(model_3, features[['age','gender','ipgs_scaled']], features['target'], name='age + gender + ipgs', alpha=1, lw=2, ax=ax)
        df_r = pd.DataFrame()
        df_r.loc['Age + Sex', 'accuracy'] = accuracy_score(features['target'], results['pred_1'])
        df_r.loc['Age + Sex', 'precision'] = precision_score(features['target'], results['pred_1'])
        df_r.loc['Age + Sex', 'sensitivity'] = recall_score(features['target'], results['pred_1'])
        df_r.loc['Age + Sex', 'specificity'] = 2*balanced_accuracy_score(features['target'], results['pred_1']) - recall_score(features['target'], results['pred_1'])
        df_r.loc['Age + Sex + IPGS', 'accuracy'] = accuracy_score(features['target'], results['pred_3'])
        df_r.loc['Age + Sex + IPGS', 'precision'] = precision_score(features['target'], results['pred_3'])
        df_r.loc['Age + Sex + IPGS', 'sensitivity'] = recall_score(features['target'], results['pred_3'])
        df_r.loc['Age + Sex + IPGS', 'specificity'] = 2*balanced_accuracy_score(features['target'], results['pred_3']) - recall_score(features['target'], results['pred_3'])
        df_r.loc['IPGS', 'accuracy'] = accuracy_score(features['target'], results['pred_2'])
        df_r.loc['IPGS', 'precision'] = precision_score(features['target'], results['pred_2'])
        df_r.loc['IPGS', 'sensitivity'] = recall_score(features['target'], results['pred_2'])
        df_r.loc['IPGS', 'specificity'] = 2*balanced_accuracy_score(features['target'], results['pred_2']) - recall_score(features['target'], results['pred_2'])
        ax = ((df_r.round(2)*100).astype(int).T).plot(kind='bar', width=0.7, color=['darkgray', 'darkorange', 'darkgreen'],
                      linewidth=2, edgecolor='black', clip_on=False, legend=False, rot=0)
        for p in ax.patches:
            ax.annotate(str(np.round(p.get_height(), 2)), (p.get_x()+0.05, p.get_height()+2))
        plt.legend(loc=3)
        ax.tick_params(right=False,top= False, left=True, bottom=False)
        #plt.ylim(0, 90)
        plt.ylabel('Percentage (%)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        pdf.savefig()
        plt.close()

def plot_distribution_ipgs(features, title = None):
    from scipy import stats
    f = plt.figure()
    ax = f.add_subplot(3,1,1)
    rvs1 = features[features['target']==0]['ipgs']
    rvs2 = features[features['target']==1]['ipgs']
    features[features['target']==0]['ipgs'].plot(kind='kde', label='non-severe')
    features[features['target']==1]['ipgs'].plot(kind='kde', label='severe')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.title('males + females p: {0:4.3f}'.format(stats.ttest_ind(rvs1,rvs2)[1]))
    ax = f.add_subplot(3,1,2)
    rvs1 = features[np.logical_and(features['target']==0,features['gender']==0)]['ipgs']
    rvs2 = features[np.logical_and(features['target']==1,features['gender']==0)]['ipgs']
    features[np.logical_and(features['target']==0,features['gender']==0)]['ipgs'].plot(kind='kde', label='non-severe')
    features[np.logical_and(features['target']==1,features['gender']==0)]['ipgs'].plot(kind='kde', label='severe')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.title('males p: {0:4.3f}'.format(stats.ttest_ind(rvs1,rvs2)[1]))
    ax = f.add_subplot(3,1,3)
    rvs1 = features[np.logical_and(features['target']==0,features['gender']==1)]['ipgs']
    rvs2 = features[np.logical_and(features['target']==1,features['gender']==1)]['ipgs']
    features[np.logical_and(features['target']==0,features['gender']==1)]['ipgs'].plot(kind='kde', label='non-severe')
    features[np.logical_and(features['target']==1,features['gender']==1)]['ipgs'].plot(kind='kde', label='severe')
    frame1 = plt.gca()
    #frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.title('females p: {0:4.3f}'.format(stats.ttest_ind(rvs1,rvs2)[1]))
    if title is not None:
        plt.xlabel(title)
    pdf.savefig()
    plt.close()

pdf = PdfPages('figures.pdf')


#--- Estimate the OLR model
pheno_train4ipgs = pd.read_csv('pheno_train4ipgs.csv', usecols = ['sample','age','gender','grading'], index_col = 'sample')
pheno_train4ipgs.dropna(inplace = True)
olr = AdjustGrading()
olr.fit(pheno_train4ipgs, grading_col = 'grading')
pheno_train4ipgs = pd.concat((pheno_train4ipgs, olr.predict(pheno_train4ipgs)), axis = 1)
pheno_train4ipgs.to_csv('pheno_train4ipgs_olr.csv')
olr.dump('olr_males.pk', 'olr_females.pk')
olr.plot(pheno_train4ipgs)

#--- Definition of the features
sel = SelectFeatures('./booleans') # folder including the boolean features
sel.bootstrap(pheno_train4ipgs, 0, 100)
 
#--- Definition of the IPGS score
ipgs = Ipgs('./booleans') # folder including the boolean features
pmm = PMM()
pheno = pd.read_csv('pheno_train4ipgs_olr.csv', usecols = ['sample','age','gender','grading','adjusted_grading'], index_col = 'sample')
pheno = ipgs.filter(pheno)
ipgs.fit_features('./Results_',range(0,100)) # set of directories with the results of logistic models
ipgs.load()
pheno = pd.concat((pheno, ipgs.predict(pheno)), axis = 1)
pheno['target'] = (pheno['grading'] >= 2).astype(int)
pheno = pmm.normalize(pheno)
pheno.to_csv('pheno_train4ipgs_ipgs.csv')
 
##--- Calculation of the IPGS score on PMM training data
pheno = pd.read_csv('./pheno_train4pmm.csv',index_col = 'sample')
pheno = ipgs.filter(pheno)
pheno = pd.concat((pheno, ipgs.predict(pheno)), axis = 1)
pheno = pmm.normalize(pheno)
pheno.to_csv('pheno_train4pmm_ipgs.csv')

#--- Training of the predictive model
pmm = PMM()
pheno = pd.read_csv('pheno_train4pmm_ipgs.csv',index_col = 'sample')
pmm.fit(pheno)
plot_distribution_ipgs(pheno)
pmm.show_score(pheno)

#--- Scale datasets with known IPGS
for dataset in ['uk', 'germany', 'canada']:
    pmm = PMM()
    pheno = pd.read_csv('{}.csv'.format(dataset),index_col = 'sample')
    pheno = pmm.normalize(pheno)
    pheno.to_csv('{}_ipgs.csv'.format(dataset))
  
#--- Testing
for dataset in ['uk_ipgs', 'germany_ipgs', 'canada_ipgs']: # testing cohorts
    pheno = pd.read_csv('{}.csv'.format(dataset),index_col = 'sample')
    plot_distribution_ipgs(pheno)
    pmm.show_score(pheno)

pdf.close()
