import numpy as np
import sys
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from time import time
from multiprocessing import get_context
import tqdm

import numpy as np

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import svm

import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import sys
from time import time
from copy import deepcopy

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from time import time


sys.path.append('./external/gnomix/')

from src.Smooth.smooth import Smoother

from src.Smooth.Calibration import Calibrator

from src.Smooth.utils import mode_filter

from src.Smooth.utils import slide_window


class Base():

    def __init__(self, chm_len, window_size, num_ancestry, missing_encoding=2,
                    context=0.5, train_admix=True, n_jobs=None, seed=94305, verbose=False):

        self.C = chm_len
        self.M = window_size
        self.W = self.C//self.M # Number of windows
        self.A = num_ancestry
        self.missing_encoding=missing_encoding
        self.context = context
        self.train_admix = train_admix
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose
        self.base_multithread = False
        self.log_inference = False
        self.vectorize = True

        self.time = {}

    def init_base_models(self, model_factory):
        """
        inputs:
            - model_factory: function that returns a model object that has the functions
                - fit
                - predict
                - predict_proba
            - and the attributes
                - classes_
        """
        self.models = [model_factory() for _ in range(self.W)]

    def pad(self,X):
        pad_left = np.flip(X[:,0:self.context],axis=1)
        pad_right = np.flip(X[:,-self.context:],axis=1)
        return np.concatenate([pad_left,X,pad_right],axis=1)
        
    def train(self, X, y, verbose=True):
        """
        inputs:
            - X: np.array of shape (N, C) where N is sample size and C chm length
            - y: np.array of shape (N, C) where N is sample size and C chm length
        """
        if self.vectorize:
            try:
                np.lib.stride_tricks.sliding_window_view
                return self.train_vectorized(X, y)
            except AttributeError:
                print("Vectorized implementation requires numpy versions 1.20+.. Using loopy version..")
                self.vectorize = False
        if not self.vectorize:
            return self.train_loopy(X, y, verbose=verbose)


    def train_base_model(self, b, X, y):
        return b.fit(X, y)

    def predict_proba_base_model(self, b, X):
        return b.predict_proba(X)

    def train_vectorized(self, X, y):

        slide_window = np.lib.stride_tricks.sliding_window_view

        t = time()

        # pad
        if self.context != 0.0:
            X = self.pad(X)
            
        # convolve
        M_ = self.M + 2*self.context        
        idx = np.arange(0,self.C,self.M)[:-2]
        X_b = slide_window(X, M_, axis=1)[:,idx,:]

        # stack
        train_args = tuple(zip( self.models[:-1], np.swapaxes(X_b,0,1), np.swapaxes(y,0,1)[:-1] ))
        rem = self.C - self.M*self.W
        train_args += ((self.models[-1], X[:,X.shape[1]-(M_+rem):], y[:,-1]),)

        # train
        log_iter = tqdm.tqdm(train_args, total=self.W, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}', position=0, leave=True)
        if self.base_multithread:
            with get_context("spawn").Pool(self.n_jobs) as pool:
                self.models = pool.starmap(self.train_base_model, log_iter) 
        else:
            self.models = [self.train_base_model(*b) for b in log_iter]

        self.time["train"] = time() - t

    def predict_proba(self, X):
        """
        inputs:
            - X: np.array of shape (N, C) where N is sample size and C chm length
        returns 
            - B: base probabilities of shape (N,W,A)
        """
        if self.vectorize:
            try:
                np.lib.stride_tricks.sliding_window_view
                return self.predict_proba_vectorized(X)
            except AttributeError:
                print("Vectorized implementation requires numpy versions 1.20+.. Using loopy version..")
                self.vectorize = False
        if not self.vectorize:
            return self.predict_proba_loopy(X)

    def predict_proba_vectorized(self, X):

        slide_window = np.lib.stride_tricks.sliding_window_view

        t = time()

        # pad
        if self.context != 0.0:
            X = self.pad(X)
            
        # convolve
        M_ = self.M + 2*self.context        
        idx = np.arange(0,self.C,self.M)[:-2]
        X_b = slide_window(X, M_, axis=1)[:,idx,:]

        # stack
        base_args = tuple(zip( self.models[:-1], np.swapaxes(X_b,0,1) ))
        rem = self.C - self.M*self.W
        base_args += ((self.models[-1], X[:,X.shape[1]-(M_+rem):]), )

        if self.log_inference:
            base_args = tqdm.tqdm(base_args, total=self.W, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}', position=0, leave=True)

        # predict proba
        if self.base_multithread:
            with get_context("spawn").Pool(self.n_jobs) as pool:
                B = np.array(pool.starmap(self.predict_proba_base_model, base_args))
        else:
            B = np.array([self.predict_proba_base_model(*b) for b in base_args])

        B = np.swapaxes(B, 0, 1)

        self.time["inference"] = time() - t

        return B
    
    def predict(self, X):
        B = self.predict_proba(X)
        return np.argmax(B, axis=-1)
        
    def evaluate(self,X=None,y=None,B=None):

        round_accr = lambda accr : round(np.mean(accr)*100,2)

        if X is not None:
            y_pred = self.predict(X)
        elif B is not None:
            y_pred = np.argmax(B, axis=-1)
        else:
            print("Error: Need either SNP input or estimated probabilities to evaluate.")

        accr = round_accr( accuracy_score(y.reshape(-1), y_pred.reshape(-1)) )
        accr_bal = round_accr( balanced_accuracy_score(y.reshape(-1), y_pred.reshape(-1)) )

        return accr, accr_bal

class LogisticRegressionBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_multithread = True

        self.init_base_models(
            lambda : LogisticRegression(penalty="l2", C = 3., solver="liblinear", max_iter=1000)
        )


class XGBBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_multithread = True
        n_jobs = self.n_jobs if not self.base_multithread else 1

        self.init_base_models(
            lambda : XGBClassifier(
                n_estimators=20, max_depth=4, learning_rate=0.1, reg_lambda=1, reg_alpha=0,
                n_jobs=n_jobs, missing=self.missing_encoding, random_state=self.seed)
        )




class XGB_Smoother(Smoother):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gnofix = True
        assert self.W >= 2*self.S, "Smoother size to large for given window size. "
        self.model = XGBClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, reg_lambda=1, reg_alpha=0,
            nthread=self.n_jobs, random_state=self.seed,
            num_class=self.A, 
            use_label_encoder=False, objective='multi:softprob', eval_metric="mlogloss", verbosity = 2
        )

    def process_base_proba(self,B,y=None):
        B_slide, y_slide = slide_window(B, self.S, y)
        return B_slide, y_slide


class Gnomix():

    def __init__(self, C, M, A, S,
                base=None, smooth=None, mode="default", # base and smooth models
                snp_pos=None, snp_ref=None, snp_alt=None, population_order=None, missing_encoding=2, # dataset specific, TODO: store in one object
                n_jobs=None, path=None, # configs
                calibrate=False, context_ratio=0.5, mode_filter=False, # hyperparams
                seed=94305, verbose=False
    ):
        """
        Inputs
           C: chromosome length in SNPs
           M: number of windows for chromosome segmentation
           A: number of ancestry considered
        """

        self.C = C
        self.M = M
        self.A = A
        self.S = S
        self.W = self.C//self.M # number of windows

        # configs
        self.path = path
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        # data
        self.snp_pos = snp_pos
        self.snp_ref = snp_ref
        self.snp_alt = snp_alt
        self.population_order = population_order

        # gnomix hyperparams
        self.context = int(self.M*context_ratio)
        self.calibrate = calibrate

        # base = LogisticRegressionBase
        base = XGBBase
        if verbose:
            print("Base models:", base)
        smooth = XGB_Smoother
        if verbose:
            print("Smoother:", smooth)

        self.base = base(chm_len=self.C, window_size=self.M, num_ancestry=self.A,
                            missing_encoding=missing_encoding, context=self.context,
                            n_jobs=self.n_jobs, seed=self.seed, verbose=self.verbose)

        self.smooth = smooth(n_windows=self.W, num_ancestry=self.A, smooth_window_size=self.S,
                            n_jobs=self.n_jobs, calibrate=self.calibrate, mode_filter=mode_filter, 
                            seed=self.seed, verbose=self.verbose)
        
        # model stats
        self.time = {}
        self.accuracies = {}

        # gen map df
        self.gen_map_df = {}

    def write_gen_map_df(self,gen_map_df):
        self.gen_map_df = deepcopy(gen_map_df)

    def conf_matrix(self, y, y_pred):

        cm = confusion_matrix(y.reshape(-1), y_pred.reshape(-1))
        indices = sorted(np.unique( np.concatenate((y.reshape(-1),y_pred.reshape(-1))) ))

        return cm, indices

    def save(self):
        if self.path is not None:
            pickle.dump(self, open(self.path+"model.pkl", "wb"))

    def train(self,data,retrain_base=True,evaluate=True,verbose=True):

        train_time_begin = time()
        old_time = train_time_begin

        (X_t1,y_t1), (X_t2,y_t2), (X_v,y_v) = data
        
        if verbose:
            print("Training base models...")
        self.base.train(X_t1, y_t1)

        print(f"time elapsed for training base model: {round(time() - train_time_begin,2)}")
        train_time_begin = time()

        if verbose:
            print("Training smoother...")
        B_t2 = self.base.predict_proba(X_t2)
        self.smooth.train(B_t2,y_t2)

        if self.calibrate:
            # calibrates the predictions to be balanced w.r.t. the train1 class distribution
            if verbose:
                print("Fitting calibrator...")
            B_t1 = self.base.predict_proba(X_t1)
            self.smooth.train_calibrator(B_t1, y_t1)

        print(f"time elapsed for training smoother model: {round(time() - train_time_begin,2)}")
        train_time_begin = time()
        # Evaluate model
        if evaluate:
            if verbose:
                print("Evaluating model...")

            Acc = {}
            CM  = {}

            # training accuracy
            B_t1 = self.base.predict_proba(X_t1)
            y_t1_pred = self.smooth.predict(B_t1)
            y_t2_pred = self.smooth.predict(B_t2)
            Acc["base_train_acc"],   Acc["base_train_acc_bal"]   = self.base.evaluate(X=None,   y=y_t1, B=B_t1)
            Acc["smooth_train_acc"], Acc["smooth_train_acc_bal"] = self.smooth.evaluate(B=None, y=y_t2, y_pred=y_t2_pred)
            CM["train"] = self.conf_matrix(y=y_t1, y_pred=y_t1_pred)
            
            # val accuracy
            if X_v is not None:
                B_v = self.base.predict_proba(X_v)
                y_v_pred  = self.smooth.predict(B_v)
                Acc["base_val_acc"],     Acc["base_val_acc_bal"]     = self.base.evaluate(X=None,   y=y_v,  B=B_v )
                Acc["smooth_val_acc"],   Acc["smooth_val_acc_bal"]   = self.smooth.evaluate(B=None, y=y_v,  y_pred=y_v_pred )
                CM["val"] = self.conf_matrix(y=y_v, y_pred=y_v_pred)

            self.accuracies = Acc
            self.Confusion_Matrices = CM
        
        print(f"time elapsed for evaluating model: {round(time() - train_time_begin,2)}")
        train_time_begin = time()
        if retrain_base:
            # Store both training data in one np.array for memory efficency
            if X_v is not None:
                X_t, y_t = np.concatenate([X_t1, X_t2, X_v]), np.concatenate([y_t1, y_t2, y_v])
            else:
                X_t, y_t = np.concatenate([X_t1, X_t2]), np.concatenate([y_t1, y_t2])

            # Re-using all the data to re-train the base models
            if verbose:
                print("Re-training base models...")
            self.base.train(X_t, y_t)

        self.save()
        print(f"time elapsed for retraining base model: {round(time() - train_time_begin,2)}")
        self.time["training"] = round(time() - old_time,2)

    def predict(self,X):

        B = self.base.predict_proba(X)
        y_pred = self.smooth.predict(B)
        return y_pred

    def predict_proba(self,X):

        B = self.base.predict_proba(X)
        y_pred = self.smooth.predict_proba(B)
        return y_pred

    def write_config(self,fname):
        with open(fname,"w") as f:
            for attr in dir(self):
                val = getattr(self,attr)
                if type(val) in [int,float,str,bool,np.float64,np.float32]:
                    f.write("{}\t{}\n".format(attr,val))

    def phase(self,X,B=None,verbose=False):
        """
        Wrapper for XGFix
        """

        assert self.smooth is not None, "Smoother is not trained, returning original haplotypes"
        assert self.smooth.gnofix, "Type of Smoother ({}) does not currently support re-phasing".format(self.smooth)

        N, C = X.shape
        n = N//2
        X_phased = np.zeros((n,2,C), dtype=int)
        Y_phased = np.zeros((n,2,self.W), dtype=int)

        if B is None:
            B = self.base.predict_proba(X)
        B = B.reshape(n, 2, self.W, self.A)

        for i, X_i in enumerate(X.reshape(n,2,C)):
            sys.stdout.write("\rPhasing individual %i/%i" % (i+1, n))
            X_m, X_p = X_i
            X_m_, X_p_, Y_m_, Y_p_, history, tracker = gnofix(X_m, X_p, B=B[i], smoother=self.smooth, verbose=verbose)
            X_phased[i] = np.copy(np.array((X_m_,X_p_)))
            Y_phased[i] = np.copy(np.array((Y_m_,Y_p_)))

        print()
        
        return X_phased.reshape(N, C), Y_phased.reshape(N, self.W)
