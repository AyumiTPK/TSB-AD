import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#class PCA_Simple:
#    def __init__(self, n_components=4):
#        self.n_components = n_components
#
#    def fit(self, X):
#        self.scaler = StandardScaler()
#        Xs = self.scaler.fit_transform(X)
#
#        self.pca = PCA(n_components=self.n_components)
#        self.pca.fit(Xs)
#
#        return self
#
#    def decision_function(self, X):
#        Xs = self.scaler.transform(X)
#
        # full reconstruction (PC4 model)
#        X_hat = self.pca.inverse_transform(self.pca.transform(Xs))
#
#        residual = Xs - X_hat
#
#        spe = np.sum(residual ** 2, axis=1)
#        return spe

class PCA_Simple:
    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X):
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(Xs)

        # training reconstruction error (model scale)
        X_hat = self.pca.inverse_transform(self.pca.transform(Xs))
        residual = Xs - X_hat

        self.K = X.shape[1] #stores num of variables
        self.N = X.shape[0] #stores num of samples
        self.SSE_model = np.sum(residual ** 2) #computes sum of squared errors

        return self

    def decision_function(self, X):
        Xs = self.scaler.transform(X)

        X_hat = self.pca.inverse_transform(self.pca.transform(Xs))
        residual = Xs - X_hat

        SSE_i = np.sum(residual ** 2, axis=1) # for each sample

        A = self.n_components #num of retailed PCs
        K = self.K # num of variables
        N = self.N # training samples

        # DModX
        numerator = np.sqrt(SSE_i / (K - A)) #distance of this sample
        denominator = np.sqrt(self.SSE_model / ((N - A - 1) * (K - A))) #typical distance of training samples

        return numerator / (denominator + 1e-12)