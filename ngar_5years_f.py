#!/usr/bin/python
# -*- coding: utf-8 -*-
# File Name: ngar.py
# Author: Changsheng Zhang
# mail: zhangcsxx@gmail.com
# Created Time: Mon Dec  7 00:26:38 2015

#########################################################################

import numpy as np
import copy
import math
from time import ctime
import random
import datetime
import matplotlib.pyplot as plt
import copy
import time
from pandas import DataFrame,Series
import pandas as pd
from scipy import stats
import sys
import threading

class NGAR():
    def __init__(self,x_df,y_df,ykey,mu_mean,mu_lambda_star,burnin,numofits,every):
        # x_list size: T*p; y_list
        self.x_list,self.y_list = self.Load_y_x(x_df, y_df, ykey)
        if len(self.y_list) < 36: #去掉不足3年基金
            # print 'less than 3 years'
            return None
        self.mu_mean = mu_mean
        self.mu_lambda_star = mu_lambda_star
        self.numofits = numofits
        self.every = every
        self.burnin = burnin
        self.ykey = ykey

        #initial value
        self.T = len(self.x_list)
        self.p = len(self.x_list[0])
        self.rho = 0.97*np.ones(self.p)
        self.rho_beta = 0.97*np.ones(self.p)
        self.lambda_ = self.mu_lambda_star*np.ones(self.p)
        self.mu = self.mu_mean*np.ones(self.p)
        self.delta = self.rho/(1-self.rho)*self.lambda_/self.mu

        self.start_samples = 500
        self.start_adap = 1000

        self.new_beta = np.zeros((self.T,self.p))
        self.psi = np.ones((self.T,self.p))
        self.kappa = np.zeros((self.T,self.p))
        for ii in xrange(self.p):
            for jj in xrange(self.T):
                self.kappa[jj][ii] = np.random.poisson(self.delta[ii]*self.psi[jj][ii])

        self.kappa_sigma_sq = np.ones(self.T)
        self.lambda_sigma = 3
        self.mu_sigma = 0.03
        self.rho_sigma = 0.95
        self.sigma_sq = self.mu_sigma*np.ones(self.T)
        self.lambda_star =1.0
        self.mu_star = 1.0

        self.psi_sd = 0.01*np.ones((self.T,self.p))
        self.log_lambda_sd = np.log(0.1)*np.ones(self.p)
        self.log_mean_sd = np.log(0.1)*np.ones(self.p)
        self.log_scale = 0.5*np.log(2.4**2/4)*np.ones(self.p)
        self.log_rho_beta_sd = np.log(0.1)*np.ones(self.p)
        self.log_rho_sd = np.log(0.1)*np.ones(self.p)
        self.log_rho_sigma_sd = np.log(0.01)
        self.log_lambda_sigma_sd = np.log(0.001)
        self.log_gamma_sigma_sq_sd = np.log(0.001)
        self.log_scale_sigma_sq = 0.5*np.log(2.4**2/3)
        self.log_sigma_sq_sd = np.log(0.001)*np.ones(self.T)
        self.log_kappa_q = 4.0/3*np.ones((self.T-1,self.p))
        self.log_kappa_sigma_sqq = 4.0/3*np.ones(self.T)
        self.mu_gamma_sd = 0.003
        self.v_gamma_sd = 0.003

        self.kappa_accept = 0
        self.kappa_count =0
        self.kappa_lambda_sigma_accept = 0
        self.kappa_sigma_sq_count =0
        self.sigma_sq_param_accept = np.zeros(self.p)
        self.sigma_sq_param_count = np.zeros(self.p)
        self.psi_param_accept = np.zeros(self.p)
        self.psi_param_count = np.zeros(self.p)
        self.mu_gamma_accept = 0
        self.mu_gamma_count = 0
        self.v_gamma_accept = 0
        self.v_gamma_count = 0
        self.sigma_sq1_accept = np.zeros(self.T)
        self.sigma_sq1_count = np.zeros(self.T)
        self.number_of_iteration = self.burnin + self.every*self.numofits

        self.sum_1 = np.zeros((4,self.p))
        self.sum_2 = np.zeros((10,self.p))
        self.sum_1_sigma_sq = np.zeros(3)
        self.sum_2_sigma_sq = np.zeros(6)
        self.limit = 0.9999

        self.hold_psi = np.zeros((self.T,self.p,self.numofits))
        self.hold_beta =   np.zeros((self.T,self.p,self.numofits))
        self.hold_sigma_sq =   np.zeros((self.T,self.numofits))
        self.hold_lambda =  np.zeros((self.p,self.numofits))
        self.hold_mu =   np.zeros((self.p,self.numofits))
        self.hold_rho_beta =  np.zeros((self.p,self.numofits))
        self.hold_rho =  np.zeros((self.p,self.numofits))
        self.hold_lambda_sigma = np.zeros(self.numofits)
        self.hold_mu_sigma = np.zeros(self.numofits)
        self.hold_rho_sigma = np.zeros(self.numofits)
        self.hold_lambda_star = np.zeros(self.numofits)
        self.hold_mu_star = np.zeros(self.numofits)

        # beta 初始值
        self.mean_kf,self.var_kf,loglike = self.KalmanFilter(self.x_list,self.y_list,self.psi,self.sigma_sq,self.rho_beta)

        chol_star = np.linalg.cholesky(self.var_kf[:,:,self.T-1])

        self.new_beta[self.T-1]=self.mean_kf[:,self.T-1] + np.dot(chol_star,np.random.normal(size=len(chol_star[0])))
        # 测试时间
        start_time = datetime.datetime.now()
        for jj in xrange(self.T-2,-1,-1):
            Gkal = np.diag(self.rho_beta*np.sqrt(1.0*self.psi[jj+1]/self.psi[jj]))
            invQ = np.diag(1.0/(1-self.rho_beta**2)*1.0/self.psi[jj+1])
            var_fb = np.linalg.inv(np.linalg.inv(self.var_kf[:,:,jj])+ np.dot(np.dot(np.transpose(Gkal),invQ),Gkal))
            mean_fb = np.dot(var_fb,np.dot(np.linalg.inv(self.var_kf[:,:,jj]),self.mean_kf[:,jj])+ np.dot(np.dot(np.transpose(Gkal),invQ),self.new_beta[jj+1]))

            chol_star = np.linalg.cholesky(var_fb)
            self.new_beta[jj] = mean_fb + np.dot(chol_star,np.array(np.random.normal(size=len(chol_star[0]))))

        self.beta = copy.deepcopy(self.new_beta[:,0:self.p])

        # print self.beta
        # print self.beta.shape



        self.check_star = 1

        #注意，迭代次数的下标是从1开始
        for it in xrange(1,self.number_of_iteration+1):
            # iteration
            # if it%100==0:
            #     print "start %s th iteration:\n" %it
            #     #print self.beta
            if self.check_star ==1:
                # run iteration
                self.UpdatePsi(it)
                self.UpdateKappa(it)
                self.UpdateSigmaSq(it)
                self.UpdateKappaSigmaSq(it)
            else:
                pass
                # print "check star = 0. \n"
            self.UpdateTheta(it)
            self.UpdateMuStar(it)
            self.UpdateLambdaStar(it)
            self.UpdateOutput(it,self.burnin,self.every)
            # if it% 500==0:
            #     end_time = datetime.datetime.now()
            #     print (end_time-start_time).seconds
        self.Plot()
        print "done."

    def KalmanFilter(self,x_list,y_list,psi,sigma_sq,rho_beta):
        p = len(x_list[0])
        T = len(y_list)
        mean_kf = np.zeros((p,T))
        var_kf = np.zeros((p,p,T))

        aminus = np.zeros(p)
        pminus = np.diag(psi[0])
        x_star_list = x_list[0]
        e = y_list[0] - np.dot(x_star_list,aminus)
        invF = 1.0/(sigma_sq[0]+ np.dot(np.dot(x_star_list,pminus),x_star_list))
        mean_kf[:,0] = aminus + np.dot(pminus,x_star_list)*invF*e

        var_kf[:,:,0] = pminus - np.dot(np.dot(np.dot(pminus,invF*x_star_list.reshape(p,1)),[x_star_list]),pminus)
        loglike = -0.5*e**2*invF+ 0.5*np.log(invF)

        for ii in xrange(1,T):
            x_star_list = x_list[ii]
            Q = np.diag((1-rho_beta**2)*psi[ii])
            Gkal = np.diag(rho_beta*np.sqrt(1.0*np.exp(np.log(psi[ii])-np.log(psi[ii-1]))))
            aminus = np.dot(Gkal,mean_kf[:,ii-1])
            pminus = np.dot(np.dot(Gkal,var_kf[:,:,ii-1]),np.transpose(Gkal))+Q
            e = y_list[ii] - np.dot(x_star_list,aminus)
            invF = 1.0/(sigma_sq[ii]+ np.dot(np.dot(x_star_list,pminus),x_star_list))
            mean_kf[:,ii] = aminus +np.dot(pminus,x_star_list)*invF*e
            var_kf[:,:,ii] = pminus -np.dot(np.dot(np.dot(pminus,invF*x_star_list.reshape(p,1)),[x_star_list]),pminus)
            loglike = loglike -0.5*e**2*invF+0.5*np.log(invF)

        return mean_kf,var_kf,loglike


    # it 是当前迭代次数
    def UpdatePsi(self,it):
        for ii in xrange(self.T):
            for jj in xrange(self.p):
                new_psi = self.psi[ii][jj]*np.exp(self.psi_sd[ii][jj]*np.random.normal())
                # new_loglike = 0
                # loglike = 0

                if ii ==0:
                    loglike = (self.lambda_[jj]-1)*np.log(self.psi[0][jj])-1.0*self.lambda_[jj]*self.psi[0][jj]/self.mu[jj]
                    pnmean = self.psi[ii][jj]*self.delta[jj]
                    loglike = loglike - pnmean + self.kappa[ii][jj]*np.log(pnmean)-0.5*np.log(self.psi[0][jj])-0.5*self.beta[0][jj]**2/self.psi[0][jj]
                    var1 = self.psi[1][jj]*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(1.0*self.psi[1][jj]/self.psi[0][jj])*self.beta[0][jj]
                    loglike = loglike -0.5*(self.beta[1][jj]-mean1)**2/var1

                    new_loglike = (self.lambda_[jj]-1)*np.log(new_psi)-self.lambda_[jj]*new_psi/self.mu[jj]
                    pnmean = new_psi*self.delta[jj]
                    new_loglike = new_loglike - pnmean+self.kappa[ii][jj]*np.log(pnmean)-0.5*np.log(new_psi)-0.5*self.beta[0][jj]**2/new_psi
                    var1 = self.psi[1][jj]*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(self.psi[1][jj]/new_psi)*self.beta[0][jj]
                    new_loglike = new_loglike - 0.5*(self.beta[1][jj]-mean1)**2/var1

                elif ii<self.T-1:
                    lam1 = self.lambda_[jj]+self.kappa[ii-1][jj]
                    gam1 = self.lambda_[jj]/self.mu[jj] +self.delta[jj]
                    loglike = (lam1-1)*np.log(self.psi[ii][jj])-gam1*self.psi[ii][jj]-self.psi[ii][jj]*self.delta[jj]+1.0*self.kappa[ii][jj]*np.log(self.psi[ii][jj]*self.delta[jj])
                    var1 = self.psi[ii][jj]*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(self.psi[ii][jj]/self.psi[ii-1][jj])*self.beta[ii-1][jj]
                    loglike = loglike - 0.5*np.log(var1)-0.5*(self.beta[ii][jj]-mean1)**2/var1
                    var1 = self.psi[ii+1][jj]*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(self.psi[ii+1][jj]/self.psi[ii][jj])*self.beta[ii][jj]
                    loglike = loglike -0.5*(self.beta[ii+1][jj]-mean1)**2/var1

                    lam1 = self.lambda_[jj]+ self.kappa[ii-1][jj]
                    gam1 = self.lambda_[jj]/self.mu[jj] +self.delta[jj]
                    new_loglike = (lam1-1)*np.log(new_psi)- gam1*new_psi-new_psi*self.delta[jj]+self.kappa[ii][jj]*np.log(new_psi*self.delta[jj])
                    var1 = new_psi*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(new_psi/self.psi[ii-1][jj])*self.beta[ii-1][jj]
                    new_loglike = new_loglike -0.5*np.log(var1) - 0.5*(self.beta[ii][jj]-mean1)**2/var1
                    var1 = self.psi[ii+1][jj]*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(self.psi[ii+1][jj]/new_psi)*self.beta[ii][jj]
                    new_loglike = new_loglike-0.5*(self.beta[ii+1][jj]-mean1)**2/var1

                else:
                    lam1 = self.lambda_[jj]+self.kappa[ii-1][jj]
                    gam1 = self.lambda_[jj]/self.mu[jj] + self.delta[jj]
                    loglike =(lam1-1)*np.log(self.psi[ii][jj])-gam1*self.psi[ii][jj]
                    var1 = self.psi[ii][jj]*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(self.psi[ii][jj]/self.psi[ii-1][jj])*self.beta[ii-1][jj]
                    loglike = loglike -0.5*np.log(var1)-0.5*(self.beta[ii][jj]-mean1)**2/var1

                    lam1 = self.lambda_[jj]+self.kappa[ii-1][jj]
                    gam1 = self.lambda_[jj]/self.mu[jj]+self.delta[jj]
                    new_loglike = (lam1-1)*np.log(new_psi)-gam1*new_psi
                    var1 = new_psi*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(new_psi/self.psi[ii-1][jj])*self.beta[ii-1][jj]
                    new_loglike = new_loglike -0.5*np.log(var1)-0.5*(self.beta[ii][jj]-mean1)**2/var1

                log_accept = new_loglike-loglike +np.log(new_psi)-np.log(self.psi[ii][jj])
                accept =1
                if np.isnan(log_accept) or np.isinf(log_accept):
                    accept =0
                elif log_accept <0:
                    accept = np.exp(log_accept)

                self.psi_sd[ii][jj] = self.psi_sd[ii][jj] + (accept-0.3)/(it**0.6)

                if np.random.random() <accept:
                    self.psi[ii][jj] = new_psi

    def UpdateKappa(self, it):
        for ii in xrange(self.T-1):
            for jj in xrange(self.p):

                new_kappa = self.kappa[ii][jj]+(2*np.ceil(2*np.random.random())-3)*(np.random.geometric(1.0/(1+np.exp(self.log_kappa_q[ii][jj])))-1)

                if new_kappa < 0:
                    accept = 0
                else:
                    lam1 = self.lambda_[jj] + 1.0*self.kappa[ii][jj]
                    gam1 = self.lambda_[jj]/self.mu[jj] + self.delta[jj]
                    loglike = lam1*np.log(gam1) - math.lgamma(lam1)+(lam1-1)*np.log(self.psi[ii+1][jj])
                    pnmean = self.psi[ii][jj] * self.delta[jj]
                    loglike = loglike + 1.0*self.kappa[ii][jj]*np.log(pnmean) - math.lgamma(1.0*self.kappa[ii][jj]+1)

                    lam1 = self.lambda_[jj] + 1.0*new_kappa
                    gam1 = self.lambda_[jj]/self.mu[jj] + self.delta[jj]
                    new_loglike = lam1*np.log(gam1) - math.lgamma(lam1)+(lam1-1)*np.log(self.psi[ii+1][jj])
                    pnmean = self.psi[ii][jj]*self.delta[jj]
                    new_loglike = new_loglike + new_kappa*np.log(pnmean)-math.lgamma(1.0*new_kappa+1)
                    log_accept = new_loglike - loglike
                    accept =1
                    if np.isnan(log_accept) or np.isinf(log_accept):
                        accept =0
                    elif log_accept <0:
                        accept = np.exp(log_accept)

                self.kappa_accept = self.kappa_accept + accept
                self.kappa_count = self.kappa_count +1

                if np.random.random() < accept:
                    self.kappa[ii][jj] = new_kappa
                self.log_kappa_q[ii][jj] = self.log_kappa_q[ii][jj] + 1.0/it**0.55*(accept-0.3)
                # if np.isnan(self.kappa[ii][jj]) or np.isreal(self.kappa[ii][jj]) ==False:
                #     stop

    def UpdateSigmaSq(self,it):
        for ii in xrange(self.T):
            chi1 = (self.y_list[ii] - sum(self.x_list[ii]*self.beta[ii]))**2
            if ii == 0:
                lam1 = self.kappa_sigma_sq[ii] + self.lambda_sigma -0.5
                psi1 = 2*(self.lambda_sigma/self.mu_sigma+self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma)
            elif ii ==self.T-1:
                lam1 = self.kappa_sigma_sq[ii-1]+self.lambda_sigma -0.5
                psi1 = 2*(self.lambda_sigma/self.mu_sigma+self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma)
            else:
                lam1= self.kappa_sigma_sq[ii]+self.kappa_sigma_sq[ii-1]+self.lambda_sigma-0.5
                psi1 = 2*(self.lambda_sigma/self.mu_sigma+2*self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma)
            new_sigma_sq = self.sigma_sq[ii]*np.exp(np.exp(self.log_sigma_sq_sd[ii])*np.random.normal())
            loglike = (lam1-1)*np.log(self.sigma_sq[ii])-0.5*chi1/self.sigma_sq[ii]-0.5*psi1*self.sigma_sq[ii]
            new_loglike = (lam1-1)*np.log(new_sigma_sq)-0.5*chi1/new_sigma_sq-0.5*psi1*new_sigma_sq
            log_accept = new_loglike - loglike +np.log(new_sigma_sq)-np.log(self.sigma_sq[ii])

            accept =1
            if np.isnan(log_accept) or np.isinf(log_accept):
                accept =0
            elif log_accept <0:
                accept = np.exp(log_accept)

            self.sigma_sq1_accept[ii] = self.sigma_sq1_accept[ii] +accept
            self.sigma_sq1_count[ii] = self.sigma_sq1_count[ii] +1

            if np.random.random() < accept :
                self.sigma_sq[ii] = new_sigma_sq
            self.log_sigma_sq_sd[ii] = self.log_sigma_sq_sd[ii] +1.0/it**0.55*(accept-0.3)

    def UpdateKappaSigmaSq(self,it):
        for ii in xrange(self.T-1):
            new_kappa_sigma_sq = self.kappa_sigma_sq[ii]+(2*np.ceil(2*np.random.random())-3)*(np.random.geometric(1.0/(1+np.exp(self.log_kappa_sigma_sqq[ii])))-1)

            if new_kappa_sigma_sq <0:
                accept = 0
            else:
                lam1 = 1.0*self.lambda_sigma + self.kappa_sigma_sq[ii]
                gam1 = 1.0*self.lambda_sigma/self.mu_sigma + 1.0*self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma
                loglike = lam1*np.log(gam1)-math.lgamma(lam1)+(lam1-1)*np.log(self.sigma_sq[ii+1])
                pnmean = self.sigma_sq[ii]*self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma
                loglike = loglike + self.kappa_sigma_sq[ii]*np.log(pnmean)- math.lgamma(1.0*self.kappa_sigma_sq[ii]+1)

                lam1 = 1.0*self.lambda_sigma + new_kappa_sigma_sq
                gam1 = 1.0*self.lambda_sigma/self.mu_sigma + self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma
                new_loglike = lam1*np.log(gam1)-math.lgamma(lam1)+(lam1-1)*np.log(self.sigma_sq[ii+1])
                pnmean = self.sigma_sq[ii]*self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma
                new_loglike = new_loglike + new_kappa_sigma_sq*np.log(pnmean)-math.lgamma(1.0*new_kappa_sigma_sq+1)
                log_accept = new_loglike - loglike
                accept =1
                if np.isnan(log_accept) or np.isinf(log_accept):
                    accept = 0
                elif log_accept <0:
                    accept = np.exp(log_accept)

            self.kappa_lambda_sigma_accept = self.kappa_lambda_sigma_accept + accept
            self.kappa_sigma_sq_count = self.kappa_sigma_sq_count +1
            if np.random.random()<accept :
                self.kappa_sigma_sq[ii] = new_kappa_sigma_sq
            self.log_kappa_sigma_sqq[ii] = self.log_kappa_sigma_sqq[ii]+1.0/it**0.55*(accept-0.3)

            # if np.isnan(self.kappa_sigma_sq[ii]) or np.isreal(self.kappa_sigma_sq[ii]) == 0:
            #     stop

    def UpdateTheta(self, it):
        z_star = np.random.random(self.p) < 5.0/self.p

        y_star = self.y_list - (self.x_list[:,z_star ==0 ]*self.beta[:,z_star ==0]).sum(axis=1)
        x_star = copy.deepcopy(self.x_list[:,z_star==1])
        psi_star = copy.deepcopy(self.psi[:,z_star==1])
        kappa_star = copy.deepcopy(self.kappa[:,z_star==1])

        self.mean_kf,self.var_kf,loglike = self.KalmanFilter(x_star,y_star,psi_star,self.sigma_sq,self.rho_beta[z_star==1])
        xi_star = np.array([np.log(self.lambda_sigma),np.log(self.mu_sigma),np.log(self.rho_sigma)-np.log(1-self.rho_sigma)])

        if it <100:
            new_xi_star = xi_star +np.array([np.exp(self.log_lambda_sigma_sd),np.exp(self.log_gamma_sigma_sq_sd),np.exp(self.log_rho_sigma_sd)])*np.random.normal(size=3)
        else:
            var_star_1 =(np.array([[self.sum_2_sigma_sq[0],self.sum_2_sigma_sq[1],self.sum_2_sigma_sq[3]],[self.sum_2_sigma_sq[1],self.sum_2_sigma_sq[2],self.sum_2_sigma_sq[4]],[self.sum_2_sigma_sq[3],self.sum_2_sigma_sq[4],self.sum_2_sigma_sq[5]]])- np.dot(np.transpose([self.sum_1_sigma_sq]),[self.sum_1_sigma_sq])*1.0/it )*1.0/(it-1)
            new_xi_star = xi_star + np.dot(np.linalg.cholesky(np.exp(self.log_scale_sigma_sq)*var_star_1),np.random.normal( size= 3))

        new_lambda_sigma = 1.0*np.exp(new_xi_star[0])
        new_mu_sigma = 1.0*np.exp(new_xi_star[1])
        new_rho_sigma = 1.0*np.exp(new_xi_star[2])/(1+np.exp(new_xi_star[2]))

        if new_rho_sigma > self.limit:
            accept =0
        else:
            new_sigma_sq = copy.deepcopy(self.sigma_sq)
            new_kappa_sigma_sq = copy.deepcopy(self.kappa_sigma_sq)

            new_sigma_sq[0] = self.sigma_sq[0]*new_mu_sigma/self.mu_sigma
            if new_lambda_sigma >self.lambda_sigma:
                try:
                    new_sigma_sq[0] = new_sigma_sq[0]+ np.random.gamma(new_lambda_sigma-self.lambda_sigma,scale= new_mu_sigma/new_lambda_sigma)
                except:
                    new_sigma_sq[0] = float('nan')
            else:
                try:
                    new_sigma_sq[0] = new_sigma_sq[0]*np.random.beta(new_lambda_sigma,self.lambda_sigma-new_lambda_sigma)
                except:
                    new_sigma_sq[0] = float('nan')

            for ii in xrange(1,self.T):
                old_mean = 1.0*self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma*self.sigma_sq[ii-1]
                new_mean = 1.0*new_rho_sigma/(1-new_rho_sigma)*new_lambda_sigma/new_mu_sigma*new_sigma_sq[ii-1]
                if new_mean > old_mean:
                    try:
                        new_kappa_sigma_sq[ii-1] = self.kappa_sigma_sq[ii-1]+ np.random.poisson(new_mean-old_mean)
                    except:
                        new_kappa_sigma_sq[ii-1] = float('nan')
                else:
                    try:
                        new_kappa_sigma_sq[ii-1] = np.random.binomial(self.kappa_sigma_sq[ii-1],1.0*new_mean/old_mean)
                    except:
                        new_kappa_sigma_sq[ii-1] = float('nan')
                old_lam = 1.0*self.kappa_sigma_sq[ii-1]+ self.lambda_sigma
                old_gam = 1.0*self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma+self.lambda_sigma/self.mu_sigma
                new_lam = 1.0*new_kappa_sigma_sq[ii-1]+new_lambda_sigma
                new_gam = 1.0*new_rho_sigma/(1-new_rho_sigma)*new_lambda_sigma/new_mu_sigma+new_lambda_sigma/new_mu_sigma
                new_sigma_sq[ii] = self.sigma_sq[ii]*old_gam/new_gam


                if new_lam > old_lam:
                    try:
                        new_sigma_sq[ii] = new_sigma_sq[ii] + np.random.gamma(new_lam-old_lam,scale = 1.0/new_gam)
                    except:
                        new_sigma_sq[ii] = float('nan')
                else:
                    try:
                        new_sigma_sq[ii] = new_sigma_sq[ii]*np.random.beta(new_lam,old_lam-new_lam)
                    except:
                        new_sigma_sq[ii] = float('nan')
            new_mean_kf,new_var_kf,new_loglike = self.KalmanFilter(x_star,y_star,psi_star,new_sigma_sq,self.rho_beta[z_star==1])

            log_accept = new_loglike -loglike+3*(np.log(new_lambda_sigma)-np.log(self.lambda_sigma))-1.0*(new_lambda_sigma-self.lambda_sigma)+np.log(new_mu_sigma)-np.log(self.mu_sigma)
            log_accept = log_accept-(1+0.5)*np.log(1.0+new_mu_sigma)+(1+0.5)*np.log(1.0+self.mu_sigma)
            log_accept = log_accept+np.log(1.0/self.rho_sigma+1.0/(1-self.rho_sigma))-np.log(1.0/new_rho_sigma+1.0/(1-new_rho_sigma))
            log_accept = log_accept +40*0.95*(np.log(new_rho_sigma)-np.log(self.rho_sigma))+40*0.05*(np.log(1-new_rho_sigma)-np.log(1-self.rho_sigma))

            accept =1
            if np.isnan(log_accept) or np.isinf(log_accept):
                accept =0
            elif log_accept<0:
                accept = np.exp(log_accept)

        self.sigma_sq_param_accept = self.sigma_sq_param_accept +accept
        self.sigma_sq_param_count = self.sigma_sq_param_count +1

        if np.random.random() < accept:
            self.lambda_sigma = copy.deepcopy(new_lambda_sigma)
            self.mu_sigma = copy.deepcopy(new_mu_sigma)
            self.rho_sigma = copy.deepcopy(new_rho_sigma)
            self.kappa_sigma_sq = copy.deepcopy(new_kappa_sigma_sq)
            self.sigma_sq = copy.deepcopy(new_sigma_sq)
            loglike = copy.deepcopy(new_loglike)
            self.mean_kf = copy.deepcopy(new_mean_kf)
            self.var_kf = copy.deepcopy(new_var_kf)

        if it <100:
            self.log_lambda_sigma_sd = self.log_lambda_sigma_sd+1.0/(it**0.55)*(accept-0.3)
            self.log_gamma_sigma_sq_sd = self.log_gamma_sigma_sq_sd +1.0/it**0.55*(accept-0.3)
            self.log_rho_sigma_sd = self.log_rho_sigma_sd+1.0/it**0.55*(accept-0.3)
        else:
            self.log_scale_sigma_sq = self.log_scale_sigma_sq +1.0/(it-99)**0.55*(accept-0.3)
        x0 = np.log(self.lambda_sigma)
        x1 = np.log(self.mu_sigma)
        x2 = np.log(self.rho_sigma) -np.log(1-self.rho_sigma)
        self.sum_1_sigma_sq[0] = self.sum_1_sigma_sq[0] +x0
        self.sum_1_sigma_sq[1] = self.sum_1_sigma_sq[1] +x1
        self.sum_1_sigma_sq[2] = self.sum_1_sigma_sq[2] +x2
        self.sum_2_sigma_sq[0] = self.sum_2_sigma_sq[0] + x0**2
        self.sum_2_sigma_sq[1] = self.sum_2_sigma_sq[1] + x0*x1
        self.sum_2_sigma_sq[2] = self.sum_2_sigma_sq[2] + x1**2
        self.sum_2_sigma_sq[3] = self.sum_2_sigma_sq[3] + x0*x2
        self.sum_2_sigma_sq[4] = self.sum_2_sigma_sq[4] + x1*x2
        self.sum_2_sigma_sq[5] = self.sum_2_sigma_sq[5] + x2**2


        counter = -1

        for jj in xrange(self.p):
            if z_star[jj]== True:
                counter = counter +1
                xi_star = np.array([np.log(self.lambda_[jj]),np.log(self.mu[jj]),np.log(self.rho[jj])-np.log(1-self.rho[jj]),np.log(self.rho_beta[jj])-np.log(1-self.rho_beta[jj])])

                if it < self.start_adap:
                    new_xi_star = xi_star +np.array([np.exp(self.log_lambda_sd[jj]),np.exp(self.log_mean_sd[jj]),np.exp(self.log_rho_sd[jj]),np.exp(self.log_rho_beta_sd[jj])])*np.random.normal(size=4)
                else:
                    sxx = np.array([[self.sum_2[0][jj],self.sum_2[1][jj],self.sum_2[3][jj],self.sum_2[6][jj]],[self.sum_2[1][jj],self.sum_2[2][jj],self.sum_2[4][jj],self.sum_2[7][jj]],[self.sum_2[3][jj],self.sum_2[4][jj],self.sum_2[5][jj],self.sum_2[8][jj]],[self.sum_2[6][jj],self.sum_2[7][jj],self.sum_2[8][jj],self.sum_2[9][jj]]])

                    var_star_1 = 1.0*(sxx-np.dot(np.transpose([self.sum_1[:,jj]]),[self.sum_1[:,jj]])*1.0/(it-self.start_samples))/(it-self.start_samples-1)
                    new_xi_star = xi_star + np.dot(np.linalg.cholesky(np.exp(self.log_scale[jj])*var_star_1),np.random.normal(size=4))

                new_lambda = np.exp(new_xi_star[0])
                new_mu = np.exp(new_xi_star[1])
                new_rho = np.exp(new_xi_star[2])/(1+np.exp(new_xi_star[2]))
                new_rho_beta = copy.deepcopy(self.rho_beta)
                new_rho_beta[jj] = np.exp(new_xi_star[3])/(1.0+np.exp(new_xi_star[3]))
                new_delta = 1.0*new_rho/(1-new_rho)*new_lambda/new_mu


                if new_rho_beta[jj] >self.limit or new_rho >self.limit:
                    accept =0
                else:
                    new_psi_star = copy.deepcopy(psi_star)
                    new_kappa_star = copy.deepcopy(kappa_star)

                    new_psi_star[0][counter] = 1.0*psi_star[0][counter]*new_mu/self.mu[jj]
                    if new_lambda > self.lambda_[jj]:
                        try:
                            new_psi_star[0][counter] = new_psi_star[0][counter]+ np.random.gamma(new_lambda-self.lambda_[jj],scale= 1.0*new_mu/new_lambda)
                        except:
                            new_psi_star[0][counter] = float('nan')
                    else:
                        try:
                            new_psi_star[0][counter] = 1.0*new_psi_star[0][counter]*np.random.beta(new_lambda,self.lambda_[jj]-new_lambda)
                        except:
                            new_psi_star[0][counter] = float('nan')
                    for ii in xrange(1,self.T):
                        old_mean = 1.0*self.delta[jj]*psi_star[ii-1][counter]
                        new_mean = 1.0*new_delta*new_psi_star[ii-1][counter]

                        if new_mean > old_mean:
                            try:
                                new_kappa_star[ii-1][counter] = kappa_star[ii-1][counter]+np.random.poisson(new_mean-old_mean)
                            except:
                                new_kappa_star[ii-1][counter] = float('nan')
                        else:
                            try:
                                new_kappa_star[ii-1][counter] = np.random.binomial(kappa_star[ii-1][counter],1.0*new_mean/old_mean)
                            except:
                                new_kappa_star[ii-1][counter] = float('nan')

                        old_lam = 1.0*kappa_star[ii-1][counter]+ self.lambda_[jj]
                        old_gam = 1.0*self.delta[jj] + self.lambda_[jj]/self.mu[jj]

                        new_lam = 1.0*new_kappa_star[ii-1][counter] + new_lambda
                        new_gam = 1.0*new_delta + new_lambda/new_mu

                        new_psi_star[ii][counter] = 1.0*psi_star[ii][counter]*old_gam/new_gam

                        if new_lam > old_lam:
                            try:
                                new_psi_star[ii][counter] = new_psi_star[ii][counter] + np.random.gamma(new_lam-old_lam,scale= 1.0/new_gam)
                            except:
                                new_psi_star[ii][counter] = float('nan')
                        else:
                            try:
                                new_psi_star[ii][counter] = new_psi_star[ii][counter]*np.random.beta(new_lam,old_lam-new_lam)
                            except:
                                new_psi_star[ii][counter] = float('nan')
                    new_mean_kf,new_var_kf,new_loglike = self.KalmanFilter(x_star,y_star,new_psi_star,self.sigma_sq,new_rho_beta[z_star==1])
                    log_accept = new_loglike - loglike

                    if jj ==0:
                        log_accept = log_accept +2.0*np.log(new_lambda)-2.0*np.log(self.lambda_[jj])-4.0*np.log(0.5+new_lambda)+4.0*np.log(0.5+self.lambda_[jj])+np.log(new_mu)-np.log(self.mu[jj])
                        log_accept = log_accept -1.5*np.log(1.0+new_mu)+1.5*np.log(1.0+self.mu[jj])
                    else:
                        log_accept = log_accept +2.0*np.log(new_lambda)- 2.0*np.log(self.lambda_[jj])- 4.0*np.log(0.5+new_lambda) + 4.0*np.log(0.5+self.lambda_[jj])
                        log_accept = log_accept + self.lambda_star*(np.log(new_mu)-np.log(self.mu[jj])) - 1.0*self.lambda_star/self.mu_star*(new_mu-self.mu[jj])

                    log_accept = log_accept +np.log(1.0/self.rho[jj]+1.0/(1.0 -self.rho[jj]))- np.log(1.0/new_rho+1.0/(1.0 -new_rho))
                    log_accept = log_accept+80*0.97*(np.log(new_rho)-np.log(self.rho[jj]))+80*0.03*(np.log(1-new_rho)-np.log(1-self.rho[jj]))
                    log_accept = log_accept + np.log(1.0/self.rho_beta[jj]+1.0/(1-self.rho_beta[jj]))-np.log(1.0/new_rho_beta[jj]+1.0/(1.0 -new_rho_beta[jj]))
                    log_accept = log_accept + 80*0.97*(np.log(new_rho_beta[jj])-np.log(self.rho_beta[jj]))+80*0.03*(np.log(1.0 -new_rho_beta[jj])-np.log(1 -self.rho_beta[jj]))

                    accept =1
                    if np.isnan(log_accept) or np.isinf(log_accept):
                        accept = 0
                    elif log_accept<0:
                        accept = np.exp(log_accept)

                self.psi_param_accept[jj] = self.psi_param_accept[jj] + accept
                self.psi_param_count[jj] = self.psi_param_count[jj] +1

                if np.random.random() < accept:
                    self.lambda_[jj] = new_lambda
                    self.mu[jj] = new_mu
                    self.rho[jj] = new_rho
                    self.rho_beta[jj] = new_rho_beta[jj]
                    self.delta[jj] = new_delta
                    psi_star[:,counter] = copy.deepcopy(new_psi_star[:,counter])
                    kappa_star[:,counter] = copy.deepcopy(new_kappa_star[:,counter])
                    loglike = new_loglike
                    self.mean_kf = copy.deepcopy(new_mean_kf)
                    self.var_kf = copy.deepcopy(new_var_kf)

                if it < self.start_adap:
                    self.log_lambda_sd[jj] = self.log_lambda_sd[jj] +1.0/it**0.55*(accept-0.3)
                    self.log_mean_sd[jj] = self.log_mean_sd[jj] +1.0/it**0.55*(accept-0.3)
                    self.log_rho_sd[jj] = self.log_rho_sd[jj]+ 1.0/it**0.55*(accept-0.3)
                    self.log_rho_beta_sd[jj] = self.log_rho_beta_sd[jj] +1.0/it**0.55*(accept-0.3)

                else:
                    self.log_scale[jj] = self.log_scale[jj] + 1.0/(it-99)**0.55*(accept-0.3)


        if it >= self.start_samples:
            x0 = np.log(self.lambda_)
            x1 = np.log(self.mu)
            x2 = np.log(self.rho)- np.log(1.0 -self.rho)
            x3 = np.log(self.rho_beta) - np.log(1.0 -self.rho_beta)

            self.sum_1[0] = self.sum_1[0] + x0
            self.sum_1[1] = self.sum_1[1] + x1
            self.sum_1[2] = self.sum_1[2] + x2
            self.sum_1[3] = self.sum_1[3] + x3

            self.sum_2[0] = self.sum_2[0] + x0**2
            self.sum_2[1] = self.sum_2[1] + x0*x1
            self.sum_2[2] = self.sum_2[2] + x1**2
            self.sum_2[3] = self.sum_2[3] + x0*x2
            self.sum_2[4] = self.sum_2[4] + x1*x2
            self.sum_2[5] = self.sum_2[5] + x2**2
            self.sum_2[6] = self.sum_2[6] + x0*x3
            self.sum_2[7] = self.sum_2[7] + x1*x3
            self.sum_2[8] = self.sum_2[8] + x2*x3
            self.sum_2[9] = self.sum_2[9] + x3**2


        self.psi[:,z_star==1] = copy.deepcopy(psi_star)
        self.kappa[:,z_star==1] = copy.deepcopy(kappa_star)

        new_beta = np.zeros((self.T,sum(z_star)))

        self.check_star =1
        chol_star,check = self.Chol(self.var_kf[:,:,self.T-1])
        if check ==0:
            new_beta[self.T-1] = self.mean_kf[:,self.T-1]+ np.dot(chol_star,np.random.normal(size=len(chol_star[0])))
        else:
            self.check_star = 0

        for ii in xrange(self.T-2,-1,-1):
            Gkal = np.diag(self.rho_beta[z_star ==1]*np.sqrt(1.0*psi_star[ii+1]/psi_star[ii]))
            invQ = np.diag(1.0/(1-self.rho_beta[z_star==1]**2)*1.0/psi_star[ii+1])
            try:
                inv_var_kf = np.linalg.inv(self.var_kf[:,:,ii])
            except:
                # print "var kf det",np.linalg.det(self.var_kf[:,:,ii])
                # print self.var_kf[:,:,ii]
                var_len = len(self.var_kf[:,:,ii])
                inv_var_kf = np.zeros((var_len,var_len))
                for rr in xrange(var_len):
                    for uu in xrange(var_len):
                        inv_var_kf[rr][uu] = float('inf')
            try:

                var_fb = np.linalg.inv(inv_var_kf+np.dot(np.dot(np.transpose(Gkal),invQ),Gkal))
            except:
                # print "var fb det", np.linalg.det(inv_var_kf+np.dot(np.dot(np.transpose(Gkal),invQ),Gkal))
                var_len = len(inv_var_kf)
                var_fb = np.zeros((var_len, var_len))
                for rr in xrange(var_len):
                    for uu in xrange(var_len):
                        var_fb[rr][uu] = float('inf')

            mean_fb = np.dot(var_fb, np.dot(inv_var_kf, self.mean_kf[:, ii])+np.dot(np.dot(np.transpose(Gkal),invQ), new_beta[ii+1]))
            chol_star, check = self.Chol(var_fb)
            if check == 0:
                new_beta[ii] = mean_fb + np.dot(chol_star, np.random.normal(size=len(chol_star[0])))
            else:
                self.check_star = 0

        if self.check_star == 1 and sum(sum(np.isnan(new_beta))) == 0 and self.CheckBeta(new_beta):
            #print "update beta"
            self.beta[:, z_star == 1] = copy.deepcopy(new_beta)
        else:
            pass
            # print "not update beta"

    def CheckBeta(self, beta):
        for ii in xrange(len(beta)):
            for jj in xrange(len(beta[ii])):
                if beta[ii][jj] > 100:
                    return 0
        return 1

    def Chol(self, matrix):
        try:
            chol_star = np.linalg.cholesky(matrix)
            check = 0
        except:
            chol_star = np.array([1])
            check = 1
        return chol_star, check

    def UpdateMuStar(self, it):
        new_mu_star = 1.0*self.mu_star*np.exp(self.mu_gamma_sd*np.random.normal())
        log_accept = 1.0*(self.p-1)*self.lambda_star*(np.log(self.mu_star)-np.log(new_mu_star))
        log_accept = log_accept-self.lambda_star*(1.0/new_mu_star-1.0/self.mu_star)*sum(self.mu[1:])
        log_accept = log_accept + np.log(new_mu_star)-np.log(self.mu_star)-3.0*np.log(new_mu_star+self.mu_mean)+3.0*np.log(self.mu_star+self.mu_mean)

        accept = 1
        if np.isnan(log_accept) or np.isinf(log_accept):
            accept = 0
        elif log_accept < 0:
            accept = np.exp(log_accept)
        self.mu_gamma_accept = self.mu_gamma_accept + accept
        self.mu_gamma_count = self.mu_gamma_count + 1

        if np.random.random() < accept:
            self.mu_star = new_mu_star
        new_mu_gammma_sd = self.mu_gamma_sd +1.0/it**0.5*(accept-0.3)
        if new_mu_gammma_sd > 10**(-3) and new_mu_gammma_sd < 10**3:
            self.mu_gamma_sd = new_mu_gammma_sd

    def UpdateLambdaStar(self,it):
        new_lambda_star = self.lambda_star*np.exp(self.v_gamma_sd*np.random.normal())
        log_accept = (self.p-1)*(new_lambda_star*np.log(new_lambda_star/self.mu_star)-self.lambda_star*np.log(self.lambda_star/self.mu_star))
        log_accept = log_accept -(self.p-1)*(math.lgamma(new_lambda_star)-math.lgamma(self.lambda_star))
        log_accept = log_accept +(new_lambda_star-self.lambda_star)*sum(np.log(self.mu[1:]))-(new_lambda_star-self.lambda_star)/self.mu_star*sum(self.mu[1:])
        log_accept = log_accept +np.log(new_lambda_star)-np.log(self.lambda_star)-1.0/self.mu_lambda_star*(new_lambda_star-self.lambda_star)
        accept =1
        if np.isnan(log_accept) or np.isinf(log_accept):
            accept =0
        elif log_accept <0:
            accept = np.exp(log_accept)

        self.v_gamma_accept = self.v_gamma_accept +accept
        self.v_gamma_count = self.v_gamma_count +1
        if np.random.random() < accept:
            self.lambda_star = new_lambda_star
        new_gamma_sd = self.v_gamma_sd +1.0/it**0.5*(accept-0.3)

        if new_gamma_sd >10**(-3) and new_gamma_sd <10**3:
            self.v_gamma_sd = new_gamma_sd

    def UpdateOutput(self,it,burnin,every):
        if it > burnin and (it-burnin)%every ==0:
            self.hold_beta[:,:,(it-burnin)/every-1] = copy.deepcopy(self.beta)
            #print self.beta
            self.hold_psi[:,:,(it-burnin)/every-1] = copy.deepcopy(self.psi)
            self.hold_sigma_sq[:,(it-burnin)/every-1] = copy.deepcopy(self.sigma_sq)
            self.hold_lambda[:,(it-burnin)/every-1] = copy.deepcopy(self.lambda_)
            self.hold_mu[:,(it-burnin)/every-1] = copy.deepcopy(self.mu)
            self.hold_rho[:,(it-burnin)/every-1] = copy.deepcopy(self.rho)
            self.hold_rho_beta[:,(it-burnin)/every-1] = copy.deepcopy(self.rho_beta)
            self.hold_lambda_sigma[(it-burnin)/every-1] = copy.deepcopy(self.lambda_sigma)
            self.hold_mu_sigma[(it-burnin)/every-1] = copy.deepcopy(self.mu_sigma)
            self.hold_rho_sigma[(it-burnin)/every-1] = copy.deepcopy(self.rho_sigma)
            self.hold_lambda_star[(it-burnin)/every-1] = copy.deepcopy(self.lambda_star)
            self.hold_mu_star[(it-burnin)/every-1] = copy.deepcopy(self.mu_star)

    def Plot(self):

        final_beta_median = np.median(self.hold_beta, axis=2)
        final_beta_mean = np.mean(self.hold_beta,axis=2)
        beta_95_lower = np.percentile(self.hold_beta, 2.5,axis=2)
        beta_95_upper = np.percentile(self.hold_beta, 97.5,axis=2)
        beta_90_lower = np.percentile(self.hold_beta, 5,axis=2)
        beta_90_upper = np.percentile(self.hold_beta, 95,axis=2)
        outfile = open('output/interval/fund_'+self.ykey+'_95_lower.txt', 'w')
        np.savetxt(outfile, beta_95_lower)
        outfile.close()
        outfile = open('output/interval/fund_'+self.ykey+'_95_upper.txt', 'w')
        np.savetxt(outfile, beta_95_upper)
        outfile.close()
        outfile = open('output/interval/fund_'+self.ykey+'_90_lower.txt', 'w')
        np.savetxt(outfile, beta_90_lower)
        outfile.close()
        outfile = open('output/interval/fund_'+self.ykey+'_90_upper.txt', 'w')
        np.savetxt(outfile, beta_90_upper)
        outfile.close()
        #Convergence Diagnostic

        # 存储中位数的alpha
        outfile = open('output/fund_'+self.ykey+'_alpha_median.txt', 'w')
        np.savetxt(outfile, final_beta_median[:,0])
        outfile.close()
        # 存储平均数的alpha
        outfile = open('output/fund_'+self.ykey+'_alpha_mean.txt', 'w')
        np.savetxt(outfile, final_beta_mean[:,0])
        outfile.close()

        outfile = open('output/fund_'+self.ykey+'_beta.txt', 'w')
        for ii in xrange(self.p):
            # plt.figure(ii+1)
            # plt.plot(final_beta_median[:,ii])
            # plt.title("beta"+str(ii+1))
            # plt.xlabel("T")
            # plt.ylabel("beta")
            # png_name = "output/"+"fund_"+self.ykey+"beta"+str(ii+1)+".png"
            # plt.savefig(png_name, dpi=100)
            # plt.close()
            if ii>0:
                outfile.write("beta"+str(ii+1)+'\n')
                np.savetxt(outfile, final_beta_median[:,ii])
        outfile.close()

            # outfile.write('# beta'+str(ii+1)+'\n')

#从Excel基金数据读取Y，ykey是sheetname，指哪只基金
    def Load_y_x(self,x_df,y_df,ykey):
        y = DataFrame(y_df[ykey])
        y.index=pd.to_datetime(y_df.index,format = '%Y-%m-%d')
        y = y.resample('M',how='first',kind='period')
        y = y['2011':] #取2011年之后的

        yx = pd.merge(y,x_df,left_index=True,right_index=True)
        yx = yx.dropna(axis=0,how='any')
        y_list = np.array(yx.ix[:,0])
        x_list = np.array(yx.ix[:,1:6])
        
        return x_list,y_list

def run(y_names,start_index,end_index,x_df,y_df):
    for ii in range(start_index,end_index):
        ykey = y_names[ii]
        print "runone"
        gdp_case = NGAR(x_df,y_df,ykey,0.1,0.1,2000,5000,5)

if __name__ == "__main__":

    x_path = "data/factor2.csv"
    y_path = "data/fund_return.csv"

    y_df = pd.read_csv(y_path,index_col=0)
    y_df.index = pd.date_range(start='1962/01/01',end='2016/04/01',freq='M')
    x_df = pd.read_csv(x_path,index_col=0)
    x_df.index = pd.to_datetime(x_df.index,format = '%m/%d/%y')
    x_df = x_df.resample('M',how='first',kind='period')
    x_df = x_df['2011':] #只取2011年之后的数据

    y_names = y_df.columns
    counter = 0
    threads = []

    # ykey = '167'    #基金代码
    print "start:\n"

    t1 = ctime()
    #不同的服务器可以分配不同的区间(start,end)
    run(y_names, 4000, 4127, x_df, y_df)

    t2 = ctime()
    print t1
    print t2