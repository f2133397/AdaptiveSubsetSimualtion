import numpy as np
import pandas as pd
from important_samplingV0223 import ImportanceSampling
from cut_in_6_JCopy import CutIn
#from idm_cut_in import IDMCutIn
import scipy
from scipy.special import ndtri
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy import stats
import csv

class SubsetSimulation:
    def __init__(self, data):
        """
        根据数据拟合初始GMM分布
        :param data: 已有的驾驶数据 np.array
        """
        #拟合归一化工具
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        #将data进行归一化
        data_trans = self.scaler.transform(data)
        #通过bic指数，确定最适合的高斯混合模型组分数量
        best_components = 1
        best_bic = 999999999
        for i in range(15):
            gmm = GaussianMixture(n_components=i + 1, tol=0.0001, max_iter=10000)
            gmm.fit(data_trans)
            bic = gmm.bic(data_trans)
            if bic < best_bic:
                best_bic = bic
                best_components = i + 1
        #确定完毕最合适的高斯混合模型组分数量后，使用部分数据拟合高斯混合模型
        self.gmm = GaussianMixture(n_components=best_components, tol=0.0001, max_iter=10000)
        self.gmm.fit(data_trans)


    def train(self,sut):
        illiterate = 0
        self.N =5000
        self.p0 = 0.1
        self.Nc = 10
        #第一轮首先从GMM中采样
        samples = self.gmm.sample(self.N)[0]
        #将采样结果进行反归一化
        samples = self.scaler.inverse_transform(samples)
        #用sut测试samples得到测试结果
        results = sut.test(samples)
        #找到前0.1XN个危险样本对应的samples
        samples = samples[results.argsort()[:int(self.N * self.p0)]]
        #将samples进行归一化
        samples_seed = self.scaler.transform(samples)
        #b_threshold 等于samples_seed对应的最大result
        self.b_threshold = results[results.argsort()[:int(self.N * self.p0)]].max()
        #samples作为seeds，进行第二轮采样，采用MCMC的MMA算法,直到打到stop criteria
        while True:
            if illiterate >0:
                #对generate_samples_all归一化
                generate_samples_all = self.scaler.transform(generate_samples_all)
                #找到前0.1XN个危险样本对应的samples
                samples_seed = generate_samples_all[generate_samples_result_all.argsort()[:int(self.N * self.p0)]]
                #更新b_threshold
                self.b_threshold = generate_samples_result_all[generate_samples_result_all.argsort()[:int(self.N * self.p0)]].max()
                if self.b_threshold <0:
                    #统计generate_samples_result_all中小于0的样本比例
                    last_rate = (generate_samples_result_all<0).sum()/generate_samples_result_all.shape[0]
                    #计算最终碰撞率为last_rate乘以0.1的illiterate次方
                    self.collision_rate = last_rate * (0.1 ** (illiterate+1))

                    break

            generate_samples_all = []
            generate_samples_result_all = []
            #对samples_seed进行rosenblatt_transform
            samples_seed = self.rosenblatt_transform(samples_seed, self.gmm.weights_, self.gmm.means_, self.gmm.covariances_)
            #循环每个sample_seed，计算MCMC链，一条链生成Nc个
            #每个sample_seed对应的MCMC链，都是从sample_seed开始，生成Nc个sample
            for sample_seed_i in samples_seed:
                x_i = sample_seed_i
                #遍历每个维度，每个维度的变量建立一个提议分布
                #提议分布的均值为当前sample_seed_i的值，方差为1.07
                #proposal_dists = [stats.norm(loc=sample_seed_i[i], scale=1.07) for i in range(sample_seed_i.shape[0])]
                #遍历每个维度的提议分布，生成Nc个sample
                generate_samples = []
                generate_samples_result = []
                for i in range (self.Nc):
                    #提议分布的均值为x_i的值，方差为1.07
                    proposal_dists = [stats.norm(loc=x_i[i], scale=1.07) for i in range(x_i.shape[0])]
                    #先每个维度采样，得到一个generate_sample
                    generate_sample_i = []
                    for j ,proposal_dist in proposal_dists:
                        #在proposal_dist中采样得到一个值
                        generate_sample = proposal_dist.rvs()
                        #计算acceptance ratio
                        #计算generate_sample在原始分布也就是标准正态分布下的概率密度
                        generate_sample_pdf = stats.norm.pdf(generate_sample)
                        #计算对应的sample_seed_i中的值在原始分布也就是标准正态分布下的概率密度，index对应的是proposal_dist在proposal_dists中的索引
                        sample_seed_i_pdf = stats.norm.pdf(x_i[j])
                        #计算generate_sample在提议分布下的概率密度
                        generate_sample_pdf_q = proposal_dist.pdf(generate_sample)
                        #计算sample_seed_i在以generate_sample为均值，方差为1.07的提议分布下的概率密度
                        sample_seed_i_pdf_q = stats.norm.pdf(x_i[j], loc=generate_sample, scale=1.07)
                        #计算acceptance ratio
                        acceptance_ratio = (generate_sample_pdf * sample_seed_i_pdf_q) / (sample_seed_i_pdf * generate_sample_pdf_q)
                        #以min（1，acceptance_ratio）的概率接受generate_sample，否则接受sample_seed_i[j]
                        if np.random.rand() < min(1, acceptance_ratio):
                            accept_sample = generate_sample
                        else:
                            accept_sample = x_i[j]
                        #将accept_sample放入generate_sample_i中
                        generate_sample_i.append(accept_sample)
                    #对generate_sample_i进行反rosenblatt_transform
                    generate_sample_i_test = self.inverse_rosenblatt_transform(np.array(generate_sample_i), self.gmm.weights_, self.gmm.means_, self.gmm.covariances_)
                    #将generate_sample_i反归一化
                    generate_sample_i_test = self.scaler.inverse_transform(generate_sample_i_test)
                    #使用sut对generate_sample_i进行测试
                    result = sut.test(generate_sample_i_test)
                    #如果result小于b_threshold，即上一轮的samples_seed对应的测试结果阈值
                    if result <= self.b_threshold:
                        #将generate_sample_i放入generate_samples中
                        generate_samples.append(generate_sample_i_test)
                        generate_samples_result.append(result)
                        #将generate_sample_i_test添加输出到csv文件中，不覆盖原来的内容
                        with open('generate_samples.csv', 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(generate_sample_i_test)
                        #将result添加到csv文件中，不覆盖原来的内容
                        with open('generate_samples_result.csv', 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(result)
                        x_i = generate_sample_i
                    else:
                        generate_samples.append(generate_samples[-1])
                        #将generate_samples_result最后一位复制添加到generate_samples_result中
                        generate_samples_result.append(generate_samples_result[-1])

                        with open('generate_samples.csv', 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(generate_samples[-1])
                        with open('generate_samples_result.csv', 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(generate_samples_result[-1])
                    
                
                #将generate_samples和generate_samples_result放入generate_samples_all和generate_samples_result_all中
                generate_samples_all.append(generate_samples)
                generate_samples_result_all.append(generate_samples_result)
                illiterate+=1

    def rosenblatt_transform(gmm_samples, gmm_weights, gmm_means, gmm_covs):
        """
        将GMM样本应用于Rosenblatt变换
        :param gmm_weights: gmm component权重
        :param gmm_means: 均值
        :param gmm_covs: 协方差矩阵
        """
        transformed_samples = np.zeros_like(gmm_samples)
        for i in range(gmm_samples.shape[0]):
            sample = gmm_samples[i, :]
            cdf_values = []
            #对K个变量依次进行变换
            for j in range(gmm_means.shape[1]):
                cdf_value = 0
                for k , weight in enumerate(gmm_weights):
                    mean = gmm_means[k, j]
                    cov = gmm_covs[k, j, j]
                    #求解对应的cdf value
                    cdf_value += weight * scipy.stats.norm.cdf(sample[j], loc=mean, scale=cov)

                cdf_values.append(cdf_value)
            
            #将每个变量转换为标准正态分布
            transformed_sample = scipy.stats.norm.ppf(cdf_values)
            transformed_samples[i] = transformed_sample

        return transformed_samples

    def inverse_rosenblatt_transform(transformed_samples, gmm_weights, gmm_means, gmm_covs):
        """
        将Rosenblatt变换的样本应用于反变换
        :param gmm_weights: gmm component权重
        :param gmm_means: 均值
        :param gmm_covs: 协方差矩阵
        """
        inverse_cdfs = [stats.norm.ppf for _ in range(8)]

        #反转Roseblatt变换
        original_samples = np.zeros_like(transformed_samples)
        for i in range (transformed_samples.shape[0]):
            for j in range(transformed_samples.shape[1]):
                original_samples[i, j] = inverse_cdfs[j](transformed_samples[i, j])

        return original_samples













