import numpy as np
import pandas as pd
from important_samplingV0223 import ImportanceSampling
from cut_in_6_JCopy import CutIn
#from idm_cut_in import IDMCutIn
import scipy
from scipy.special import ndtri

def computeGMMpdf(BGMM, sample,scaler=None):
    if scaler is not None:
        sample = scaler.transform(sample)
    for i in range(BGMM.weights_.shape[0]):
        weights = BGMM.weights_[i]
        mean = BGMM.means_[i, :]
        cov = BGMM.covariances_[i, :, :]
        mn = scipy.stats.multivariate_normal(mean=mean, cov=cov)
        if i == 0:
            res = mn.pdf(sample) * weights
        else:
            res += mn.pdf(sample) * weights
    return res

def cal_crash_and_width(result, z_rate=0.95):
    """计算事故率以及相对半宽

    :param result: 测试环境得到的测试结果。注意，对于重要度抽样方法计算得到的result，需要先抵消不同分布带来的误差。即 测试结果 * p(x) / q(x)，再传入本方法中。
    :param z_rate: 置信度
    :return:
        rate,half_width
        事故率与相对半宽
    """
    z = ndtri(1 - (1 - z_rate) / 2)

    rate = result.mean()
    if rate != 0:
        # 置信区间为 rate + ndtri(0.9)*S/n
        # l_r = z * np.sqrt((1 - rate) / rate / result.shape[0])
        l_r = z * result.std() / np.sqrt(result.shape[0]) / rate
    else:
        l_r = 1
    return rate, l_r

if __name__ == "__main__":
    data = pd.read_csv("cutin_diswithin50.csv", sep=',')
    data = data.loc[:, ['v1', 'v2', 'v3', 'dis1x', 'dis2x', 'dis1y']]
    data = data.loc[:, ['v3', 'v2', 'v1', 'dis2x', 'dis1x', 'dis1y']]
    #data = data.loc[:, ['v3', 'v2', 'v1', 'dis2x', 'dis1x', 'dis1y']]
    imp = ImportanceSampling(np.array(data))
    env = CutIn()
    imp.fit(env)
    half_width = 1
    rate = 1
    sample = np.zeros((0, data.shape[1]))
    result = np.array([])
    #遍历测试data中的样本，计算事故率
    # for i in range(data.shape[0]):
    #     result = np.append(result, env.test(np.array(data.iloc[i, :]).reshape(1, -1), metric='minimum_adjusted_TTC'))
    #将result大于0的数字统计为1，小于0的数字统计为0
    # result[result > 0] = 1
    # result[result <= 0] = 0
    # rate, half_width = cal_crash_and_width(result)
    # print(rate)

    while half_width > 0.2:
        # #从重要性分布gmm_i中抽取样本
        # sample = gmm_i.sample(100)[0]
        # #计算抽取的样本在gmm_i中的概率密度
        # p = gmm_i.score_samples(sample)
        # p = np.exp(p)
        # #计算抽取的样本在imp.gmm中的概率密度
        # q = imp.gmm.score_samples(sample)
        # q = np.exp(q)
        # #p为重要性采样的概率密度，q为原始分布的概率密度
        # temp_result = np.array([])
        # for i in range(sample.shape[0]):
        #     #将sample反归一化
        #     sample[i, :] = imp.scaler.inverse_transform(sample[i, :].reshape(1, -1))
        #     temp_result = np.append(temp_result, env.test(sample[i, :].reshape(1, -1), metric='minimum_adjusted_TTC'))
        # #将temp_result大于0的数字统计为1，小于0的数字统计为0
        # temp_result[temp_result > 0] = 1
        # temp_result[temp_result <= 0] = 0
        # #样本在重要性分布中的概率密度为p，样本在原始分布中的概率密度为q，计算抵消不同分布带来的误差
        # temp_result = temp_result * q / p
        # result = np.append(result, temp_result)
        # rate, half_width = cal_crash_and_width(result)
        # print(rate, half_width)


        #从重要性分布imp.gmm_i中抽样
        sample = imp.gmm_i.sample(100)[0]
        #计算抽取的样本在gmm_i中的概率密度
        q = computeGMMpdf(imp.gmm_i, sample)

        #计算抽取的样本在imp.gmm中的概率密度
        p = imp.gmm.score_samples(sample)
        p = np.exp(p)
        #q为重要性采样的概率密度，p为原始分布的概率密度
        temp_result = np.array([])
        for i in range(sample.shape[0]):
            temp_result = np.append(temp_result, env.test(sample[i, :].reshape(1, -1), metric='danger'))
        #依据q和p计算抵消不同分布带来的误差
        temp_result = temp_result * p / q
        print(p / q)
        result = np.append(result, temp_result)
        rate, half_width = cal_crash_and_width(result)
        print(rate, half_width)



        #从样本原始分布imp.gmm中抽取样本，做蒙特卡洛测试
        sample = imp.gmm.sample(100)[0]
        temp_result = np.array([])
        for i in range(sample.shape[0]):
            #将sample反归一化
            #sample[i, :] = imp.scaler.inverse_transform(sample[i, :].reshape(1, -1))
            temp_result = np.append(temp_result, env.test(sample[i, :].reshape(1, -1), metric='danger'))
        #将temp_result大于0的数字统计为1，小于0的数字统计为0
        # temp_result[temp_result > 0] = 1
        # temp_result[temp_result <= 0] = 0
        result = np.append(result, temp_result)
        rate, half_width = cal_crash_and_width(result)
        print(rate, half_width)






        
        








