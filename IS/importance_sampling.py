# -*- coding: utf-8 -*-
import warnings

import geatpy as ea
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import ndtri


class ImportanceSampling:
    """根据数据以及测试环境，初始化方法

    :param data: 采集到的场景真实数据
    :param env: 测试环境，包含env.test(scenario, sut) 方法
    :param name_list: 目前仅在绘制p(x)图像中用到，默认为"None"
    """

    def __init__(self, data, env, name_list=None):
        #assert (env.scenario_shape == data.shape[1])
        self.upper_bound = data.max(axis=0)
        self.lower_bound = data.min(axis=0)
        self.data = data
        self.env = env
        self.name_list = name_list
        self.theta_best = [0] * data.shape[1]
        self.m_best = [1] * data.shape[1]
        self.theta_q = [0] * data.shape[1]
        self.m_q = [1] * data.shape[1]

    def initialize(self, sut=None, iterate=10, num=200, nind=200, gene_q=True, plot_key=False):
        """拟合p(x),经过一定次数的遗传算法的迭代，得到q(x)。

        :param sut: 用于测试环境的规控器，默认为 None，即环境不需要额外的规控器。
        :param iterate: 遗传算法迭代次数
        :param num: 每次遗传算法迭代，进行测试的场景数。注意：遗传算法所需的测试场景总数 >= 迭代次数×每次迭代进行测试的场景数。这是因为，极小化交叉熵方法要求每次迭代的场景中存在危险场景。
        :param nind: 遗传算法种群规模，越大越好，但会增加计算负担
        :param gene_q: 是否迭代得到q(x)，默认为True。若设为False，则只拟合p(x)，不求解q(x)
        :param plot_key: 是否绘制p(x)拟合图像，默认为False。可在此功能基础上，增加保存图像的功能。
        """
        self._fit_distribution(plot_key=plot_key)
        print("fitted!")
        if gene_q:
            self._generate_q(sut, iterate, num, nind)

    @staticmethod
    def _best_fit_distribution(data, bins=500, ax=None):
        """从一系列分布中，找到拟合数据效果最好的分布。

        :param data: 数据
        :param bins: 比较拟合优度的细分度。例如：某参数取值范围为0-1，bins=500意味着，比较各个不同分布的效果时，会从0-1中均匀取500个点，比较拟合的分布与原始分布的差异。
        :param ax: 绘图使用。若传入ax，则会将能够拟合的所有分布曲线绘制于图上。
        :return:
            [最合适分布的名字， 最合适分布的参数]
        """
        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Distributions to check
        DISTRIBUTIONS = [
            st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2,
            st.cosine,
            st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife, st.fisk,
            st.foldcauchy, st.foldnorm, #st.frechet_r, st.frechet_l,
            st.gausshyper, st.gamma, st.gilbrat, st.gompertz, st.gumbel_r,
            st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.hypsecant, st.invgamma, st.invgauss,
            st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace, st.levy, st.levy_l,
            st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami, st.ncx2,
            st.ncf,
            st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist,
            st.reciprocal,
            st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm,
            st.tukeylambda,
            st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy
        ]  # st.levy_stable,st.dgamma,st.dweibull,st.genlogistic,st.genpareto,st.gennorm,st.genextreme,,st.gengamma,st.genexpon,st.genhalflogistic,st.halfgennorm,
        # Best holders
        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf

        # Estimate distribution parameters from data
        for distribution in DISTRIBUTIONS:
            #         clear_output()
            #         print("Trying:%s"%str(distribution))
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit.
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # if axis pass in add to plot
                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax)
                        # end
                    except Exception:
                        pass

                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse

            except Exception:
                pass

        return (best_distribution.name, best_params)

    @staticmethod
    def _make_pdf(dist, params, size=10000):
        """得到密度函数曲线，绘图使用的。

        :param dist: scipy.stats中的分布
        :param params: 分布对应的参数
        :param size: 绘图的精细程度，size=10000 意味着在分布曲线定义域中取10000个点。
        :return:
            pd.Series. 第二列为x, 第一列为对应的密度函数值
        """

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Get sane start and end points of distribution
        start = dist.ppf(
            0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc,
                       scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf_curve = pd.Series(y, x)

        return pdf_curve

    def _fit_data(self, y, plot_key=False, title=None):
        """调用_best_fit_distribution方法以及_make_pdf方法，完成单参数的拟合以及绘图任务

        :param y: 数据
        :param plot_key: 是否进行绘图
        :param title: 绘制图像的标题
        :return:
            [最合适分布的名字， 最合适分布的参数]
        """
        # Load data
        data = pd.Series(y)
        ax = None
        if plot_key:
            # Plot for comparison
            plt.subplot(2, self.data.shape[1], self.plot_key)
            ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5)
            # Save plot limits
            dataYLim = ax.get_ylim()
        # Find best fit distribution
        best_fit_name, best_fit_params = self._best_fit_distribution(
            data, 200, ax)
        best_dist = getattr(st, best_fit_name)

        if plot_key:
            # Update plots
            ax.set_ylim(dataYLim)
            ax.set_title(title)  # + u'\n All Fitted Distributions'
            # ax.set_xlabel(name)
            # ax.set_ylabel('Frequency')

            # Make PDF with best params
            pdf = self._make_pdf(best_dist, best_fit_params)

            # Display
            plt.subplot(2, self.data.shape[1], self.plot_key + self.data.shape[1])
            ax = pdf.plot(lw=2, label='PDF', legend=True)
            data.plot(kind='hist', bins=50, density=True,
                      alpha=0.5, label='Data', legend=True, ax=ax)

            param_names = (best_dist.shapes + ', loc, scale').split(
                ', ') if best_dist.shapes else ['loc', 'scale']
            param_str = ', '.join(['{}={:0.2f}'.format(k, v)
                                   for k, v in zip(param_names, best_fit_params)])
            dist_str = '{}({})'.format(best_fit_name, param_str)
            ax.set_title('Best fit distribution')
            # ax.set_xlabel(name)
            # ax.set_ylabel('Frequency')
            self.plot_key += 1

        return best_fit_name, best_fit_params

    def _fit_distribution(self, plot_key=False):
        """拟合p(x)

        p(x)的分布与参数，存储在ImportanceSampling对象的 fit_model和 param_model中

        :param plot_key: 是否绘制p(x)的拟合图像
        """
        self.plot_key = 1
        plt.figure(figsize=(4 * self.data.shape[1], 4))
        best_fit_name_list = []
        if self.name_list is None:
            name_list = range(self.data.shape[1])
        else:
            name_list = self.name_list
        for i, name in zip(range(self.data.shape[1]), name_list):
            y = self.data[:, i]
            best_fit_name, best_fit_params = self._fit_data(y, plot_key=plot_key, title=name)
            best_fit_name_list += [best_fit_name]
        # fit model
        fit_model = []
        param_model = []
        for i, dist_name in zip(range(self.data.shape[1]), best_fit_name_list):
            y = self.data[:, i]
            dist = getattr(st, dist_name)
            param = dist.fit(y)
            fit_model += [dist]
            param_model += [param]
        self.fit_model = fit_model
        self.param_model = param_model
        if plot_key:
            plt.tight_layout()
            plt.show()

    def _get_pdf_single(self, sample, ind, key='ori'):
        """得到样本中，单参数的概率密度函数值

        :param sample: 样本
        :param ind: 得到样本中哪一个参数的概率密度函数值
        :param key: 得到哪一个分布的概率密度函数值。'ori':原始分布,'q'：重要度分布,'best':在遗传算法迭代过程中用到
        :return:
            概率密度函数值
        """
        assert (self.param_model is not None)
        assert (key in ['ori', 'best', 'q'])
        if key == 'ori':
            theta_list = [0] * self.data.shape[1]
            m_list = [1] * self.data.shape[1]
        elif key == 'q':
            theta_list = self.theta_q
            m_list = self.m_q
        else:
            theta_list = self.theta_best
            m_list = self.m_best
        arg = self.param_model[ind][:-2]
        loc = self.param_model[ind][-2]
        scale = self.param_model[ind][-1]
        pdf = self.fit_model[ind].pdf(sample, loc=loc, scale=scale, *arg)
        return pdf * np.exp(theta_list[ind] * sample) / m_list[ind]

    def get_pdf_all(self, sample, key='ori'):
        """得到样本总体的概率密度函数值。

        由于假设各个参数是独立分布的，场景总体的概率密度函数值等于构成场景的各个参数的概率密度函数值的乘积。

        :param sample: 样本
        :param key: 得到分布的概率密度函数值。'ori':原始分布,'q'：重要度分布,'best':在遗传算法迭代过程中用到
        :return:
            样本总体的概率密度函数值
        """
        """得到样本总体的概率密度函数值。由于假设各个参数是独立分布的，场景总体的概率密度函数值等于构成场景的各个参数的概率密度函数值的乘积。

        :param sample:样本
        :param key:得到分布的概率密度函数值。'ori':原始分布,'q'：重要度分布,'best':在遗传算法迭代过程中用到
        :return:
            样本总体的概率密度函数值
        """
        assert (self.param_model is not None)
        assert (key in ['ori', 'best', 'q'])
        pdf = np.ones((sample.shape[0],))
        for i in range(sample.shape[1]):
            pdf_sub = self._get_pdf_single(sample[:, i], ind=i, key=key)
            pdf *= pdf_sub
        return pdf

    def generate_sample(self, sample_num, sut=None, key='ori'):
        """根据分布采样，得到样本，并且使用env测试样本，得到测试结果

        :param sample_num: 采样数量
        :param sut: 用于测试环境的规控器，默认为 None，即环境不需要额外的规控器。
        :param key: 得到哪一个分布的概率密度函数值。'ori':原始分布,'q'：重要度分布,'best':在遗传算法迭代过程中用到
        :return:
            [采样得到的样本，对应的测试结果]
        """
        assert (key in ['ori', 'best', 'q'])
        sample = np.zeros((sample_num, self.data.shape[1]))
        for i in range(self.data.shape[1]):
            a = self.upper_bound[i]
            b = self.lower_bound[i]
            test_x = np.linspace(a, b, 5000)
            p_test_x = self._get_pdf_single(test_x, i, key)
            k = p_test_x.max() * (b - a) + 1
            L = np.array([])
            while L.shape[0] < sample_num:
                L_sub = np.random.uniform(a, b, 2 * sample_num)
                u_sub = np.random.uniform(0, k / (b - a), 2 * sample_num)
                p = self._get_pdf_single(L_sub, i, key)
                keep_index = (p > u_sub)
                L_sub = L_sub[keep_index]
                L = np.append(L, L_sub)
            L = L[:sample_num]
            sample[:, i] = L
        result = self.env.test(sample)
        return sample, result

    def _update_m(self, key, num=10000):
        """根据分布的theta值，更新对应的M(归一化参数)。

        :param key: 更新哪一个分布的M。'q'：重要度分布,'best':在遗传算法迭代过程中用到
        :param num: 计算M的细分度。例如：某参数取值范围为0-1，num=1e4意味着，在0-1中均匀采样1e4个样本，用以估算M的值。
        """
        assert (key in ['best', 'q'])
        if key == 'q':
            theta_list = self.theta_q
            m_list = self.m_q
        else:
            theta_list = self.theta_best
            m_list = self.m_best
        for i in range(len(theta_list)):
            arg = self.param_model[i][:-2]
            loc = self.param_model[i][-2]
            scale = self.param_model[i][-1]
            x = np.linspace(self.lower_bound[i], self.upper_bound[i], num)
            pdf = self.fit_model[i].pdf(x, loc=loc, scale=scale, *arg)
            m_sub = (self.upper_bound[i] - self.lower_bound[i]) / \
                    num * np.sum(np.exp(theta_list[i] * x) * pdf)
            m_list[i] = m_sub
        if key == 'q':
            self.m_q = m_list
        else:
            self.m_best = m_list

    def _generate_q(self, sut, iterate=10, num=500, nind=200):
        """利用遗传算法，迭代求解q(x)

        本遗传算法使用了geatpy工具箱，可查阅相关的帮助文档。

        :param sut: 用于测试环境的规控器，默认为 None，即环境不需要额外的规控器。
        :param iterate: 遗传算法迭代次数
        :param num: 每次遗传算法迭代，进行测试的场景数。注意：遗传算法所需的测试场景总数 >= 迭代次数×每次迭代进行测试的场景数。这是因为，极小化交叉熵方法要求每次迭代的场景中存在危险场景。
        :param nind: 遗传算法种群规模，越大越好，但会增加计算负担
        """

        # 自定义问题类
        class MyProblem(ea.Problem):  # 继承Problem父类
            def __init__(self, M, Dim, IS):
                self.test_num = 0
                self.M = M
                self.IS = IS
                self.best_obj = 1
                name = 'DTLZ1'  # 初始化name（函数名称，可以随意设置）
                # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
                maxormins = [1] * M
                # 初始化varTypes（决策变量的类型，0：实数；1：整数）
                varTypes = np.array([0] * Dim)
                lb = [-1] * Dim  # 决策变量下界
                ub = [0] * Dim  # 决策变量上界
                lbin = [1] * Dim  # 决策变量下边界
                ubin = [1] * Dim  # 决策变量上边界
                # 调用父类构造方法完成实例化
                ea.Problem.__init__(self, name, M, maxormins,
                                    Dim, varTypes, lb, ub, lbin, ubin)

            def aimFunc(self, pop):  # 目标函数
                Vars = pop.Phen  # 得到决策变量矩阵
                res = []
                result_ori = np.array([0])
                while result_ori.sum() == 0:
                    sample_ori, result_ori = self.IS.generate_sample(
                        num, sut=sut, key='best')
                    self.test_num += sample_ori.shape[0]
                self.best_obj = (-np.log(self.IS.get_pdf_all(sample_ori, key='best')) * self.IS.get_pdf_all(
                    sample_ori, key='ori') / self.IS.get_pdf_all(sample_ori, key='best') * result_ori).mean()
                for parameters in Vars:
                    self.IS.theta_q = parameters.copy()
                    self.IS._update_m(key='q')
                    result = (-np.log(self.IS.get_pdf_all(sample_ori, key='q')) * self.IS.get_pdf_all(
                        sample_ori, key='ori') / self.IS.get_pdf_all(sample_ori, key='best') * result_ori).mean()
                    res += [result]
                print(self.test_num)
                pop.ObjV = np.array(res).reshape(-1, self.M)

        problem = MyProblem(1, self.data.shape[1], self)  # 生成问题对象
        Encoding = 'RI'  # 编码方式
        NIND = nind  # 种群规模
        Field = ea.crtfld(Encoding, problem.varTypes,
                          problem.ranges, problem.borders)  # 创建区域描述器
        # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        population = ea.Population(Encoding, Field, NIND)
        myAlgorithm = ea.soea_DE_rand_1_L_templet(
            problem, population)  # 实例化一个算法模板对象
        myAlgorithm.MAXGEN = 1  # 最大进化代数
        myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制过程动画）

        t = 0
        for i in range(iterate):
            #[population, obj_trace, var_trace] = myAlgorithm.run()  # 执行算法模板
            [obj_trace, var_trace] = myAlgorithm.run()  # 执行算法模板
            print("Ori:\t\t", problem.best_obj, "\t", self.theta_best)
            #获取obj_trace中的ObjV属性

            s = obj_trace.ObjV[0][0]
            theta = obj_trace.Phen[0]
            if s < problem.best_obj:
                problem.best_obj = s
                self.theta_best = theta
                self._update_m(key='best')
            print("This Time:\t", s, "\t", theta)
            print("After:\t\t", problem.best_obj, "\t", self.theta_best)
            # if obj_trace[:, 1] < problem.best_obj:
            #     problem.best_obj = obj_trace[:, 1]
            #     self.theta_best = var_trace.reshape(-1).tolist()
            #     self._update_m(key='best')
            # print("This Time:\t", obj_trace[:, 1],
            #       "\t", var_trace.reshape(-1).tolist())
            # print("After:\t\t", problem.best_obj, "\t", self.theta_best)
            #     theta_list = var_trace.reshape(-1).tolist()
            #     modified_M()
            t += myAlgorithm.passTime
        # population.save() # 把最后一代种群的信息保存到文件中
        # 输出结果
        print('最优的目标函数值为：%s' % (problem.best_obj))
        print('最优的控制变量值为：', self.theta_best)
        print('时间已过 %s 秒' % (t))
        self.theta_q = self.theta_best.copy()
        self._update_m(key='q')

    @staticmethod
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
    import sys
    import os
    if os.path.abspath('../') not in sys.path:
        sys.path.insert(0, os.path.abspath('../'))
    if os.path.abspath('./') not in sys.path:
        sys.path.insert(0, os.path.abspath('./'))
    from env.car_following_3 import CarFollowing
    from sut.idm import IDM

    data = pd.read_csv('car_following_3.csv')
    env = CarFollowing()
    sut = IDM(a_bound=5)

    name_list = data.columns.to_list()
    print("数据包含:", name_list)
    data = np.array(data)

    method = ImportanceSampling(data, env, name_list=name_list)
    method.initialize(sut, num=500)

    print("蒙特卡洛")
    half_width = 1
    rate = 1
    sample = np.zeros((0, data.shape[1]))
    result = np.array([])
    while half_width > 0.2:
        # print("sample:%d\trate:%f\tstd(result):%f\thalf_width:%f"%(sample.shape[0],rate,result.std(),half_width))
        sample_sub, result_sub = method.generate_sample(500, sut)
        sample = np.concatenate((sample, sample_sub), axis=0)
        result = np.append(result, result_sub)
        rate, half_width = method.cal_crash_and_width(result)
        print("sample:%d\trate:%f\tstd(result):%f\thalf_width:%f" %
              (sample.shape[0], rate, result.std(), half_width))
    print("sample:%d\trate:%f\tstd(result):%f\thalf_width:%f" %
          (sample.shape[0], rate, result.std(), half_width))

    print("重要度抽样")
    half_width = 1
    rate = 1
    sample = np.zeros((0, data.shape[1]))
    result = np.array([])
    while half_width > 0.2:
        # print("sample:%d\trate:%f\tstd(result):%f\thalf_width:%f"%(sample.shape[0],rate,result.std(),half_width))
        sample_sub, result_sub = method.generate_sample(50, sut, key='q')
        result_sub = result_sub * \
                     method.get_pdf_all(sample_sub, key='ori') / \
                     method.get_pdf_all(sample_sub, key='q')
        sample = np.concatenate((sample, sample_sub), axis=0)
        result = np.append(result, result_sub)
        rate, half_width = method.cal_crash_and_width(result)
    print("sample:%d\trate:%f\tstd(result):%f\thalf_width:%f" %
          (sample.shape[0], rate, result.std(), half_width))
