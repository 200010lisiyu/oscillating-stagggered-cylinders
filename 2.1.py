import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, least_squares
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
    # 读取Excel文件
df = pd.read_excel(r'lading3-10.xlsx')
df1 = pd.read_excel(r'kc357.xlsx')
# 提取数据
import logging
import warnings
import scipy.optimize
# 忽略特定的警告
warnings.filterwarnings("ignore", category=scipy.optimize.OptimizeWarning)
# 配置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    # 初始参数猜测
popt0= [-2.83,-1.0,  0.25,  0.14, 0.45,  0.74  ,0.75,4.02 ,0.25]#CD
popt0 = [ 3.36 ,-1.82,  0.23,  0.09 ,-0.16,  0.59 , 0.1 , 3.57,0.3]#Cm
popt0=[-7.66, -0.59,  0.19,  0.12,  0.81,  0.81,1.03 , 4.99,1.01]#e
popt0=[ 0, -1   ,0. , 0.  , 0. ,1 , 4. ,3]#CD


popt0 = [ 3.36 ,-1.82,  0.23,  0.09 ,0.16,  0.59 ,  3.57,0.3]#Cm
popt0=[-7.66, -0.59,  0.19,  0.12,  0.81,  0.81, 4.99,1.01]#e,f=0.81,e=0.81
#popt0= [-2.83,-1.0,  0.25,  0.14, 0.45,  0.74 ,4.02,0.25]#CD,f=0.74,e=0.45.h=4.02
popt0 = [ 3.36 ,-1.82,  0.23,  0.09 ,0.16,  0.59 ,  3.57,0.3]#Cm
popt0=[0, 0,  0.,  5,  1, 0]#e,f=0.81,e=0.81
#popt0= [0 ,-1.0 ,  0.,  15,  1 ,5]#CD,f=0.74,e=0.45.h=4.02
popt0= [0 ,-1.0 ,  0.25,  10,  0. ,0]#CD,f=0.74,e=0.45.h=4.02
popt0= [1 ,-1.0 ,  0.,  10,  0. ,-1]
# 忽略特定的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
    # 提取数据
x1_data = df.iloc[:, 5].values
x2_data = df.iloc[:, 6].values
x3_data = df.iloc[:, 3].values
#x4_data=np.sqrt((x1_data+1)*(x1_data+1)-0.25*(x2_data+1)*(x2_data+1))/(1+x1_data)
y_data = df.iloc[:,0].values*0.3333#Cd,因为之前没有取平均
#y_data = df.iloc[:,2].values#E
df1 = pd.read_excel(r'kc357.xlsx')
# 提取数据

# 提取数据
x1_data1 = df1.iloc[:, 4].values
x2_data1 = df1.iloc[:, 5].values
#x4_data1=np.sqrt((x1_data1+1)*(x1_data1+1)-0.25*(x2_data1+1)*(x2_data1+1))/(1+x1_data1)
x3_data1 = df1.iloc[:, 3].values
y_data1= df1.iloc[:, 0].values*0.3333#Cd,因为之前没有取平均
#y_data1 = df1.iloc[:,2].values#E

    # 创建训练集参数数组
params_array = np.column_stack((x1_data, x2_data, x3_data))

# 创建验证集参数数组
params_array1 = np.column_stack((x1_data1, x2_data1, x3_data1))
# 定义非线性模型函数
def nonlinear_func(x, *p):
    a,b, c,d,f,h=p
    x1 = x[:, 0]  # 提取x1数据
    x2 = x[:, 1]  # 提取x2数据
    x3 = x[:, 2]
    x4 = np.sqrt((x1+1)*(x1+1)-0.25*(x2+1)*(x2+1))/(1+x1)
    x5=0.5*(1+x2)/(1+x1)
    x6=x1/(2*x1+x2)
    x7=x2/(2*x1+x2)
    x8=x2*x4*(1+d/x3)+1
    x9=x1*(1+d/x3)+1
    x10=c*(h+a * (np.expm1(np.log1p(x3 - 1) *b)+1) )
    x11=f*(h+a * (np.expm1(np.log1p(x3 - 1) *b)+1))
    #return (h+a * (np.expm1(np.log1p(x3 - 1) *b)+1) )*pow((((1+ i/x3)*(g*x2+c*x1)+d) / ((1+ i/x3)*(g*x2+c*x1)+1) )* pow((x1 + e * x2 + 1) / (x1 + e * x2 + f*x5),2),2)
    
    return (h+a * (np.expm1(np.log1p(x3 - 1) *b)+1) )* (pow((1-x10/x8) *x4*(x1+x2*x5+1)/ (x1+x2*x5+f) ,2)+pow(x5*(1-x10/x9),2))#Cm
    #return (h+a * (np.expm1(np.log1p(x3 - 1) *b)+1) )*pow( (pow((1-x10/x8) *x4*(x1+x2*x5+1)/ (x1+x2*x5+f) ,2)+pow(x5*(1-x10/x9),2)),2)#Cm
    #return (g+a * (np.expm1(np.log1p(x3 - 1) * b)+1) )*(1-pow((x2+c*x1,-h) )* pow((x1 + e * x2 + 1) / (x1 + e * x2 + f*x5),2),2)
bounds = ([-np.inf , -np.inf,  0 , 0 , 0 ,0], [np.inf,np.inf,1,np.inf, 1,10])#应该是np.inf而不是None
# 定义一个函数来执行单个参数组合的拟合和评估
def fit_and_evaluate(params):
    try:
       #对训练集拟合
        popt1, _ = curve_fit(nonlinear_func, params_array, y_data, p0=params, absolute_sigma=False)
       #对验证集评估
        y_fit = nonlinear_func(params_array1, *popt1)
        mse1 = mean_squared_error(y_data1, y_fit)
        if(mse1<0.04):
           y_fit1 = nonlinear_func(params_array, *popt1)
           mse2 = mean_squared_error(y_data, y_fit1)
           print("较佳验证R²值：",mse1)
           print("较佳训练R²值：",mse2)
           print(params)
        return (params, mse1)
    except RuntimeError:
        # 如果拟合失败，返回一个非常大的MSE值
        return (params, float('inf'))
        
from skopt import gp_minimize
from skopt.space import Real

# 定义残差函数，用于 least_squares
def residuals0(params,x,y):
    # 计算残差：模型预测 - 实际值
    a=params[0]
    b = params[1]
    c = params[2]
    d=params[3]
    e=params[4]
    f=params[5]
    g=params[6]
    popt1=[a,b,c,d,e,f,g]
    #y1=nonlinear_func(x, a, b, c, d,e,f,g)
    popt2, _  = curve_fit(nonlinear_func, x, y, p0=popt1)
    y2 = nonlinear_func(params_array1, *popt2)
    reg_residuals = np.array([0, params[1]**2, params[2]**2, params[3]**2, params[4]**2, params[5]**2, params[6]**2])
    return mean_squared_error(y_data1, y2) +reg_residuals


# 定义包含正则化的雅可比矩阵函数
def jac_with_regularization(x,*params):
    a, b, c, d, e, f, g = params
    x1 = params_array[:, 0]
    x2 = params_array[:, 1]
    x3 = params_array[:, 2]
    
    # 计算非线性函数的各项
    exp_term = b * np.log1p(x3 - 1)
    term1 = a * (np.expm1(exp_term) + 1)
    term2 = (x1 + 1) / (x1 + 1 - c)
    term3 = (x2 + d) / (x2 + 1)
    term4 = (x1 + e * x2 + f) / (x1 + e * x2 + g)
    
    # 计算雅可比矩阵的每个元素
    da = term1 * term2 * term3 * term4
    db = a * np.expm1(exp_term) * term2 * term3 * term4 * np.log(x3 - 1)
    dc = -a * term1 * term2 * term3 * (1 / (x1 + 1 - c)**2)
    dd = a * term1 * (1 / (x2 + 1) - 1) * term2 * term4
    de = a * e * term1 * term2 * x2 * term3 * (1 / (x1 + e * x2 + g) - term4 / (x1 + e * x2 + g)**2)
    df = a * term1 * term2 * term3 * (1 / (x1 + e * x2 + g) - 2 * (x1 + e * x2 + f) / (x1 + e * x2 + g)**2)
    dg = -a * term1 * term2 * term3 * (1 / (x1 + e * x2 + g)**2)

    # 组合雅可比矩阵
    jacobian = np.array([da, db, dc, dd, de, df, dg]).T
    
    # 正则化系数
    reg_strength = 0.1
    # 创建与雅可比矩阵列数相同的正则化向量
    reg_vector = np.array([ reg_strength * a**2 ,reg_strength * b**2, reg_strength * c**2, reg_strength * d**2, reg_strength * e**2, reg_strength * f**2, reg_strength * g**2])

      # 将正则化向量扩展为二维数组，形状为 (1, 126)
    reg_matrix = reg_vector.reshape(1, -1)
    
    # 沿着列方向（axis=1）将正则化项添加到雅可比矩阵的每一行
    jacobian += reg_matrix
    return jacobian


def residuals(params0):
    # 计算残差：
    a=params0[0]
    b = params0[1]
    c = params0[2]
    d=params0[3]
    e=params0[4]
    f=params0[5]
    poptp=[a,b,c,d,e,f]
    #y1=nonlinear_func(x, a, b, c, d,e,f,g)
    try:
       #popt1, _ = curve_fit(nonlinear_func, params_array, y_data, p0=params0, absolute_sigma=False, maxfev=1000000000, method='trf', jac=jac_with_regularization,bounds=bounds)
       popt1, _ = curve_fit(nonlinear_func, params_array, y_data, p0=params0, absolute_sigma=False, maxfev=1000000000, method='dogbox',bounds=bounds)
       y2 = nonlinear_func(params_array1, *popt1)
       return  1-r2_score(y_data1, y2)
    except ValueError as e:
       try:
          #popt1, _ = curve_fit(nonlinear_func, params_array, y_data, p0=params0, absolute_sigma=False, maxfev=1000000000)
          #y2 = nonlinear_func(params_array1, *popt1)
          return  100
       except ValueError as e:   
          return 100
# 定义参数的搜索范围，这里只是示例，你需要根据实际情况设置
search_space = [
    Real(-10, 10, name='a'), Real(-3,0, name='b'), Real(0.01, 0.99, name='c'),
    Real(0.01, 10, name='d'),
    Real(0.01, 0.99, name='f'), Real(0.01, 5, name='h')
]

# 定义你的目标函数，用于贝叶斯优化
def objective(params):
    # 返回你希望优化库最小化的目标，这里是残差
    try:
        return residuals(params)
    except RuntimeError:
        return 100
def callback(xk):
    global n_calls
    n_calls += 1
    if n_calls % 1 == 0:  # 当 n_calls 是 20 的倍数时
        print(f"Function calls (multiples of 20): {n_calls}")
n_calls = 0  # 初始化调用次数            
            

if __name__ == '__main__':

    #mp.set_start_method('spawn')
    # 使用curve_fit对训练集进行非线性回归拟合
    popt, pcov = curve_fit(nonlinear_func, params_array, y_data, p0=popt0, method='dogbox', maxfev=1000000000)

    # 使用拟合参数进行验证集预测
    y_pred = nonlinear_func(params_array1, *popt)
    y_pred0 = nonlinear_func(params_array, *popt)
    #print(y_pred0 )
    #计算初始拟合测试集R²分数
    r20 = r2_score(y_data, y_pred0)
    mse0 = mean_squared_error(y_data, y_pred0)
    mae0 = mean_absolute_error(y_data, y_pred0)

    # 打印初始参数和验证集R²值
    print("初始参数：a={}, b={}, c={}, d={}, f={},  h={}".format(*popt0))
    print("初始拟合训练R²值：", r20)
    print("初始拟合训练MAE值：", mae0)
    print("初始拟合训练MSE值：", mse0)

    # 计算初始拟合参数的验证集R²分数
    r2 = r2_score(y_data1, y_pred)
    mse = mean_squared_error(y_data1, y_pred)
    mae = mean_absolute_error(y_data1, y_pred)

    # 打印最佳拟合参数和R²值
    print("初始拟合参数：a={}, b={}, c={}, d={}, f={},  h={}".format(*popt))
    print("初始拟合验证R²值：", r2)
    print("初始拟合验证MAE值：", mae)
    print("初始拟合验证MSE值：", mse)
    # 准备参数网格
    #param_grid = [
        #[a, b, c, d, e, f, g]
        #for a in np.linspace(0, 60, 21)
        #for b in np.linspace(-2, 2, 21)
        #for c in np.linspace(0, 1, 11)
        #for d in np.linspace(0, 1, 11)
        #for e in np.linspace(-10, -10, 21)
        #for f in np.linspace(-10, 10, 21)
        #for g in np.linspace(-10, 10, 21)
    #]
    x0=[-2.60476046 ,5,  0.25130317  ,0.14895042 , 0.27446488 , 0.79317938, 4.13111227,0]

    x1=[ 6.3229506 , -2.33870074 , 0.3071548 ,  0.05867618 , 1.43971221 ,  2.63424543,1,1]
    x2=[-7.526550745870324, -0.622497234461779, 0.5659857857824466, 0.6578410006055246,-0.3051383732973501,  0.7455242021438628,4.889761952113057]
    # 执行贝叶斯优化
    result = gp_minimize(
        objective,
        search_space,
        n_calls=14,
        n_points=10000000,
        n_jobs=50,
        callback=callback,
        x0=x0,
        #random_state=0
    )

    # 打印最优解和对应的函数值
    print("Optimal point:", result.x)
    print("Function value:", result.fun) 

    # 创建一个Pool，参数为CPU核心数
    #pool = Pool(processes=cpu_count())  # 这里硬编码为100，也可以使用 cpu_count()
# 输出CPU核心数
    #print("CPU核心数:", cpu_count())
    # 使用map方法并行执行fit_and_evaluate函数
    #results = pool.map(fit_and_evaluate, param_grid)

    # 关闭Pool，释放资源
    #pool.close()
    #pool.join()

    # 从结果中找到最小MSE的参数组合
    #min_mse = float('inf')
    #best_initial = None
    #for params, mse in results:
        #if mse < min_mse:
            #min_mse = mse
            #best_initial = params
    
    # 打印最佳初始参数
    best_initial=result.x
    print("最佳初始参数:", best_initial)

    # 如果找到更好的参数，则使用这些参数重新进行拟合
    if best_initial is not None:
        popt1, _ = curve_fit(nonlinear_func, params_array, y_data, p0=best_initial, maxfev=100000000)
        y_predbest0 = nonlinear_func(params_array, *popt1)
        print("最佳初始拟合参数:", popt1)
        r21 = r2_score(y_data, y_predbest0)
        mse1 = mean_squared_error(y_data, y_predbest0)
        mae1 = mean_absolute_error(y_data, y_predbest0)
        print("使用最佳初始参数对训练集的MSE:", mse1)
        print("R²值：", r21)
        print("MAE值：", mae1)
        print("MSE值：", mse1)
        y_predbest = nonlinear_func(params_array1, *popt1)
        r22 = r2_score(y_data1, y_predbest)
        mse2 = mean_squared_error(y_data1, y_predbest)
        mae2 = mean_absolute_error(y_data1, y_predbest)
        print("使用最佳初始参数对测试集的MSE:", mse2)
        print("R²值：", r22)
        print("MAE值：", mae2)
        print("MSE值：", mse2)
    with open('optimization_results.Cd.1.txt', 'w') as file:
    # 写入最优参数
        file.write(f'Cd Optimal parameters: {best_initial}\n')
    # 写入目标函数值
        file.write(f'Cd Function value at optimal parameters: {result.fun}\n')